"""
model.py — Architecture Encoder-Decoder pour l'image captioning.

  • CustomCNN        : CNN from scratch avec skip connections
  • BahdanauAttention: Attention additive
  • DecoderRNN       : LSTM avec attention + scheduled sampling
"""

import torch
import torch.nn as nn


# ======================================================================
#  BLOC CONVOLUTIF RESIDUEL
# ======================================================================
class ResidualConvBlock(nn.Module):
    """Conv3x3 + BN + ReLU + Conv3x3 + BN + skip + ReLU + MaxPool."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.pool(self.relu(out + identity))


# ======================================================================
#  CNN FROM SCRATCH
# ======================================================================
class CustomCNN(nn.Module):
    """CNN from scratch: 5 blocs residuels, sortie (B, 49, embed_size)."""
    def __init__(self, embed_size):
        super().__init__()
        self.features = nn.Sequential(
            ResidualConvBlock(3, 64),
            ResidualConvBlock(64, 128),
            ResidualConvBlock(128, 256),
            ResidualConvBlock(256, 512),
            ResidualConvBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.embed = nn.Linear(512, embed_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images):
        f = self.pool(self.features(images))
        B, C, H, W = f.shape
        return self.embed(f.permute(0, 2, 3, 1).reshape(B, H * W, C))


# ======================================================================
#  ATTENTION (Bahdanau)
# ======================================================================
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.enc_att = nn.Linear(enc_dim, att_dim)
        self.dec_att = nn.Linear(dec_dim, att_dim)
        self.full_att = nn.Linear(att_dim, 1)

    def forward(self, enc_out, dec_hid):
        e = self.full_att(torch.tanh(
            self.enc_att(enc_out) +
            self.dec_att(dec_hid).unsqueeze(1)
        )).squeeze(2)
        alpha = torch.softmax(e, dim=1)
        return (enc_out * alpha.unsqueeze(2)).sum(dim=1), alpha


# ======================================================================
#  DECODER LSTM
# ======================================================================
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,
                 dropout=0.5, attention_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_dropout = nn.Dropout(0.3)
        self.attention = BahdanauAttention(
            embed_size, hidden_size, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_size * 2, hidden_size)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def _init_hidden(self, enc_out):
        m = enc_out.mean(dim=1)
        return torch.tanh(self.init_h(m)), torch.tanh(self.init_c(m))

    def forward(self, features, captions, ss_prob=0.0):
        cap_in = captions[:, :-1]
        B, T = cap_in.shape
        h, c = self._init_hidden(features)
        outputs = torch.zeros(B, T, self.vocab_size,
                              device=features.device)
        alphas = torch.zeros(B, T, features.size(1),
                             device=features.device)
        embed_t = self.embed_dropout(self.embedding(cap_in[:, 0]))

        for t in range(T):
            ctx, a = self.attention(features, h)
            h, c = self.lstm_cell(
                torch.cat([embed_t, ctx], 1), (h, c))
            logits = self.fc(self.dropout(h))
            outputs[:, t, :] = logits
            alphas[:, t, :] = a
            if t < T - 1:
                if ss_prob > 0.0 and self.training:
                    use_pred = (torch.rand(B, device=features.device)
                                < ss_prob)
                    pred_w = logits.argmax(dim=1)
                    next_w = torch.where(
                        use_pred, pred_w, cap_in[:, t + 1])
                    embed_t = self.embed_dropout(
                        self.embedding(next_w))
                else:
                    embed_t = self.embed_dropout(
                        self.embedding(cap_in[:, t + 1]))
        return outputs, alphas

    def generate(self, features, vocab, max_length=50, beam_size=1):
        if beam_size > 1:
            return self._beam(features, vocab, max_length, beam_size)
        return self._greedy(features, vocab, max_length)

    def _greedy(self, features, vocab, max_length):
        dev = features.device
        words, att_list = [], []
        h, c = self._init_hidden(features)
        wi = torch.tensor([vocab.word2idx['<start>']], device=dev)
        for _ in range(max_length):
            e = self.dropout(self.embedding(wi))
            ctx, a = self.attention(features, h)
            h, c = self.lstm_cell(
                torch.cat([e, ctx], 1), (h, c))
            wi = self.fc(h).argmax(1)
            att_list.append(a.detach().cpu())
            w = vocab.idx2word.get(wi.item(), '<unk>')
            if w == '<end>':
                break
            words.append(w)
        return ' '.join(words), att_list

    def _beam(self, features, vocab, max_length, beam_size):
        dev = features.device
        si = vocab.word2idx['<start>']
        ei = vocab.word2idx['<end>']
        h, c = self._init_hidden(features)
        beams = [(0.0, [si], (h, c), [])]
        done = []
        for _ in range(max_length):
            new_beams = []
            for sc, seq, (hb, cb), al in beams:
                wi = torch.tensor([seq[-1]], device=dev)
                e = self.embedding(wi)
                ctx, a = self.attention(features, hb)
                hn, cn = self.lstm_cell(
                    torch.cat([e, ctx], 1), (hb, cb))
                lp = torch.log_softmax(self.fc(hn), 1)
                tp, ti = lp.topk(beam_size)
                for i in range(beam_size):
                    ns = sc + tp[0, i].item()
                    nseq = seq + [ti[0, i].item()]
                    nal = al + [a.detach().cpu()]
                    if ti[0, i].item() == ei:
                        done.append(
                            (ns / len(nseq), nseq, nal))
                    else:
                        new_beams.append((
                            ns, nseq,
                            (hn.clone(), cn.clone()), nal))
            beams = sorted(new_beams, key=lambda x: x[0],
                           reverse=True)[:beam_size]
            if len(done) >= beam_size:
                break
        if done:
            best = max(done, key=lambda x: x[0])
        else:
            best = (beams[0][0], beams[0][1], beams[0][3])
        words = [vocab.idx2word.get(i, '<unk>')
                 for i in best[1] if i not in (si, ei, 0)]
        return ' '.join(words), best[2]


# ======================================================================
#  WRAPPER
# ======================================================================
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=5000,
                 dropout=0.5, attention_dim=256, **kwargs):
        super().__init__()
        self.encoder = CustomCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size,
                                  dropout, attention_dim)

    def forward(self, images, captions, ss_prob=0.0):
        return self.decoder(self.encoder(images), captions, ss_prob)

    def generate_caption(self, image, vocab, max_length=50, beam_size=3):
        self.eval()
        with torch.no_grad():
            feat = self.encoder(image)
            return self.decoder.generate(feat, vocab, max_length, beam_size)