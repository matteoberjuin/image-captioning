"""
train.py — Entraînement avec améliorations :
  • Label smoothing (régularisation du décodeur)
  • Scheduled sampling (réduit le gap train/inférence)
  • LR warmup + ReduceLROnPlateau (stabilise le début d'entraînement)
  • Doubly stochastic attention regularization

Usage :
    python train.py --data_dir ./flickr30k_images/flickr30k_images \
                    --captions_file ./flickr30k_images/results.csv \
                    --encoder_type custom --epochs 30 --batch_size 32
"""

import os
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import ImageCaptioningModel
from dataset import build_vocab_and_loaders
from vocabulary import Vocabulary
from evaluate import compute_bleu_scores, print_bleu_results


# ======================================================================
#  CONFIGURATION
# ======================================================================
def get_args():
    p = argparse.ArgumentParser(description="Image Captioning Training")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--captions_file", type=str, required=True)
    # Modèle
    p.add_argument("--encoder_type", type=str, default="custom",
                   choices=["custom", "resnet"])
    p.add_argument("--embed_size", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--attention_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--use_attention", action="store_true", default=True)
    p.add_argument("--no_attention", dest="use_attention",
                   action="store_false")
    p.add_argument("--fine_tune_encoder", action="store_true", default=False)
    # Entraînement
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--freq_threshold", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2)
    # Améliorations
    p.add_argument("--alpha_c", type=float, default=1.0,
                   help="Poids régularisation doubly stochastic")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing (0 = désactivé)")
    p.add_argument("--ss_start_epoch", type=int, default=5,
                   help="Epoch où le scheduled sampling commence")
    p.add_argument("--ss_max_prob", type=float, default=0.25,
                   help="Probabilité max de scheduled sampling")
    p.add_argument("--ss_ramp_epochs", type=int, default=10,
                   help="Nombre d'epochs pour atteindre ss_max_prob")
    p.add_argument("--warmup_epochs", type=int, default=3,
                   help="Nombre d'epochs de warmup du LR")
    # Resume
    p.add_argument("--resume", type=str, default=None)
    # Sauvegarde
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args()


# ======================================================================
#  SCHEDULED SAMPLING
# ======================================================================
def get_ss_prob(epoch, start_epoch, max_prob, ramp_epochs):
    """
    Calcule la probabilité de scheduled sampling pour l'epoch courante.

    Rampe linéaire de 0 à max_prob entre start_epoch et
    start_epoch + ramp_epochs.
    """
    if epoch < start_epoch:
        return 0.0
    progress = min((epoch - start_epoch) / max(ramp_epochs, 1), 1.0)
    return max_prob * progress


# ======================================================================
#  LR WARMUP SCHEDULER
# ======================================================================
def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Warmup linéaire pendant warmup_epochs, puis passage au LR normal.
    Retourne un LambdaLR scheduler.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Rampe linéaire de 0.1 à 1.0
            return 0.1 + 0.9 * (epoch / max(warmup_epochs, 1))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


# ======================================================================
#  RÉGULARISATION DOUBLY STOCHASTIC
# ======================================================================
def attention_regularization(alphas, alpha_c=1.0):
    return alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()


# ======================================================================
#  BOUCLE D'ENTRAÎNEMENT
# ======================================================================
def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_clip, alpha_c, use_attention, ss_prob=0.0):
    model.train()
    total_loss, n = 0, 0
    for images, captions in tqdm(loader, desc="  Train", leave=False):
        images, captions = images.to(device), captions.to(device)

        # Scheduled sampling : passer ss_prob au modèle
        outputs, alphas = model(images, captions, ss_prob=ss_prob)

        targets = captions[:, 1:]
        B, T, V = outputs.shape
        loss = criterion(outputs.reshape(B * T, V), targets.reshape(B * T))

        if use_attention and alphas is not None:
            loss += attention_regularization(alphas, alpha_c)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, alpha_c, use_attention):
    model.eval()
    total_loss, n = 0, 0
    for images, captions in tqdm(loader, desc="  Val  ", leave=False):
        images, captions = images.to(device), captions.to(device)
        outputs, alphas = model(images, captions, ss_prob=0.0)
        targets = captions[:, 1:]
        B, T, V = outputs.shape
        loss = criterion(outputs.reshape(B * T, V), targets.reshape(B * T))
        if use_attention and alphas is not None:
            loss += attention_regularization(alphas, alpha_c)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ======================================================================
#  PLOT
# ======================================================================
def plot_training_curves(train_losses, val_losses, bleu_scores, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label="Train Loss", lw=2)
    axes[0].plot(val_losses, label="Val Loss", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Courbes de Loss"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if bleu_scores:
        ep = [s["epoch"] for s in bleu_scores]
        for k in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]:
            axes[1].plot(ep, [s[k] for s in bleu_scores],
                         label=k, lw=2, marker="o")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("BLEU")
        axes[1].set_title("Scores BLEU"); axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================================
#  MAIN
# ======================================================================
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Device] {device}")
    print(f"[Encoder] {args.encoder_type.upper()}"
          + (" (CNN from scratch + skip connections)"
             if args.encoder_type == "custom"
             else " (ResNet-50 pré-entraîné)"))
    print(f"[Améliorations] Label smoothing={args.label_smoothing} | "
          f"Scheduled sampling (start={args.ss_start_epoch}, "
          f"max={args.ss_max_prob}) | Warmup={args.warmup_epochs} epochs")

    # --- Données ---
    print("\n[1/4] Chargement des données...")
    vocab, datasets, loaders = build_vocab_and_loaders(
        images_dir=args.data_dir,
        captions_file=args.captions_file,
        batch_size=args.batch_size,
        freq_threshold=args.freq_threshold,
        num_workers=args.num_workers,
    )
    vocab.save(os.path.join(args.save_dir, "vocab.json"))

    # --- Modèle ---
    print("\n[2/4] Construction du modèle...")
    if args.resume:
        print(f"[Resume] {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        ca = ckpt.get("args", {})
        model = ImageCaptioningModel(
            embed_size=ca.get("embed_size", args.embed_size),
            hidden_size=ca.get("hidden_size", args.hidden_size),
            vocab_size=len(vocab),
            num_layers=ca.get("num_layers", args.num_layers),
            dropout=args.dropout,
            fine_tune_encoder=args.fine_tune_encoder,
            use_attention=ca.get("use_attention", args.use_attention),
            attention_dim=ca.get("attention_dim", args.attention_dim),
            encoder_type=ca.get("encoder_type", args.encoder_type),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        prev_epoch = ckpt.get("epoch", 0)
        prev_val_loss = ckpt.get("val_loss", float("inf"))
    else:
        model = ImageCaptioningModel(
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            vocab_size=len(vocab),
            num_layers=args.num_layers,
            dropout=args.dropout,
            fine_tune_encoder=args.fine_tune_encoder,
            use_attention=args.use_attention,
            attention_dim=args.attention_dim,
            encoder_type=args.encoder_type,
        ).to(device)
        prev_epoch = 0
        prev_val_loss = float("inf")

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres totaux    : {total_p:,}")
    print(f"  Paramètres entraînés : {train_p:,}")

    # --- Optimiseur + Schedulers ---
    if args.fine_tune_encoder and args.encoder_type == "resnet":
        optimizer = torch.optim.Adam([
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.decoder.parameters(), "lr": args.lr},
        ])
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        )

    # Warmup scheduler (premiers epochs)
    warmup_scheduler = get_warmup_scheduler(
        optimizer, args.warmup_epochs, args.epochs)
    # Plateau scheduler (après warmup)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # Loss avec label smoothing
    pad_idx = vocab.word2idx[Vocabulary.PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=args.label_smoothing,
    )
    print(f"[Loss] CrossEntropy (label_smoothing={args.label_smoothing})")

    # --- Entraînement ---
    print(f"\n[3/4] Entraînement — {args.epochs} epochs")
    best_val_loss = prev_val_loss
    train_losses, val_losses, bleu_history = [], [], []

    for epoch in range(1, args.epochs + 1):
        ge = prev_epoch + epoch
        t0 = time.time()

        # Scheduled sampling : probabilité croissante
        ss_prob = get_ss_prob(epoch, args.ss_start_epoch,
                              args.ss_max_prob, args.ss_ramp_epochs)

        tl = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
            args.grad_clip, args.alpha_c, args.use_attention,
            ss_prob=ss_prob,
        )
        vl = validate(model, loaders["val"], criterion, device,
                      args.alpha_c, args.use_attention)

        # Scheduler : warmup puis plateau
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(vl)

        train_losses.append(tl)
        val_losses.append(vl)
        lr = optimizer.param_groups[-1]['lr']
        elapsed = time.time() - t0

        print(f"  Epoch {ge:>3} ({epoch}/{args.epochs}) | "
              f"Train: {tl:.4f} | Val: {vl:.4f} | "
              f"LR: {lr:.2e} | SS: {ss_prob:.2f} | {elapsed:.1f}s",
              end="")

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save({
                "epoch": ge,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl,
                "args": vars(args),
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f" ★ best", end="")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            bleu = compute_bleu_scores(
                model, datasets["val"], vocab, device,
                num_samples=100, beam_size=3)
            bleu["epoch"] = ge
            bleu_history.append(bleu)
            print_bleu_results(bleu)

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": ge,
                "model_state_dict": model.state_dict(),
                "val_loss": vl,
                "args": vars(args),
            }, os.path.join(args.save_dir, f"checkpoint_epoch{ge}.pth"))

        print()

    # --- Évaluation finale ---
    print("\n[4/4] Évaluation finale sur le test set...")
    best = torch.load(os.path.join(args.save_dir, "best_model.pth"),
                      map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_results = compute_bleu_scores(
        model, datasets["test"], vocab, device, beam_size=3)
    print_bleu_results(test_results)
    plot_training_curves(train_losses, val_losses, bleu_history,
                         args.save_dir)

    print(f"\n[Terminé] Meilleur modèle : "
          f"{os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
