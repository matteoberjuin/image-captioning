"""
demo.py — Script de demonstration pour l'examen oral.

Usage :
    python demo.py --image ./test.jpg \
                   --checkpoint ./checkpoints/best_model.pth \
                   --vocab ./checkpoints/vocab.json

    python demo.py --interactive \
                   --checkpoint ./checkpoints/best_model.pth \
                   --vocab ./checkpoints/vocab.json
"""

import os
import argparse
import warnings
import torch
import numpy as np

# Matplotlib backend : TkAgg si dispo, sinon Agg
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
warnings.filterwarnings('ignore', category=FutureWarning)

# Pillow compatibility (BILINEAR renamed in Pillow 10+)
BILINEAR = getattr(Image, 'Resampling', Image).BILINEAR

from model import ImageCaptioningModel
from vocabulary import Vocabulary

import torchvision.transforms as T


def get_transforms(split='val'):
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def clean_caption(text):
    """Supprime les mots dupliques consecutifs."""
    words = text.split()
    cleaned, prev = [], None
    for w in words:
        if w != prev:
            cleaned.append(w)
            prev = w
    return ' '.join(cleaned)


def visualize_attention(image_path, caption, alphas, save_path=None):
    """Affiche l'image avec les poids d'attention pour chaque mot."""
    if not alphas:
        print("  [!] Pas de poids d'attention disponibles.")
        return

    image = Image.open(image_path).convert("RGB").resize((224, 224))
    words = caption.split()
    n_words = min(len(words), len(alphas))

    n_cols = min(5, n_words + 1)
    n_rows = (n_words + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3, n_rows * 3))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for i in range(n_words):
        alpha = alphas[i].squeeze().numpy().reshape(7, 7)
        alpha_resized = np.array(
            Image.fromarray(alpha).resize((224, 224), BILINEAR))
        axes[i + 1].imshow(image)
        axes[i + 1].imshow(alpha_resized, alpha=0.6, cmap="jet")
        axes[i + 1].set_title(words[i], fontsize=11, fontweight="bold")
        axes[i + 1].axis("off")

    for j in range(n_words + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(caption, fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [Sauvegarde] {save_path}")

    plt.show()
    plt.close()


def generate_for_image(model, image_path, vocab, device,
                       beam_size=3, show_attention=True, save_dir=None):
    transform = get_transforms()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    caption, alphas = model.generate_caption(
        image_tensor, vocab, max_length=50, beam_size=beam_size)
    caption = clean_caption(caption)

    print(f"\n  Image   : {os.path.basename(image_path)}")
    print(f"  Caption : {caption}")

    if show_attention and alphas:
        save_path = None
        if save_dir:
            name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(save_dir, f"attention_{name}.png")
        visualize_attention(image_path, caption, alphas, save_path)

    return caption, alphas


def print_model_info(model, checkpoint):
    args = checkpoint.get("args", {})
    print("\n" + "=" * 60)
    print("  MODELE D'IMAGE CAPTIONING")
    print("=" * 60)
    print("  Encoder   : CNN from scratch (5 blocs residuels)")
    print("  Attention : Bahdanau (additive)")
    print("  Decoder   : LSTM")
    print(f"  Embed     : {args.get('embed_size', 256)}")
    print(f"  Hidden    : {args.get('hidden_size', 512)}")
    print(f"  Epoch     : {checkpoint.get('epoch', '?')}")
    vl = checkpoint.get('val_loss')
    if isinstance(vl, float):
        print(f"  Val loss  : {vl:.4f}")
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Params    : {total_p:,}")
    print("=" * 60)


def interactive_mode(model, vocab, device, beam_size=3, save_dir=None):
    print("\n[Mode interactif] Entrez le chemin d'une image "
          "(ou 'q' pour quitter)")
    while True:
        path = input("\n  Image > ").strip()
        if path.lower() in ('q', 'quit', 'exit'):
            print("  Au revoir !")
            break
        if not os.path.exists(path):
            print(f"  [!] Fichier introuvable : {path}")
            continue
        try:
            generate_for_image(model, path, vocab, device,
                               beam_size=beam_size,
                               show_attention=True,
                               save_dir=save_dir)
        except Exception as e:
            print(f"  [Erreur] {e}")


def main():
    parser = argparse.ArgumentParser(description="Image Captioning Demo")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="./demo_output")
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Charger le vocabulaire
    vocab = Vocabulary.load(args.vocab)

    # Charger le modele
    print("[Modele] Chargement du checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    model_args = checkpoint.get("args", {})

    model = ImageCaptioningModel(
        embed_size=model_args.get("embed_size", 256),
        hidden_size=model_args.get("hidden_size", 512),
        vocab_size=len(vocab),
        dropout=0.0,
        attention_dim=model_args.get("attention_dim", 256),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print_model_info(model, checkpoint)

    if args.interactive:
        interactive_mode(model, vocab, device, args.beam_size,
                         args.save_dir)
        return

    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for f in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                image_paths.append(os.path.join(args.image_dir, f))

    if not image_paths:
        print("[!] Aucune image. Utilisez --image, --image_dir "
              "ou --interactive")
        return

    print(f"\n  GENERATION — {len(image_paths)} image(s)\n")
    for img_path in image_paths:
        generate_for_image(
            model, img_path, vocab, device,
            beam_size=args.beam_size,
            show_attention=not args.no_attention,
            save_dir=args.save_dir)

    print(f"\n[Termine] Resultats dans {args.save_dir}")


if __name__ == "__main__":
    main()