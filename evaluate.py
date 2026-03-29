"""
evaluate.py — Évaluation du modèle avec les métriques BLEU-1 à BLEU-4.

Utilise nltk.translate.bleu_score pour un calcul rigoureux avec
smoothing (méthode de Chen & Cherry, 2014).
"""

import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from dataset import get_transforms
from vocabulary import Vocabulary
from PIL import Image


# Télécharger punkt si nécessaire
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ======================================================================
#  CALCUL DU SCORE BLEU
# ======================================================================
def compute_bleu_scores(
    model,
    dataset,
    vocab: Vocabulary,
    device: torch.device,
    max_length: int = 50,
    beam_size: int = 3,
    num_samples: int = None,
):
    """
    Calcule les scores BLEU-1 à BLEU-4 sur un dataset.

    Args:
        model       : ImageCaptioningModel
        dataset     : FlickrDataset (test ou val)
        vocab       : Vocabulary
        device      : torch device
        max_length  : longueur max pour la génération
        beam_size   : taille du beam search
        num_samples : nombre d'images à évaluer (None = toutes)

    Returns:
        dict avec BLEU-1, BLEU-2, BLEU-3, BLEU-4 et exemples
    """
    model.eval()
    transform = get_transforms("val")
    smoother = SmoothingFunction().method1

    # Regrouper les captions par image
    image_captions = defaultdict(list)
    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        img_name = row["image"]
        caption = str(row["caption"]).lower().strip()
        image_captions[img_name].append(caption.split())

    # Limiter le nombre d'images
    image_names = list(image_captions.keys())
    if num_samples is not None:
        image_names = image_names[:num_samples]

    references_corpus = []
    hypotheses_corpus = []
    examples = []

    print(f"\n[Évaluation] BLEU sur {len(image_names)} images "
          f"(beam_size={beam_size})...")

    for img_name in tqdm(image_names, desc="BLEU"):
        img_path = os.path.join(dataset.root_dir, img_name)
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Générer la caption
        caption_pred, _ = model.generate_caption(
            image_tensor, vocab, max_length, beam_size
        )
        hypothesis = caption_pred.split()

        # Références
        refs = image_captions[img_name]
        references_corpus.append(refs)
        hypotheses_corpus.append(hypothesis)

        # Garder quelques exemples
        if len(examples) < 10:
            examples.append({
                "image": img_name,
                "predicted": caption_pred,
                "references": [" ".join(r) for r in refs],
            })

    if not references_corpus:
        print("[Évaluation] Aucune image évaluée !")
        return {"BLEU-1": 0, "BLEU-2": 0, "BLEU-3": 0, "BLEU-4": 0,
                "num_images": 0, "examples": []}

    # Calcul des scores BLEU corpus-level
    bleu_1 = corpus_bleu(references_corpus, hypotheses_corpus,
                         weights=(1.0, 0, 0, 0),
                         smoothing_function=smoother)
    bleu_2 = corpus_bleu(references_corpus, hypotheses_corpus,
                         weights=(0.5, 0.5, 0, 0),
                         smoothing_function=smoother)
    bleu_3 = corpus_bleu(references_corpus, hypotheses_corpus,
                         weights=(0.33, 0.33, 0.33, 0),
                         smoothing_function=smoother)
    bleu_4 = corpus_bleu(references_corpus, hypotheses_corpus,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=smoother)

    results = {
        "BLEU-1": bleu_1,
        "BLEU-2": bleu_2,
        "BLEU-3": bleu_3,
        "BLEU-4": bleu_4,
        "num_images": len(references_corpus),
        "examples": examples,
    }

    return results


# ======================================================================
#  AFFICHAGE DES RÉSULTATS
# ======================================================================
def print_bleu_results(results: dict):
    """Affiche les scores BLEU de manière claire."""
    print("\n" + "=" * 60)
    print("  RÉSULTATS — SCORES BLEU")
    print("=" * 60)
    print(f"  Images évaluées : {results['num_images']}")
    print(f"  BLEU-1 : {results['BLEU-1']:.4f}")
    print(f"  BLEU-2 : {results['BLEU-2']:.4f}")
    print(f"  BLEU-3 : {results['BLEU-3']:.4f}")
    print(f"  BLEU-4 : {results['BLEU-4']:.4f}")
    print("=" * 60)

    if results.get("examples"):
        print("\n  EXEMPLES DE PRÉDICTIONS :")
        print("-" * 60)
        for ex in results["examples"][:5]:
            print(f"  Image     : {ex['image']}")
            print(f"  Prédiction: {ex['predicted']}")
            print(f"  Référence : {ex['references'][0]}")
            print("-" * 60)


# ======================================================================
#  BLEU SCORE POUR UNE SEULE IMAGE
# ======================================================================
def evaluate_single_image(
    model,
    image_path: str,
    vocab: Vocabulary,
    device: torch.device,
    references: list = None,
    beam_size: int = 3,
):
    """Évalue une seule image et retourne la caption + BLEU si références dispo."""
    model.eval()
    transform = get_transforms("val")
    smoother = SmoothingFunction().method1

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    caption, alphas = model.generate_caption(
        image_tensor, vocab, max_length=50, beam_size=beam_size
    )

    result = {"caption": caption, "alphas": alphas}

    if references:
        hypothesis = caption.split()
        refs = [r.lower().split() for r in references]
        result["BLEU-1"] = sentence_bleu(
            refs, hypothesis, weights=(1, 0, 0, 0),
            smoothing_function=smoother)
        result["BLEU-4"] = sentence_bleu(
            refs, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoother)

    return result
