"""
dataset.py — Dataset Flickr30k/8k, transforms, vocabulaire et DataLoaders.

Compatible avec :
  - Flickr30k (results.csv : image_name | comment_number | comment)
  - Flickr8k  (captions.txt : image | caption)
  - Tout CSV avec colonnes image + caption
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T

from vocabulary import Vocabulary


# ======================================================================
#  TRANSFORMS
# ======================================================================
def get_transforms(split='train'):
    """
    Retourne les transformations d'image selon le split.

    Args:
        split: 'train', 'val', ou 'test'

    Returns:
        torchvision.transforms.Compose
    """
    if split == 'train':
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
            ),
            T.RandomRotation(degrees=10),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def get_transform(split='train'):
    """Alias pour get_transforms."""
    return get_transforms(split)


# ======================================================================
#  CHARGEMENT FLEXIBLE DES CAPTIONS
# ======================================================================
def load_captions(captions_file: str) -> pd.DataFrame:
    """
    Charge les captions depuis un fichier CSV, en détectant
    automatiquement le format (Flickr30k ou Flickr8k).

    Returns:
        DataFrame avec colonnes ['image', 'caption']
    """
    # Essayer de lire avec séparateur | (Flickr30k results.csv)
    try:
        df = pd.read_csv(captions_file, sep="|")
        df.columns = [c.strip().lower() for c in df.columns]
        # Flickr30k : image_name | comment_number | comment
        if "image_name" in df.columns and "comment" in df.columns:
            df = df.rename(columns={"image_name": "image", "comment": "caption"})
            df = df[["image", "caption"]].dropna()
            df["image"] = df["image"].str.strip()
            df["caption"] = df["caption"].str.strip()
            print(f"[Dataset] Format détecté : Flickr30k (séparateur |)")
            print(f"[Dataset] {len(df)} paires image-caption chargées")
            return df
    except Exception:
        pass

    # Flickr8k ou CSV classique (séparateur ,)
    df = pd.read_csv(captions_file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Renommer si nécessaire
    if "image" not in df.columns:
        # Chercher la colonne image
        for col in df.columns:
            if "image" in col or "filename" in col or "file" in col:
                df = df.rename(columns={col: "image"})
                break
        else:
            # Fallback : première colonne = image
            df.columns = ["image"] + list(df.columns[1:])

    if "caption" not in df.columns:
        for col in df.columns:
            if "caption" in col or "comment" in col or "text" in col:
                df = df.rename(columns={col: "caption"})
                break
        else:
            # Fallback : deuxième colonne = caption
            cols = list(df.columns)
            cols[1] = "caption"
            df.columns = cols

    df = df[["image", "caption"]].dropna()
    df["image"] = df["image"].str.strip()
    df["caption"] = df["caption"].str.strip()
    print(f"[Dataset] Format détecté : CSV classique")
    print(f"[Dataset] {len(df)} paires image-caption chargées")
    return df


# ======================================================================
#  DATASET
# ======================================================================
class FlickrDataset(Dataset):
    """
    Dataset Flickr30k / Flickr8k.
    Chaque élément : (image_tensor, caption_indices).
    """

    def __init__(
        self,
        root_dir: str,
        captions_file: str,
        vocab: Vocabulary,
        transform=None,
        split: str = "train",
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        df: pd.DataFrame = None,
    ):
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform if transform is not None else get_transform(split)

        # --- Charger les captions ---
        if df is not None:
            self.df = df.copy()
        else:
            self.df = load_captions(captions_file)

        # --- Filtrer les images qui existent réellement sur disque ---
        existing = set()
        for img in self.df["image"].unique():
            if os.path.exists(os.path.join(root_dir, img)):
                existing.add(img)
        before = len(self.df["image"].unique())
        self.df = self.df[self.df["image"].isin(existing)].reset_index(drop=True)
        after = len(self.df["image"].unique())
        if before != after:
            print(f"[Dataset] {before - after} images introuvables sur disque, "
                  f"ignorées ({after} restantes)")

        # --- Split train / val / test ---
        all_images = self.df["image"].unique()
        n = len(all_images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        rng = np.random.RandomState(42)
        idx = np.arange(n)
        rng.shuffle(idx)
        all_images_shuffled = all_images[idx]

        if split == "train":
            keep = set(all_images_shuffled[:n_train])
        elif split == "val":
            keep = set(all_images_shuffled[n_train:n_train + n_val])
        else:  # test
            keep = set(all_images_shuffled[n_train + n_val:])

        self.df = self.df[self.df["image"].isin(keep)].reset_index(drop=True)
        print(f"[Dataset] split={split} — {len(self.df)} paires image-caption "
              f"({len(keep)} images)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        caption = str(row["caption"])
        img_name = row["image"]

        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        caption_indices = self.vocab.numericalize(caption)
        return image, torch.tensor(caption_indices, dtype=torch.long)


# Alias pour compatibilité
Flickr8kDataset = FlickrDataset


# ======================================================================
#  COLLATE (padding dynamique)
# ======================================================================
class CaptionCollate:
    """Pad les captions à la même longueur dans un batch."""

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        captions = pad_sequence(captions, batch_first=True,
                                padding_value=self.pad_idx)
        return images, captions


# ======================================================================
#  UTILITAIRE : nettoyage de caption générée
# ======================================================================
def clean_generated_caption(caption_words):
    """Supprime les tokens spéciaux et les mots dupliqués consécutifs."""
    cleaned = []
    prev_word = None

    for word in caption_words:
        if word in ['<start>', '<end>', '<pad>', '<unk>']:
            continue
        if word != prev_word:
            cleaned.append(word)
            prev_word = word

    return cleaned


# ======================================================================
#  HELPER : créer vocab + dataloaders
# ======================================================================
def build_vocab_and_loaders(
    images_dir: str,
    captions_file: str,
    batch_size: int = 32,
    freq_threshold: int = 5,
    num_workers: int = 2,
):
    """Construit le vocabulaire et renvoie les DataLoaders train/val/test."""
    # Charger toutes les captions avec détection automatique du format
    df = load_captions(captions_file)

    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_vocabulary(df["caption"].tolist())

    pad_idx = vocab.word2idx[Vocabulary.PAD_TOKEN]

    datasets = {}
    loaders = {}
    for split in ["train", "val", "test"]:
        transform = get_transforms("train" if split == "train" else "val")
        ds = FlickrDataset(
            root_dir=images_dir,
            captions_file=captions_file,
            vocab=vocab,
            transform=transform,
            split=split,
            df=df,  # Passer le df déjà chargé pour éviter re-lecture
        )
        datasets[split] = ds
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=CaptionCollate(pad_idx),
        )

    return vocab, datasets, loaders
