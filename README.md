# Image Captioning — Encoder CNN + Decoder LSTM avec Attention

## Architecture

```
Image (224×224×3) → ResNet-50 (Encoder) → Features spatiales (49×256)
                                            ↓
            Attention (Bahdanau) ← Hidden state LSTM
                    ↓
            Context vector + Word embedding → LSTM (Decoder) → Softmax → Mot suivant
```

### Composants principaux

| Module | Description |
|--------|-------------|
| **EncoderCNN** | ResNet-50 pré-entraîné (ImageNet), sans la couche de classification. Produit 49 features spatiales (grille 7×7) projetées en dimension `embed_size`. |
| **BahdanauAttention** | Attention additive — à chaque pas de temps, le décodeur "regarde" les zones pertinentes de l'image. |
| **DecoderRNN** | LSTMCell avec attention. Génère les mots un par un, conditionné par l'image et les mots précédents. |

### Métriques d'évaluation

Le projet utilise le **score BLEU** (Bilingual Evaluation Understudy) de BLEU-1 à BLEU-4, calculé au niveau du corpus avec smoothing (Chen & Cherry, 2014).

---

## Structure du projet

```
image_captioning_project/
├── model.py          # EncoderCNN + BahdanauAttention + DecoderRNN
├── dataset.py        # Dataset Flickr8k + DataLoader + transforms
├── vocabulary.py     # Vocabulaire (word ↔ index)
├── train.py          # Boucle d'entraînement complète
├── evaluate.py       # Calcul des scores BLEU-1 à BLEU-4
├── demo.py           # Inférence + visualisation d'attention
├── requirements.txt  # Dépendances Python
└── README.md         # Ce fichier
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset : Flickr8k

Télécharger depuis Kaggle : https://www.kaggle.com/datasets/adityajn105/flickr8k

Placer les fichiers comme suit :
```
flickr8k/
├── Images/
│   ├── 1000268201_693b08cb0e.jpg
│   ├── ...
└── captions.txt
```

---

## Entraînement

```bash
python train.py \
    --data_dir ./flickr8k/Images \
    --captions_file ./flickr8k/captions.txt \
    --epochs 20 \
    --batch_size 32 \
    --embed_size 256 \
    --hidden_size 512 \
    --lr 3e-4 \
    --use_attention \
    --save_dir ./checkpoints
```

### Options importantes

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--embed_size` | 256 | Dimension des embeddings |
| `--hidden_size` | 512 | Dimension du hidden state LSTM |
| `--use_attention` | True | Activer l'attention Bahdanau |
| `--fine_tune_encoder` | False | Fine-tuner le ResNet-50 |
| `--epochs` | 20 | Nombre d'epochs |
| `--batch_size` | 32 | Taille du batch |
| `--grad_clip` | 5.0 | Gradient clipping |
| `--alpha_c` | 1.0 | Poids de régularisation d'attention |

### Sortie de l'entraînement

- `checkpoints/best_model.pth` — meilleur modèle (val loss min)
- `checkpoints/vocab.json` — vocabulaire sauvegardé
- `checkpoints/training_curves.png` — courbes loss + BLEU

---

## Évaluation BLEU

Les scores BLEU sont calculés automatiquement pendant et après l'entraînement. Scores typiques sur Flickr8k :

| Métrique | Valeur attendue |
|----------|----------------|
| BLEU-1 | 0.55 – 0.65 |
| BLEU-2 | 0.35 – 0.45 |
| BLEU-3 | 0.24 – 0.32 |
| BLEU-4 | 0.16 – 0.24 |

---

## Démonstration (Inférence)

```bash
# Une seule image
python demo.py \
    --image ./test_image.jpg \
    --checkpoint ./checkpoints/best_model.pth \
    --vocab ./checkpoints/vocab.json \
    --beam_size 3

# Plusieurs images
python demo.py \
    --image_dir ./test_images/ \
    --checkpoint ./checkpoints/best_model.pth \
    --vocab ./checkpoints/vocab.json
```

Cela génère :
- La caption prédite pour chaque image
- La **visualisation des poids d'attention** (heatmap superposée à l'image)

---

## Concepts clés

### Teacher Forcing
Pendant l'entraînement, on fournit le **vrai mot précédent** au décodeur (pas le mot prédit). Cela stabilise l'apprentissage.

### Beam Search
À l'inférence, au lieu de choisir le mot le plus probable à chaque pas (greedy), on maintient les `k` meilleures séquences candidates. Cela améliore la qualité des captions.

### Doubly Stochastic Attention
Régularisation qui encourage chaque zone de l'image à recevoir une attention cumulée ≈ 1 sur l'ensemble de la phrase générée.

---

## Références

- Vinyals et al., "Show and Tell: A Neural Image Caption Generator", CVPR 2015
- Xu et al., "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", ICML 2015
- Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation", ACL 2002
