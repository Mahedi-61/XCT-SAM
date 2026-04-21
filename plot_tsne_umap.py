"""
plot_tsne_umap.py
────────────────────────────────────────────────────────────────────────────────
Computes a t-SNE projection of deep features extracted from random foreground
150×150 crops of GAN-generated and Israt training/test images.

Feature extractor : torchvision ViT-B/16 pretrained on ImageNet-1K
                    (strong general-purpose transformer encoder)
Preprocessing     : CLAHE (clipLimit=2.0, tileGridSize=8×8) per image to
                    enhance local contrast before crop extraction
Datasets          : Tr-1/Tr-2 (train), Te-1…Te-6 (GAN test), ITe-3…ITe-6 (Israt test)
Crops per dataset : N_CROPS foreground-only crops from N_IMAGES random images

Usage:
    python3 plot_tsne_umap.py
────────────────────────────────────────────────────────────────────────────────
"""

import os
import random
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ─── Config ───────────────────────────────────────────────────────────────────
BASE       = "/media/mahedi/NVMe_8TB/CGFR/Paper_SAM_AM/prac/new_work/autogluon/examples/automm/Conv-LoRA"
GAN_DIR    = os.path.join(BASE, "datasets/gan-generated")
ISRAT_DIR  = os.path.join(BASE, "datasets/israt")
OUT_DIR    = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

CROP_SIZE       = 150    # pixels — same as FID setup in the reference paper
N_IMAGES        = 50     # images sampled per dataset
N_CROPS         = 7      # random crops per image  →  350 feature vectors per dataset
SEED            = 42
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
FONT_SIZE       = 11

# Foreground-only crop sampling
# A crop is accepted only if ≥ MIN_FG_FRACTION of its pixels are above
# FG_THRESHOLD (0–255).  XCT images have a dark circular background;
# pixels outside the specimen sit well below the metallic grey level.
# Empirically, specimen pixels are typically > 30; black border ≈ 0-15.
FG_THRESHOLD    = 30    # pixel intensity above which a pixel is "metal"
MIN_FG_FRACTION = 0.70  # at least 70 % of crop must be foreground
MAX_RETRIES     = 200   # max candidate crops per image before giving up

# CLAHE settings — applied per-image before cropping to sharpen local
# contrast so that ViT features respond to intensity texture (defect signal)
# rather than global brightness.
CLAHE_CLIP      = 2.0
CLAHE_TILE      = (8, 8)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Dataset paths and file-prefix filters ────────────────────────────────────
# Training is split by GAN pair setting: pair_1 (879 images) and pair_2 (121)
DATASETS = {
    "Tr-1":  (os.path.join(GAN_DIR,   "train/input"), "pair_1"),
    "Tr-2":  (os.path.join(GAN_DIR,   "train/input"), "pair_2"),
    "Te-1":  (os.path.join(GAN_DIR,   "test1/input"), None),
    "Te-2":  (os.path.join(GAN_DIR,   "test2/input"), None),
    "Te-3":  (os.path.join(GAN_DIR,   "test3/input"), None),
    "Te-4":  (os.path.join(GAN_DIR,   "test4/input"), None),
    "Te-5":  (os.path.join(GAN_DIR,   "test5/input"), None),
    "Te-6":  (os.path.join(GAN_DIR,   "test6/input"), None),
    "ITe-3": (os.path.join(ISRAT_DIR, "test3/input"), None),
    "ITe-4": (os.path.join(ISRAT_DIR, "test4/input"), None),
    "ITe-5": (os.path.join(ISRAT_DIR, "test5/input"), None),
    "ITe-6": (os.path.join(ISRAT_DIR, "test6/input"), None),
}

# Colour palette — visually distinct, print-friendly
COLOURS = {
    "Tr-1":  "#1f77b4",   # blue         (training pair_1)
    "Tr-2":  "#17becf",   # cyan         (training pair_2)
    "Te-1":  "#ff7f0e",   # orange
    "Te-2":  "#2ca02c",   # green
    "Te-3":  "#d62728",   # red          (InD)
    "Te-4":  "#9467bd",   # purple
    "Te-5":  "#8c564b",   # brown
    "Te-6":  "#e377c2",   # pink
    "ITe-3": "#bcbd22",   # yellow-green (Israt)
    "ITe-4": "#7f7f7f",   # grey
    "ITe-5": "#aec7e8",   # light blue
    "ITe-6": "#ffbb78",   # light orange
}

MARKERS = {
    "Tr-1":  "s",    # square   (training pair_1)
    "Tr-2":  "D",    # diamond  (training pair_2)
    "Te-1":  "o",
    "Te-2":  "o",
    "Te-3":  "*",    # star     (InD — stands out)
    "Te-4":  "o",
    "Te-5":  "o",
    "Te-6":  "o",
    "ITe-3": "^",    # triangle (Israt sets — distinct shape)
    "ITe-4": "^",
    "ITe-5": "^",
    "ITe-6": "^",
}

# ─── Feature extractor: ViT-B/16 pretrained ───────────────────────────────────
print("Loading ViT-B/16 pretrained weights ...")
weights   = ViT_B_16_Weights.IMAGENET1K_V1
model     = vit_b_16(weights=weights).to(DEVICE)
model.eval()

# Hook to capture the CLS token from the last transformer block
_features = {}
def _hook(module, inp, out):
    # out shape: (B, seq_len, hidden_dim) — take CLS token at position 0
    _features["cls"] = out[:, 0, :].detach().cpu()

model.encoder.layers[-1].register_forward_hook(_hook)

# ViT-B/16 expects 224×224 RGB; our images are 512×512 grayscale
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),  # replicate L channel to RGB
    T.ToTensor(),
    T.Normalize(mean=weights.transforms().mean,
                std=weights.transforms().std),
])

# ─── Crop extraction and feature computation ──────────────────────────────────

def apply_clahe(pil_img_L):
    """Apply CLAHE to a grayscale PIL image; return a grayscale PIL image."""
    arr = np.array(pil_img_L, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return Image.fromarray(clahe.apply(arr))


def random_crops(img_dir, n_images, n_crops, crop_size, prefix=None):
    """Return list of PIL crops sampled from foreground (metallic) regions only.

    Steps per image:
      1. Load as grayscale.
      2. Apply CLAHE to enhance local contrast → ViT sees intensity texture.
      3. Accept a crop only when ≥ MIN_FG_FRACTION of pixels in the
         *original* (pre-CLAHE) image exceed FG_THRESHOLD — this avoids
         crops dominated by the dark circular background border.
      4. Retry up to MAX_RETRIES times per image before moving on.

    If prefix is given, only files whose name contains that string are used.
    """
    files = [f for f in sorted(os.listdir(img_dir))
             if f.lower().endswith((".png", ".tif", ".tiff", ".jpg"))
             and (prefix is None or prefix in f)]
    sampled = random.sample(files, min(n_images, len(files)))
    crops = []
    for fname in sampled:
        raw   = Image.open(os.path.join(img_dir, fname)).convert("L")
        enh   = apply_clahe(raw)          # CLAHE-enhanced version for features
        raw_arr = np.array(raw)
        w, h  = raw.size
        collected = 0
        for _ in range(MAX_RETRIES):
            if collected >= n_crops:
                break
            x = random.randint(0, max(0, w - crop_size))
            y = random.randint(0, max(0, h - crop_size))
            patch_raw = raw_arr[y:y + crop_size, x:x + crop_size]
            fg_frac   = (patch_raw > FG_THRESHOLD).mean()
            if fg_frac < MIN_FG_FRACTION:
                continue                  # skip background-heavy crops
            crops.append(enh.crop((x, y, x + crop_size, y + crop_size)))
            collected += 1
    return crops


def extract_features(crops):
    """Run ViT-B/16 on a list of PIL crops; return (N, 768) numpy array."""
    feats = []
    batch_size = 32
    for i in range(0, len(crops), batch_size):
        batch = torch.stack([preprocess(c) for c in crops[i:i + batch_size]])
        batch = batch.to(DEVICE)
        with torch.no_grad():
            model(batch)               # triggers the forward hook
        feats.append(_features["cls"].numpy())
    return np.vstack(feats)


print(f"\nExtracting features ({N_IMAGES} images × {N_CROPS} crops = "
      f"{N_IMAGES * N_CROPS} vectors per dataset) ...")

all_features = []
all_labels   = []

for name, (img_dir, prefix) in DATASETS.items():
    crops = random_crops(img_dir, N_IMAGES, N_CROPS, CROP_SIZE, prefix=prefix)
    feats = extract_features(crops)
    all_features.append(feats)
    all_labels.extend([name] * len(feats))
    print(f"  {name:6s}  crops={len(crops)}  features={feats.shape}")

X      = np.vstack(all_features)
labels = np.array(all_labels)

# Standardise before dimensionality reduction
X_scaled = StandardScaler().fit_transform(X)

# ─── t-SNE ────────────────────────────────────────────────────────────────────
print("\nRunning t-SNE ...")
tsne    = TSNE(n_components=2, perplexity=40, n_iter_without_progress=1500,
               random_state=SEED, init="pca", learning_rate="auto")
X_tsne  = tsne.fit_transform(X_scaled)

# ─── Plotting ─────────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

fig, ax = plt.subplots(figsize=(9, 7))

for name in DATASETS:
    mask = labels == name
    size = 90 if name == "Te-3" else 55 if name in ("Tr-1", "Tr-2") else 45
    ax.scatter(
        X_tsne[mask, 0], X_tsne[mask, 1],
        c=COLOURS[name], marker=MARKERS[name],
        s=size, alpha=0.75, linewidths=0.3,
        edgecolors="white", label=name, zorder=3,
    )

ax.set_title("t-SNE of ViT-B/16 features",
             fontsize=FONT_SIZE + 1, fontweight="bold", pad=8)
ax.set_xlabel("t-SNE Dimension 1", fontsize=FONT_SIZE)
ax.set_ylabel("t-SNE Dimension 2", fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE - 1)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=False))
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=False))
ax.spines[["top", "right"]].set_visible(False)
ax.set_facecolor("#fafafa")
ax.grid(linestyle=":", alpha=0.35, zorder=0)

# Legend — 12 datasets: 3 rows × 4 columns
LABEL_SUFFIX = {
    "Tr-1":  "  (pair 1)",
    "Tr-2":  "  (pair 2)",
    "Te-3":  "  (InD)",
    "ITe-3": "  (Israt)",
    "ITe-4": "  (Israt)",
    "ITe-5": "  (Israt)",
    "ITe-6": "  (Israt)",
}

legend_handles = [
    Line2D([0], [0], marker=MARKERS[n], color="w",
           markerfacecolor=COLOURS[n], markersize=9,
           label=f"{n}{LABEL_SUFFIX.get(n, '')}")
    for n in DATASETS
]
ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=4,
    fontsize=FONT_SIZE - 1,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.2,
    handletextpad=0.4,
)

plt.tight_layout()
pdf_path = os.path.join(OUT_DIR, "tsne.pdf")
png_path = os.path.join(OUT_DIR, "tsne.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"\nSaved:\n  {pdf_path}\n  {png_path}")
plt.show()
