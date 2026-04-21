"""
plot_density_iou.py
────────────────────────────────────────────────────────────────────────────────
Generates a publication-quality figure showing:
  • Volumetric density (%) per dataset for Pores (class_2) and Inclusions (class_3)
  • Inset zoom panel on the low-density test region
  • Vertical separator between GAN and Israt test sets

Usage:
    python3 plot_density_iou.py
────────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ─── Base paths ───────────────────────────────────────────────────────────────
BASE      = "/media/mahedi/NVMe_8TB/CGFR/Paper_SAM_AM/prac/new_work/autogluon/examples/automm/Conv-LoRA"
GAN_DIR   = os.path.join(BASE, "datasets/gan-generated")
ISRAT_DIR = os.path.join(BASE, "datasets/israt")

# ─── Density computation ──────────────────────────────────────────────────────

def density_from_csv(csv_path, dataset_dir):
    """Mean positive-pixel fraction (%) across all label images in a CSV."""
    df = pd.read_csv(csv_path)
    densities = []
    for rel_path in df["label"]:
        full_path = os.path.join(dataset_dir, rel_path)
        if not os.path.isfile(full_path):
            continue
        img = np.array(Image.open(full_path).convert("L"))
        densities.append((img > 0).sum() / img.size * 100.0)
    return float(np.mean(densities)) if densities else 0.0


def density_from_folder(label_dir):
    """Mean positive-pixel fraction (%) from all image files in a folder."""
    exts = (".png", ".tif", ".tiff", ".jpg")
    files = [f for f in sorted(os.listdir(label_dir))
             if f.lower().endswith(exts)]
    densities = []
    for f in files:
        img = np.array(Image.open(os.path.join(label_dir, f)).convert("L"))
        densities.append((img > 0).sum() / img.size * 100.0)
    return float(np.mean(densities)) if densities else 0.0


# ─── Compute all densities ────────────────────────────────────────────────────
print("Computing volumetric densities — this may take a minute ...")

gan_labels = {
    "Tr-1": {"pore": (f"{GAN_DIR}/train_class2.csv", GAN_DIR),
             "inclusion": (f"{GAN_DIR}/train_class3.csv", GAN_DIR)},
    "Te-1": {"pore": (f"{GAN_DIR}/test1_class2.csv", GAN_DIR),
             "inclusion": (f"{GAN_DIR}/test1_class3.csv", GAN_DIR)},
    "Te-2": {"pore": (f"{GAN_DIR}/test2_class2.csv", GAN_DIR),
             "inclusion": (f"{GAN_DIR}/test2_class3.csv", GAN_DIR)},
    "Te-3": {"pore": (f"{GAN_DIR}/test3_class2.csv", GAN_DIR),
             "inclusion": (f"{GAN_DIR}/test3_class3.csv", GAN_DIR)},
    "Te-4": {"pore": (f"{GAN_DIR}/test4_class2.csv", GAN_DIR),
             "inclusion": (f"{GAN_DIR}/test4_class3.csv", GAN_DIR)},
    "Te-5": {"pore": (f"{GAN_DIR}/test5_class2.csv", GAN_DIR),
             "inclusion": (f"{GAN_DIR}/test5_class3.csv", GAN_DIR)},
    "Te-6": {"pore": (f"{GAN_DIR}/test6_class2.csv", GAN_DIR),
             "inclusion": (f"{GAN_DIR}/test6_class3.csv", GAN_DIR)},
}

density_pore      = {}
density_inclusion = {}

for label, sources in gan_labels.items():
    density_pore[label]      = density_from_csv(*sources["pore"])
    density_inclusion[label] = density_from_csv(*sources["inclusion"])
    print(f"  {label:6s}  pore={density_pore[label]:.4f}%  "
          f"inclusion={density_inclusion[label]:.4f}%")

# Israt: folder-based; class_1 folder = ground truth labels
israt_sets = {
    "ITe-3": {"pore": os.path.join(ISRAT_DIR, "test3/class_1"), "inclusion": None},
    "ITe-4": {"pore": os.path.join(ISRAT_DIR, "test4/class_1"), "inclusion": None},
    "ITe-5": {"pore": os.path.join(ISRAT_DIR, "test5/class_1"), "inclusion": None},
    "ITe-6": {"pore": None, "inclusion": os.path.join(ISRAT_DIR, "test6/class_1")},
}

for label, sources in israt_sets.items():
    density_pore[label]      = density_from_folder(sources["pore"])      if sources["pore"]      else None
    density_inclusion[label] = density_from_folder(sources["inclusion"]) if sources["inclusion"] else None
    print(f"  {label:6s}  pore={density_pore[label]}  inclusion={density_inclusion[label]}")

# ─── X-axis layout ────────────────────────────────────────────────────────────
# Tr-1 | gap | Te-1..Te-6 | gap (separator) | ITe-3..ITe-6
LABELS    = ["Tr-1",
             "Te-1", "Te-2", "Te-3", "Te-4", "Te-5", "Te-6",
             "ITe-3", "ITe-4", "ITe-5", "ITe-6"]
X_POS     = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
SEPARATOR = 8.0

# ─── Colours ──────────────────────────────────────────────────────────────────
PORE_COLOR  = "#2166ac"   # blue
INCL_COLOR  = "#f4a261"   # orange
MARKER_SIZE  = 90
MARKER_ALPHA = 0.88
FONT_SIZE    = 15         # unified font size for main axes and inset

# ─── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))

# ── Scatter: volumetric density ───────────────────────────────────────────────
for label, xpos in zip(LABELS, X_POS):
    dp = density_pore.get(label)
    di = density_inclusion.get(label)
    if dp is not None:
        ax.scatter(xpos, dp, color=PORE_COLOR, s=MARKER_SIZE,
                   alpha=MARKER_ALPHA, zorder=3,
                   edgecolors="white", linewidths=0.5)
    if di is not None:
        ax.scatter(xpos, di, color=INCL_COLOR, s=MARKER_SIZE,
                   alpha=MARKER_ALPHA, zorder=3,
                   edgecolors="white", linewidths=0.5)

# ── Axes limits and ticks ─────────────────────────────────────────────────────
ax.set_xticks(X_POS)
ax.set_xticklabels(LABELS, fontsize=FONT_SIZE)
ax.set_xlim(-0.8, 12.8)
ax.set_ylabel("Volumetric Density (%)", fontsize=FONT_SIZE)
ax.set_xlabel("Dataset", fontsize=FONT_SIZE)
ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# ── Background shading for Israt region ──────────────────────────────────────
ax.axvspan(SEPARATOR, 12.8, color="#f5f5f5", alpha=0.5, zorder=0)

# ── Separator line ────────────────────────────────────────────────────────────
ax.axvline(x=SEPARATOR, color="gray", linestyle="--", linewidth=1.2,
           alpha=0.6, zorder=1)

# ─── Inset: zoomed view of low-density test region ────────────────────────────
# Collect all test-set densities to set zoom y-limit automatically
test_labels = ["Te-1", "Te-2", "Te-3", "Te-4", "Te-5", "Te-6",
               "ITe-3", "ITe-4", "ITe-5", "ITe-6"]
test_densities = [v for lbl in test_labels
                  for v in [density_pore.get(lbl), density_inclusion.get(lbl)]
                  if v is not None]

if test_densities:
    # ── Hard-coded axis limits ─────────────────────────────────────────────────
    Y_TOP    = 24.0   # main y-axis upper limit (%)
    zoom_max =  0.40  # inset y-axis upper limit (%) — also defines box height

    ax.set_ylim(0, Y_TOP)

    # Place inset on the LEFT so it does not overlap the ITe-3..ITe-6 dots.
    # ITe-3 sits at ~0.72 axes-fraction; inset ends at 0.07+0.44=0.51 — well clear.
    axins = ax.inset_axes([0.07, 0.33, 0.44, 0.56])   # [left, bottom, w, h]

    # Plot all dots inside the inset (including Tr-1)
    for label, xpos in zip(LABELS, X_POS):
        dp = density_pore.get(label)
        di = density_inclusion.get(label)
        if dp is not None:
            axins.scatter(xpos, dp, color=PORE_COLOR, s=MARKER_SIZE * 0.75,
                          alpha=MARKER_ALPHA, zorder=3,
                          edgecolors="white", linewidths=0.4)
        if di is not None:
            axins.scatter(xpos, di, color=INCL_COLOR, s=MARKER_SIZE * 0.75,
                          alpha=MARKER_ALPHA, zorder=3,
                          edgecolors="white", linewidths=0.4)

    # Inset axes limits — full x range including Tr-1
    axins.set_xlim(-0.5, 12.5)
    axins.set_ylim(0, zoom_max)

    # Y-axis label — same font size as main axes
    axins.set_ylabel("Vol. Density (%)", fontsize=FONT_SIZE, labelpad=3)
    axins.tick_params(labelsize=FONT_SIZE)
    axins.set_xticks([])                        # x-ticks omitted (labels on main axes)

    # Mirror the separator and shading inside the inset
    axins.axvline(x=SEPARATOR, color="gray", linestyle="--",
                  linewidth=0.9, alpha=0.5, zorder=1)
    axins.axvspan(SEPARATOR, 12.5, color="#f5f5f5", alpha=0.5, zorder=0)
    axins.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    axins.set_axisbelow(True)

    # Legend inside inset — same font size as main axes
    axins.scatter([], [], color=PORE_COLOR, s=40, label="Pores")
    axins.scatter([], [], color=INCL_COLOR, s=40, label="Inclusions")
    axins.legend(fontsize=FONT_SIZE, loc="upper left", framealpha=0.88,
                 handletextpad=0.3, borderpad=0.4)

    # ── Dashed box on main axes marking the zoomed region ──────────────────
    # zorder=1 keeps the box border BELOW the scatter dots (zorder=3)
    rect = mpatches.FancyBboxPatch(
        (-0.5, 0), 13.0, zoom_max,
        boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor="#cc44aa",
        linestyle="--", facecolor="none", zorder=1
    )
    ax.add_patch(rect)

# ── "Israt →" label on separator ─────────────────────────────────────────────
ax.text(0.683, 0.97, "Israt →", transform=ax.transAxes,
        fontsize=FONT_SIZE, color="gray", va="top", ha="left")

# ─── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout()
out_dir  = os.path.join(BASE, "output")
os.makedirs(out_dir, exist_ok=True)
pdf_path = os.path.join(out_dir, "density_plot.pdf")
png_path = os.path.join(out_dir, "density_plot.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"\nSaved:\n  {pdf_path}\n  {png_path}")
plt.show()
