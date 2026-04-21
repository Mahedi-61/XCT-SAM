"""
plot_iou_bars.py
────────────────────────────────────────────────────────────────────────────────
Grouped bar chart of mean IoU ± std across all test sets and defect classes.

IoU is recomputed per-image from saved prediction mask PNGs vs ground-truth
labels, so no experiments need to be re-run.

Layout  : two panels — Pores (class_2) and Inclusions (class_3)
X-axis  : class_2: Te-1,Te-2,Te-4,Te-5,Te-6,ITe-3…ITe-6 | class_3: Te-1,Te-2,Te-6,ITe-6
Bars    : one group per test set, one bar per method
Error   : ± 1 std over per-image IoU values

────────────────────────────────────────────────────────────────────────────────
HOW TO ADD A NEW METHOD
────────────────────────────────────────────────────────────────────────────────
Add an entry to METHODS (below). Each entry maps:
  method_label → {
      "class_2": { test_key: metrics_txt_path, ... },   # pores
      "class_3": { test_key: metrics_txt_path, ... },   # inclusions
  }
where metrics_txt_path is the metrics_*.txt file written by the run script.
The file must contain a "Per-image IoU: ..." line (written after re-running).

Set metrics_txt_path to None if a method has no result for that test set.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── Base paths ───────────────────────────────────────────────────────────────
BASE      = "/media/mahedi/NVMe_8TB/CGFR/Paper_SAM_AM/prac/new_work/autogluon/examples/automm/Conv-LoRA"
GAN_DIR   = os.path.join(BASE, "datasets/gan-generated")
ISRAT_DIR = os.path.join(BASE, "datasets/israt")
OUT_DIR   = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Test-set display order (per class) ──────────────────────────────────────
TEST_SETS = {
    "class_2": ["Te-1", "Te-2", "Te-4", "Te-5", "Te-6", "ITe-3", "ITe-4", "ITe-5", "ITe-6"],
    "class_3": ["Te-1", "Te-2", "Te-6", "ITe-6"],
}
# Separator x-position between last GAN set and first Israt set (per panel)
SEPARATOR_X = {
    "class_2": 4.5,   # between Te-6 (idx 4) and ITe-3 (idx 5)
    "class_3": 2.5,   # between Te-6 (idx 2) and ITe-6 (idx 3)
}

# ─── Methods ──────────────────────────────────────────────────────────────────
# Each entry maps:  method_label → { "class_2": { test_key: metrics_txt }, ... }
# Set metrics_txt to None where a method has no result for that test set.
_GAN   = os.path.join(BASE, "output/gan_st")
_ISRAT = os.path.join(BASE, "output/israt_st")

_GAN_MED = "/media/mahedi/NVMe_8TB/CGFR/Paper_SAM_AM/prac/new_work/MedSAM/output/gan"
_ISRAT_MED = "/media/mahedi/NVMe_8TB/CGFR/Paper_SAM_AM/prac/new_work/MedSAM/output/israt"

METHODS = {
    # ── Add baseline methods below ─────────────────────────────────────────────
    "Med_SAM": {
        "class_2": {
            "Te-1":  os.path.join(_GAN_MED,   "test1_c2/metrics_test1_class2.txt"),
            "Te-2":  os.path.join(_GAN_MED,   "test2_c2/metrics_test2_class2.txt"),
            "Te-4":  os.path.join(_GAN_MED,   "test4_c2/metrics_test4_class2.txt"),
            "Te-5":  os.path.join(_GAN_MED,   "test5_c2/metrics_test5_class2.txt"),
            "Te-6":  os.path.join(_GAN_MED,   "test6_c2/metrics_test6_class2.txt"),
            "ITe-3": os.path.join(_ISRAT_MED, "test3_c2/metrics_test3_class1.txt"),
            "ITe-4": os.path.join(_ISRAT_MED, "test4_c2/metrics_test4_class1.txt"),
            "ITe-5": os.path.join(_ISRAT_MED, "test5_c2/metrics_test5_class1.txt"),
            "ITe-6": None,
        },
        "class_3": {
            "Te-1":  os.path.join(_GAN_MED,   "test1_c3/metrics_test1_class3.txt"),
            "Te-2":  os.path.join(_GAN_MED,   "test2_c3/metrics_test2_class3.txt"),
            #"Te-4":  os.path.join(_GAN,   "test4_c3/metrics_test4_class3.txt"),
            #"Te-5":  os.path.join(_GAN,   "test5_c3/metrics_test5_class3.txt"),
            "Te-6":  os.path.join(_GAN_MED,   "test6_c3/metrics_test6_class3.txt"),
            "ITe-3": None,
            "ITe-4": None,
            "ITe-5": None,
            "ITe-6": os.path.join(_ISRAT_MED, "test6/metrics_test6_class1.txt"),
        },
    },

    "XCT-SAM (Ours)": {
        "class_2": {
            "Te-1":  os.path.join(_GAN,   "test1_c2/metrics_test1_class2.txt"),
            "Te-2":  os.path.join(_GAN,   "test2_c2/metrics_test2_class2.txt"),
            "Te-4":  os.path.join(_GAN,   "test4_c2/metrics_test4_class2.txt"),
            "Te-5":  os.path.join(_GAN,   "test5_c2/metrics_test5_class2.txt"),
            "Te-6":  os.path.join(_GAN,   "test6_c2/metrics_test6_class2.txt"),
            "ITe-3": os.path.join(_ISRAT, "test3/metrics_test3_class1.txt"),
            "ITe-4": os.path.join(_ISRAT, "test4/metrics_test4_class1.txt"),
            "ITe-5": os.path.join(_ISRAT, "test5/metrics_test5_class1.txt"),
            "ITe-6": None,
        },
        "class_3": {
            "Te-1":  os.path.join(_GAN,   "test1_c3/metrics_test1_class3.txt"),
            "Te-2":  os.path.join(_GAN,   "test2_c3/metrics_test2_class3.txt"),
            #"Te-4":  os.path.join(_GAN,   "test4_c3/metrics_test4_class3.txt"),
            #"Te-5":  os.path.join(_GAN,   "test5_c3/metrics_test5_class3.txt"),
            "Te-6":  os.path.join(_GAN,   "test6_c3/metrics_test6_class3.txt"),
            "ITe-3": None,
            "ITe-4": None,
            "ITe-5": None,
            "ITe-6": os.path.join(_ISRAT, "test6/metrics_test6_class1.txt"),
        },
    },
}

# ─── Colours and style ────────────────────────────────────────────────────────
METHOD_COLOURS = {
    "XCT-SAM (Ours)": "#2166ac",
    "Conv-LoRA SAM":  "#d62728",
    "2.5D_UNet":     "#2ca02c",
    "MedSAM":        "#8c564b",
}
FONT_SIZE = 13

# ─── Parse metrics file ───────────────────────────────────────────────────────

def parse_metrics(txt_path):
    """Read Mean IoU and Std IoU from a metrics_*.txt file.

    Returns (mean, std), or (mean, 0.0) if Std IoU line is absent
    (old runs before std was added). Returns None if file does not exist.
    """
    if txt_path is None or not os.path.isfile(txt_path):
        return None
    with open(txt_path) as fh:
        text = fh.read()

    m_mean = re.search(r"Mean IoU\s*:\s*([0-9.]+)", text)
    if not m_mean:
        return None
    mean = float(m_mean.group(1))

    m_std = re.search(r"Std IoU\s*:\s*([0-9.]+)", text)
    std = float(m_std.group(1)) if m_std else 0.0
    if not m_std:
        print(f"  [warn] no Std IoU in {os.path.basename(txt_path)} — re-run eval to get std")

    return mean, std


# ─── Collect results ──────────────────────────────────────────────────────────
print("Reading metrics files ...")

# results[method][class_key][test_set] = (mean, std) or None
results = {}
for method, class_files in METHODS.items():
    results[method] = {"class_2": {}, "class_3": {}}
    for cls in ("class_2", "class_3"):
        for ts in TEST_SETS[cls]:
            txt_path = class_files[cls].get(ts)
            val = parse_metrics(txt_path)
            results[method][cls][ts] = val
            if val:
                print(f"  {method:20s} {cls} {ts:6s}  "
                      f"IoU={val[0]:.4f} ± {val[1]:.4f}")

# ─── Plot ─────────────────────────────────────────────────────────────────────
method_names = list(METHODS.keys())
n_methods    = len(method_names)
bar_width    = 0.8 / max(n_methods, 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=False)

panel_configs = [
    ("class_2", "Pores (class 2)"),
    ("class_3", "Inclusions (class 3)"),
]

for ax, (cls_key, panel_title) in zip(axes, panel_configs):
    test_sets = TEST_SETS[cls_key]
    sep_x     = SEPARATOR_X[cls_key]
    x         = np.arange(len(test_sets))
    has_data = False
    for i, method in enumerate(method_names):
        offset = (i - (n_methods - 1) / 2) * bar_width
        means, stds, xs = [], [], []
        for j, ts in enumerate(test_sets):
            val = results[method][cls_key].get(ts)
            if val is not None:
                means.append(val[0])
                stds.append(val[1])
                xs.append(x[j] + offset)

        if means:
            has_data = True
            colour = METHOD_COLOURS.get(method, "#333333")
            ax.bar(xs, means, width=bar_width * 0.9,
                   color=colour, alpha=0.85,
                   label=method, zorder=3)
            ax.errorbar(xs, means, yerr=stds,
                        fmt="none", ecolor="black",
                        elinewidth=1.0, capsize=3, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(test_sets, fontsize=FONT_SIZE, rotation=15, ha="right")
    ax.set_xlabel("Test Set", fontsize=FONT_SIZE)
    ax.set_ylabel("Mean IoU", fontsize=FONT_SIZE)
    ax.set_title(panel_title, fontsize=FONT_SIZE + 1, fontweight="bold", pad=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.axvspan(sep_x, len(test_sets) - 0.5,
               color="#f5f5f5", alpha=0.5, zorder=0)

# Shared legend below both panels
handles, labels = axes[0].get_legend_handles_labels()
if not handles:
    handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center",
           ncol=min(n_methods, 4),
           fontsize=FONT_SIZE,
           frameon=False,
           bbox_to_anchor=(0.5, -0.08),
           handlelength=1.4,
           columnspacing=1.5)

fig.suptitle(
    "Cross-domain IoU across GAN and Israt test sets\n"
    "(mean ± std over per-image predictions)",
    fontsize=FONT_SIZE + 1, y=1.01,
)

plt.tight_layout()
pdf_path = os.path.join(OUT_DIR, "iou_bars.pdf")
png_path = os.path.join(OUT_DIR, "iou_bars.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"\nSaved:\n  {pdf_path}\n  {png_path}")
plt.show()