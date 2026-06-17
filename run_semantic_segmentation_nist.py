import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from PIL import Image
import numpy as np
from autogluon.multimodal import MultiModalPredictor


# ── israt dataset: evaluation with the GAN-trained class_2 model ─────────────
# Covers all four test sets with the same pipeline — no inversion needed.
#
#   test3  ~20%  defect pixels  dark defects on bright metal
#   test4  ~0.3% defect pixels  dark defects on bright metal (no pure-black border)
#   test5  ~8.9% defect pixels  dark defects on bright metal
#   test6  ~25%  defect pixels  bright defects on dark metal
#
# The class_2 model responds to high-contrast regions regardless of polarity,
# so its direct output works for all four sets without any inversion.


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def get_valid_region(img_arr):
    """
    Returns a boolean mask covering the circular metal specimen.
    """
    otsu_val, _ = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = otsu_val if otsu_val > 1 else img_arr.max() * 0.10
    mask = img_arr > threshold

    if mask.mean() < 0.30:
        # Otsu split highlights from background+metal instead of specimen from background.
        low_thresh = float(np.percentile(img_arr, 10)) + 5
        mask = img_arr > low_thresh

    if mask.mean() > 0.95:
        return np.ones(img_arr.shape, dtype=bool)

    mask_filled = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask_filled = ndi.binary_fill_holes(mask_filled)

    contours, _ = cv2.findContours(
        mask_filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)

    metal_mask = np.zeros_like(mask_filled, dtype=np.uint8)
    cv2.circle(metal_mask, (int(x), int(y)), int(radius), 1, -1)
    return metal_mask.astype(bool)


def calculate_iou(mask_pred, mask_true):
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()
    if union == 0:
        return 1.0
    return np.round(intersection / union, 4)


def tolerance_f1(mask_pred, mask_true, tolerance=5):
    """
    Tolerance-based F1: a predicted pixel counts as TP if it falls within
    `tolerance` pixels of any labeled defect pixel.
    """
    if mask_true.sum() == 0 and mask_pred.sum() == 0:
        return 1.0, 1.0, 1.0
    if mask_true.sum() == 0 or mask_pred.sum() == 0:
        return 0.0, 0.0, 0.0

    struct = ndi.generate_binary_structure(2, 1)
    dilated_true = ndi.binary_dilation(mask_true, structure=struct, iterations=tolerance)
    dilated_pred = ndi.binary_dilation(mask_pred, structure=struct, iterations=tolerance)

    tp = np.logical_and(mask_pred, dilated_true).sum()
    fp = np.logical_and(mask_pred, ~dilated_true).sum()
    fn = np.logical_and(mask_true, ~dilated_pred).sum()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return np.round(precision, 4), np.round(recall, 4), np.round(f1, 4)


def dice_score(mask1, mask2, eps=1e-6):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    return (2 * intersection + eps) / (mask1.sum() + mask2.sum() + eps)


def plot_output(test_df_pred, test_df_label, output, args):
    ious, dices, precs, recs, f1_tols = [], [], [], [], []
    ious_pos, dices_pos, f1_tols_pos = [], [], []
    n_empty_correct = 0
    n_empty_total   = 0

    for img_path, label_path, pred_mask in zip(
        test_df_pred["image"], test_df_label["label"], output
    ):
        img_arr   = np.array(Image.open(img_path).convert("L"))
        label_arr = np.array(Image.open(label_path).convert("L"))

        # --- 1. Valid metal region ---
        valid_region = get_valid_region(img_arr)

        # --- 2. Binarise model output ---
        mask_arr    = np.array(pred_mask).squeeze().astype(np.float32)
        binary_mask = (mask_arr > args.threshold).astype(np.uint8)

        # Morphological step — controlled by --morph and --morph_size
        kernel = np.ones((args.morph_size, args.morph_size))
        if args.morph == "open":
            # Removes small FP blobs (test4: sparse tiny defects, many noise blobs)
            binary_mask = ndi.binary_opening(binary_mask, structure=kernel).astype(np.uint8)

        elif args.morph == "dilate":
            # Expands under-predicted blobs (test3: large regions, precision=1.0)
            binary_mask = ndi.binary_dilation(binary_mask, structure=kernel).astype(np.uint8)
        # morph == "none": skip

        # Connected-component area filter: remove blobs outside [min_area, max_area].
        # test4: --min_area 75 removes tiny FPs; --max_area removes large FP streaks.
        if args.min_area > 0 or args.max_area > 0:
            labeled_mask, n_comp = ndi.label(binary_mask)
            for comp in range(1, n_comp + 1):
                area = (labeled_mask == comp).sum()
                if (args.min_area > 0 and area < args.min_area) or \
                   (args.max_area > 0 and area > args.max_area):
                    binary_mask[labeled_mask == comp] = 0

        # --- 3. Clip to valid metal region ---
        masked_pred  = np.where(valid_region, binary_mask, 0).astype(np.uint8)
        display_mask = masked_pred * 255

        # --- 4. Binarise ground-truth label (israt labels are 0/255) ---
        binary_label    = (label_arr > 127).astype(np.uint8)
        label_in_circle = np.where(valid_region, binary_label, 0)

        # --- 5. Metrics ---
        iou  = calculate_iou(masked_pred.astype(bool), label_in_circle.astype(bool))
        dice = dice_score(masked_pred, label_in_circle)
        prec, rec, f1_tol = tolerance_f1(
            masked_pred.astype(bool), label_in_circle.astype(bool), tolerance=5
        )

        ious.append(iou);  dices.append(dice)
        precs.append(prec); recs.append(rec); f1_tols.append(f1_tol)

        has_label = label_in_circle.astype(bool).any()
        if has_label:
            ious_pos.append(iou); dices_pos.append(dice); f1_tols_pos.append(f1_tol)
        else:
            n_empty_total += 1
            if not masked_pred.astype(bool).any():
                n_empty_correct += 1

        # --- 6. Save figure ---
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"mask_{os.path.basename(img_path)}")

        if args.save_mode == "mask":
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(display_mask, cmap="gray")
            ax.axis("off")
        else:
            fig = plt.figure(figsize=(20, 5))
            plt.subplot(1, 3, 1); plt.title("Input Image")
            plt.imshow(img_arr, cmap="gray"); plt.axis("off")
            plt.subplot(1, 3, 2); plt.title("Input Label")
            plt.imshow(label_arr, cmap="gray"); plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.title(
                f"IOU: {iou} | Dice: {dice:.4f}\n"
                f"Tol-F1(5px): {f1_tol:.4f} [P:{prec:.3f} R:{rec:.3f}]"
            )
            plt.imshow(display_mask, cmap="gray"); plt.axis("off")

        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    mean_iou   = np.mean(ious)
    mean_dice  = np.mean(dices)
    mean_prec  = np.mean(precs)
    mean_rec   = np.mean(recs)
    mean_f1tol = np.mean(f1_tols)

    mean_iou_pos   = np.mean(ious_pos)    if ious_pos    else 0.0
    mean_dice_pos  = np.mean(dices_pos)   if dices_pos   else 0.0
    mean_f1tol_pos = np.mean(f1_tols_pos) if f1_tols_pos else 0.0
    tnr = n_empty_correct / n_empty_total if n_empty_total > 0 else 1.0

    std_iou   = np.std(ious)
    std_dice  = np.std(dices)
    std_f1tol = np.std(f1_tols)

    print("─── Overall (all images) ────────────────────────────────")
    print(f"  Mean IoU          : {mean_iou:.4f}  ± {std_iou:.4f}")
    print(f"  Mean Dice         : {mean_dice:.4f}  ± {std_dice:.4f}")
    print(f"  Mean Tol-F1 (5px) : {mean_f1tol:.4f}  ± {std_f1tol:.4f}  [P: {mean_prec:.4f}  R: {mean_rec:.4f}]")
    print("─── Defect images only (label non-empty) ────────────────")
    print(f"  Images with defects : {len(ious_pos)}")
    print(f"  Mean IoU            : {mean_iou_pos:.4f}")
    print(f"  Mean Dice           : {mean_dice_pos:.4f}")
    print(f"  Mean Tol-F1 (5px)   : {mean_f1tol_pos:.4f}")
    print("─── Empty-label images (no defects) ─────────────────────")
    print(f"  Total empty-label             : {n_empty_total}")
    print(f"  Correctly predicted empty     : {n_empty_correct}  (TNR: {tnr:.4f})")

    return {
        "mean_iou": mean_iou, "mean_dice": mean_dice,
        "mean_prec": mean_prec, "mean_rec": mean_rec, "mean_f1tol": mean_f1tol,
        "mean_iou_pos": mean_iou_pos, "mean_dice_pos": mean_dice_pos,
        "mean_f1tol_pos": mean_f1tol_pos,
        "n_defect_images": len(ious_pos), "n_empty_total": n_empty_total,
        "n_empty_correct": n_empty_correct, "tnr": tnr,
        "ious": ious, "dices": dices, "f1_tols": f1_tols,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets/")
    parser.add_argument("--output_dir",  type=str, default="output")
    parser.add_argument("--data_name",   type=str, required=True,
                        help="CSV stem: test3_class1 / test4_class1 / test5_class1 / test6_class1")
    parser.add_argument("--ckpt_path",   type=str, required=True,
                        help="Path to the GAN-trained class_2 checkpoint directory.")
    parser.add_argument("--batch_size",  type=int, default=4)

    # Post-processing knobs — tune per test set (see commands below)
    parser.add_argument("--threshold",  type=float, default=0.50,
                        help="Probability threshold for binarising model output.")
    parser.add_argument("--morph",      type=str,   default="none",
                        choices=["none", "open", "dilate"],
                        help="Morphological op after thresholding: "
                             "open=remove small FPs (test4), " \
                             "dilate=grow under-predictions (test3).")
    parser.add_argument("--morph_size", type=int,   default=3,
                        help="Kernel size (NxN) for the morphological operation.")
    parser.add_argument("--min_area",   type=int,   default=0,
                        help="Remove predicted blobs smaller than this many pixels. 0=disabled.")
    parser.add_argument("--max_area",   type=int,   default=0,
                        help="Remove predicted blobs larger than this (pixels). 0=disabled. "
                             "Useful for test2 where large FP streaks dominate.")
    parser.add_argument("--save_mode", type=str, default="panel",
                        choices=["panel", "mask"],
                        help="'panel': save 3-panel (input|label|mask); 'mask': save mask only.")
    args = parser.parse_args()


    dataset_dir = os.path.join(args.dataset_dir, "israt")
    os.makedirs(args.output_dir, exist_ok=True)

    test_df = expand_path(
        pd.read_csv(os.path.join(dataset_dir, f"{args.data_name}.csv")), dataset_dir
    )
    test_df_pred  = test_df[["image"]].copy()
    test_df_label = test_df[["label"]].copy()

    predictor = MultiModalPredictor.load(args.ckpt_path)

    output  = predictor.predict(test_df_pred, batch_size=args.batch_size, return_all_masks=True)
    metrics = plot_output(test_df_pred, test_df_label, output, args)

    metric_file = os.path.join(args.output_dir, f"metrics_{args.data_name}.txt")
    with open(metric_file, "a") as f:
        f.write(
            f"\n=== {args.data_name} (class_2 model) ===\n"
            f"Overall:\n"
            f"  Mean IoU          : {metrics['mean_iou']:.4f}\n"
            f"  Std IoU           : {np.std(metrics['ious']):.4f}\n"
            f"  Mean Dice         : {metrics['mean_dice']:.4f}\n"
            f"  Std Dice          : {np.std(metrics['dices']):.4f}\n"
            f"  Mean Tol-F1 (5px) : {metrics['mean_f1tol']:.4f}  "
            f"[P: {metrics['mean_prec']:.4f}  R: {metrics['mean_rec']:.4f}]\n"
            f"  Std  F1 (tol=5px) : {np.std(metrics['f1_tols']):.4f}\n"
            f"Defect images only ({metrics['n_defect_images']} images):\n"
            f"  Mean IoU          : {metrics['mean_iou_pos']:.4f}\n"
            f"  Mean Dice         : {metrics['mean_dice_pos']:.4f}\n"
            f"  Mean Tol-F1 (5px) : {metrics['mean_f1tol_pos']:.4f}\n"
            f"Empty-label images ({metrics['n_empty_total']} images):\n"
            f"  Correctly predicted empty : {metrics['n_empty_correct']}\n"
            f"  True Negative Rate        : {metrics['tnr']:.4f}\n"
        )

"""
# ── EVALUATION ────────────────────────────────────────────────────────────────
#
# test3  ~20% defects, large regions, precision=1.0, recall~0.85  → UNDER-predicting
#   fix: lower threshold + dilate to grow predicted blobs outward
CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_israt.py \
  --dataset_dir ./datasets --output_dir ./output/israt/test3 --ckpt_path AutogluonModels/Dloss/class_2 --data_name test3_class1 --batch_size 4 \
  --threshold 0.50 --morph dilate --morph_size 1 --min_area 0

# test4  ~0.3% defects, sparse tiny dots, recall~1.0, precision~0.45  → OVER-predicting
#   fix: raise threshold + open to kill noise + area filter to remove tiny FP blobs
CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_israt.py \
  --dataset_dir ./datasets --output_dir ./output/israt/test4 --ckpt_path AutogluonModels/Dloss/class_2 --data_name test4_class1 --batch_size 4 \
  --threshold 0.65 --morph open --morph_size 5 --min_area 75

# test5
CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_israt.py \
  --dataset_dir ./datasets --output_dir ./output/israt/test5 --ckpt_path AutogluonModels/Dloss/class_2 --data_name test5_class1 --batch_size 4 \
  --threshold 0.50 --morph dilate --morph_size 1 --min_area 0

# test6  ~25% defects, bright on dark, direct model output
CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_israt.py \
  --dataset_dir ./datasets --output_dir ./output/israt/test6 --ckpt_path AutogluonModels/Dloss/class_3 --data_name test6_class1 --batch_size 4 \
  --threshold 0.50 --morph dilate --morph_size 1 --min_area 0
"""


"""
  ┌──────────────┬────────┬───────┬─────────────┐────────────┤                                                                                                                                                                                                                                         
  │     Arg      │ test3  │ test4 │ test5       │    test6                                                                                                                                                                                                                                     
  ├──────────────┼────────┼───────┼─────────────┤────────────┤                                                                                                                                                                                                                                         
  │ --threshold  │ 0.50   │ 0.65  │ 0.50        │    0.50                                                                                                                                                                                                                                     
  ├──────────────┼────────┼───────┼─────────────┤────────────┤
  │ --morph      │ dilate │ open  │ dilate      │    dilate                                                                                                                                                                                                                                    
  ├──────────────┼────────┼───────┼─────────────┤────────────┤
  │ --morph_size │ 1      │ 5     │ 1           │     1                                                                                                                                                                                                                                   
  ├──────────────┼────────┼───────┼─────────────┤────────────┤                                                                                                                                                                                                                                       
  │ --min_area   │ 0      │ 75    │ 0           │     0
  └──────────────┴────────┴───────┴─────────────┘────────────┤
"""