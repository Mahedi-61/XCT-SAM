import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from PIL import Image
import numpy as np
from autogluon.multimodal import MultiModalPredictor
from skimage import exposure
import torch 
torch.set_float32_matmul_precision('high')
# ---------------------------------------------------------------------------
# Stage 2: XCT domain fine-tuning, warm-started from alloy Stage-1 checkpoint.
#
# Changes vs. run_semantic_segmentation_class_2.py (cold-start):
#   1. Loads alloy-adapted checkpoint (--alloy_ckpt_path) before fitting,
#      bridging the domain gap SAM-ViT-H → alloy steel → XCT.
#   2. lr=1e-4  (vs 3e-4 cold): lower rate preserves alloy-adapted features.
#   3. max_epoch=20 (vs 30): warm start converges faster on XCT train set.
#   4. patience=8: allow early stopping once XCT val-IoU plateaus.
#   5. loss=dice_focal_loss: same sparse-defect imbalance as cold-start.
# ---------------------------------------------------------------------------

MAX_EPOCH = 25
LR        = 2e-4
PATIENCE  = 10

ALLOY_CKPT_DEFAULT = "AutogluonModels/alloy/class_2"
XCT_CKPT_DEFAULT   = "AutogluonModels/alloy_to_xct/class_2"


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def get_valid_region(img_arr):
    otsu_val, _ = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = otsu_val if otsu_val > 1 else img_arr.max() * 0.10
    mask = img_arr > threshold

    # When Otsu finds the disc fills >78% of the frame the threshold is
    # splitting inside the disc (dim edge regions fall below Otsu).
    # Scan from 30%→10% of max to find a threshold that keeps coverage ≤ 0.92,
    # preventing dark background pixels (intensity 30-60) from being included
    # when 10% of max is too low (e.g. test5 where background isn't very dark).
    if mask.mean() > 0.78:
        for pct in [0.30, 0.25, 0.20, 0.15, 0.10]:
            mask = img_arr > (img_arr.max() * pct)
            if mask.mean() <= 0.92:
                break

    # Large closing kernel bridges internal dark bands; fill_holes closes voids.
    mask_filled = ndi.binary_closing(mask, structure=np.ones((15, 15)))
    mask_filled = ndi.binary_fill_holes(mask_filled)

    contours, _ = cv2.findContours(mask_filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    # Fill the largest contour directly — no hull or circle fitting.
    # Otsu already separates bright metal from dark background corners, so the
    # filled contour follows the actual metal boundary without overestimating
    # into dark corners (which hull/circle can do for cut-off partial discs).
    metal_mask = np.zeros(img_arr.shape[:2], dtype=np.uint8)
    cv2.drawContours(metal_mask, [cnt], -1, 1, thickness=cv2.FILLED)
    return metal_mask.astype(bool)


def get_input_img_mask(img_arr, valid_region):
    mask_arr = np.array(img_arr).astype(np.float32)
    threshold = np.mean(mask_arr)
    binary_img_mask = (mask_arr < threshold).astype(np.uint8)
    masked_pred = np.where(valid_region, binary_img_mask, 0).astype(np.uint8)
    display_img_mask = masked_pred * 255
    return display_img_mask, binary_img_mask


def calculate_iou(mask_pred, mask_true):
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union        = np.logical_or(mask_true, mask_pred).sum()
    if union == 0:
        return 1.0
    return np.round(intersection / union, 4)


def tolerance_f1(mask_pred, mask_true, tolerance=5):
    if mask_true.sum() == 0 and mask_pred.sum() == 0:
        return 1.0, 1.0, 1.0
    if mask_true.sum() == 0 or mask_pred.sum() == 0:
        return 0.0, 0.0, 0.0

    struct       = ndi.generate_binary_structure(2, 1)
    dilated_true = ndi.binary_dilation(mask_true, structure=struct, iterations=tolerance)
    dilated_pred = ndi.binary_dilation(mask_pred, structure=struct, iterations=tolerance)

    tp = np.logical_and(mask_pred,  dilated_true).sum()
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
    ious_pos, dices_pos, f1_tols_pos  = [], [], []
    n_empty_correct = 0
    n_empty_total   = 0

    for img_path, label_path, pred_mask in zip(
        test_df_pred["image"], test_df_label["label"], output
    ):
        img   = Image.open(img_path).convert("L")
        label = Image.open(label_path).convert("L")

        img_arr   = np.array(img)
        label_arr = np.array(label)

        valid_region = get_valid_region(img_arr)
        if args.edge_margin > 0:
            valid_region = ndi.binary_erosion(
                valid_region, structure=np.ones((args.edge_margin * 2 + 1,
                                                args.edge_margin * 2 + 1))
            )

        mask_arr    = np.array(pred_mask).squeeze().astype(np.float32)
        binary_mask = (mask_arr > args.threshold).astype(np.uint8)

        kernel = np.ones((args.morph_size, args.morph_size))
        if args.morph == "open":
            binary_mask = ndi.binary_opening(binary_mask, structure=kernel).astype(np.uint8)
        elif args.morph == "dilate":
            binary_mask = ndi.binary_dilation(binary_mask, structure=kernel).astype(np.uint8)
        elif args.morph == "close":
            binary_mask = ndi.binary_closing(binary_mask, structure=kernel).astype(np.uint8)

        masked_pred  = np.where(valid_region, binary_mask, 0).astype(np.uint8)

        # Remove tiny noise components from the final masked prediction.
        # Filtering after valid_region masking ensures only in-region predictions
        # are evaluated — spurious edge dots (model fires near metal boundary)
        # are killed here before IoU/Dice computation.
        if args.min_area > 0 or args.max_area > 0:
            labeled_masked, n_comp = ndi.label(masked_pred)
            for comp in range(1, n_comp + 1):
                area = (labeled_masked == comp).sum()
                if (args.min_area > 0 and area < args.min_area) or \
                   (args.max_area > 0 and area > args.max_area):
                    masked_pred[labeled_masked == comp] = 0

        display_mask = masked_pred * 255

        binary_label   = label_arr.astype(np.uint8)
        label_in_circle = np.where(valid_region, binary_label, 0)

        # Detect when valid_region masked out real label pixels — metrics
        # must reflect a missed detection, not a correct true-negative.
        label_orig_has_defect = binary_label.astype(bool).any()
        label_masked_empty    = not label_in_circle.astype(bool).any()
        valid_region_missed   = label_orig_has_defect and label_masked_empty

        if valid_region_missed:
            iou, dice, prec, rec, f1_tol = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            iou = calculate_iou(masked_pred.astype(bool), label_in_circle.astype(bool))
            dice = dice_score(masked_pred, label_in_circle)
            prec, rec, f1_tol = tolerance_f1(
                masked_pred.astype(bool), label_in_circle.astype(bool), tolerance=5
            )

        ious.append(iou)
        dices.append(dice)
        precs.append(prec)
        recs.append(rec)
        f1_tols.append(f1_tol)

        has_label = label_in_circle.astype(bool).any() or valid_region_missed
        if has_label:
            ious_pos.append(iou)
            dices_pos.append(dice)
            f1_tols_pos.append(f1_tol)
        else:
            n_empty_total += 1
            if not masked_pred.astype(bool).any():
                n_empty_correct += 1

        file = os.path.basename(img_path)
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"mask_{file}")

        if args.save_mode == "mask":
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(display_mask, cmap="gray")
            ax.axis("off")
        else:
            fig = plt.figure(figsize=(20, 5))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(img_arr, cmap="gray")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.title("Input Label")
            plt.imshow(label_arr, cmap="gray")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.title(
                f"IOU: {iou} | Dice: {dice:.4f}\n"
                f"Tol-F1(5px): {f1_tol:.4f} [P:{prec:.3f} R:{rec:.3f}]"
            )
            plt.imshow(display_mask, cmap="gray")
            plt.axis("off")

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
        "n_defect_images": len(ious_pos),
        "n_empty_total": n_empty_total, "n_empty_correct": n_empty_correct,
        "tnr": tnr,
        "ious": ious, "dices": dices, "f1_tols": f1_tols,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: XCT fine-tuning warm-started from alloy Stage-1 checkpoint."
    )
    parser.add_argument("--seed",               type=int, default=42686693)
    parser.add_argument("--lora_rank",          type=int, default=2,    help="Conv-LoRA rank.")
    parser.add_argument("--rank",               type=int, default=None, help="Distributed computing rank (set by launcher, not user).")
    parser.add_argument("--lr",                 type=float, default=LR, help="Learning rate.")
    parser.add_argument("--loss_func",          type=str, default="dice_focal_loss",
                        choices=["structure_loss", "dice_focal_loss", "bce_loss",
                                 "focal_loss", "dice_loss", "focal_tversky_loss", "lovasz_loss"],
                        help="Loss function for training.")
    parser.add_argument("--expert_num",         type=int, default=8)
    parser.add_argument("--num_gpus",           type=int, default=1)
    parser.add_argument("--dataset_dir",        type=str, default="datasets/gan-generated")
    parser.add_argument("--output_dir",         type=str, default="output/alloy_to_xct/class_2")
    parser.add_argument("--data_name",          type=str, default="train_class2")
    parser.add_argument("--val_data",           type=str, default="test4_class2",
                        help="CSV stem used as validation during training (must be held-out from train).")
    parser.add_argument("--alloy_ckpt_path",    type=str, default=ALLOY_CKPT_DEFAULT,
                        help="Stage-1 alloy checkpoint to warm-start from.")
    parser.add_argument("--ckpt_path",          type=str, default=XCT_CKPT_DEFAULT,
                        help="Path where the trained checkpoint is saved/loaded.")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--batch_size",         type=int, default=4)
    parser.add_argument("--eval",               action="store_true",
                        help="Skip training; load --ckpt_path and evaluate on --data_name.")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Binarisation threshold for predicted mask probabilities.")
    parser.add_argument("--morph",      type=str,   default="none",
                        choices=["none", "open", "dilate", "close"],
                        help="Post-processing morphological operation.")
    parser.add_argument("--morph_size", type=int,   default=1,
                        help="Kernel size for morphological operation.")
    parser.add_argument("--min_area",   type=int,   default=0,
                        help="Remove connected components smaller than this (pixels).")
    parser.add_argument("--max_area",   type=int,   default=0,
                        help="Remove connected components larger than this (pixels). 0=disabled. "
                             "Useful to kill large FP blobs when GT pores are small.")
    parser.add_argument("--edge_margin", type=int,  default=0,
                        help="Erode valid region by this many pixels to exclude boundary noise.")
    parser.add_argument("--save_mode", type=str, default="panel",
                        choices=["panel", "mask"],
                        help="'panel': save 3-panel (input|label|mask); 'mask': save mask only.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = expand_path(
        pd.read_csv(os.path.join(dataset_dir, f"{args.data_name}.csv")), dataset_dir
    )
    val_df = expand_path(
        pd.read_csv(os.path.join(dataset_dir, f"{args.val_data}.csv")), dataset_dir
    )

    hyperparameters = {
        "optim.lora.r":                    args.lora_rank,
        "optim.peft":                      "conv_lora",
        "optim.lora.conv_lora_expert_num": args.expert_num,
        "env.num_gpus":                    args.num_gpus,
        "optim.loss_func":                 args.loss_func,
        "optim.max_epochs":                MAX_EPOCH,
        "optim.lr":                        args.lr,
        "optim.patience":                  PATIENCE,
        "env.per_gpu_batch_size":          args.per_gpu_batch_size,
        "env.batch_size":                  args.batch_size,
        "optim.val_check_interval":        1.0,   # validate once per epoch
        "optim.check_val_every_n_epoch":   1,     # every epoch
    }

    if args.eval:
        predictor = MultiModalPredictor.load(args.ckpt_path)
    else:
        # Load Stage-1 alloy checkpoint, then fine-tune on XCT data.
        print(f"Loading Stage-1 alloy checkpoint from: {args.alloy_ckpt_path}")
        print(f"Checkpoint path  → {args.ckpt_path}")
        predictor = MultiModalPredictor.load(args.alloy_ckpt_path)
        predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            seed=args.seed,
            save_path=args.ckpt_path,
            hyperparameters=hyperparameters,
        )

    # Evaluate
    test_df = expand_path(
        pd.read_csv(os.path.join(dataset_dir, f"{args.data_name}.csv")), dataset_dir
    )
    test_df_pred  = test_df[["image"]].copy()
    test_df_label = test_df[["label"]].copy()

    output  = predictor.predict(test_df_pred, batch_size=1, return_all_masks=True)
    metrics = plot_output(test_df_pred, test_df_label, output, args)

    metric_file = os.path.join(args.output_dir, f"metrics_{args.data_name}.txt")
    with open(metric_file, "a") as f:
        summary = (
            f"\n=== {args.data_name} ===\n"
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
        f.write(summary)
    print(f"\nMetrics written to: {metric_file}")


# =============================================================================
# USAGE
# =============================================================================

# --- Train Stage 2 (warm-start from alloy, fine-tune on XCT GAN data) ---
# CUDA_VISIBLE_DEVICES=0,1 python3 run_semantic_segmentation_alloy_class_2.py --loss bce_loss --dataset_dir datasets/gan-generated --data_name train_class2 --output_dir  output/alloy_to_xct_bce/class_2/train --alloy_ckpt_path AutogluonModels/alloy/class_2 --ckpt_path   AutogluonModels/alloy_to_xct_bce/class_2 --num_gpus 2 --batch_size 8 --rank 2

# --- Evaluate Stage-2 model ---
# CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_alloy_class_2.py \
#   --dataset_dir datasets/gan-generated \
#   --ckpt_path   AutogluonModels/alloy_to_xct/class_2 \
#   --output_dir  output/gan_adapt/test6_c2 \
#   --data_name   test6_class2 \
#   --num_gpus 1 --batch_size 4 --rank 2 --eval \
#   --threshold 0.50 --morph none --morph_size 1 --min_area 0
#
# test5 (partially-visible disc images): filter tiny FP dots
# CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_alloy_class_2.py \
#   --dataset_dir datasets/gan-generated \
#   --ckpt_path   AutogluonModels/alloy_to_xct/class_2 \
#   --output_dir  output/gan_adapt/test5_c2 \
#   --data_name   test5_class2 \
#   --num_gpus 1 --batch_size 4 --rank 2 --eval \
#   --threshold 0.50 --morph open --morph_size 3 --min_area 50


