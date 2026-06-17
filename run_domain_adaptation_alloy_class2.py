import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
import numpy as np
from autogluon.multimodal import MultiModalPredictor


# ---------------------------------------------------------------------------
# Alloy dataset — class_2 intermediate domain adaptation
# (dark defects on bright alloy surface, P92HBN specimens)
#
# Differences vs. run_semantic_segmentation_class_2.py (XCT/GAN):
#   1. No circular specimen border — full frame is valid metal.
#      get_valid_region returns an all-ones mask; no Otsu/circle logic needed.
#   2. Labels are 0/255 PNG → binarised with > 127.
#   3. Validation set is val_class2.csv (15% random split, seed=42).
#   4. DiceFocalLoss: positive pixels ~0.08-0.21%, same sparsity as GAN-XCT.
#   5. lr=2e-4 (conservative): larger domain gap SAM→alloy than alloy→XCT;
#      preserves SAM's pretrained features while adapting LoRA weights to
#      material-microstructure texture and sparse defect detection.
# ---------------------------------------------------------------------------

MAX_EPOCH = 15
LR        = 1e-4
LOSS      = "dice_focal_loss"


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def get_valid_region(img_arr):
    """
    Alloy images fill the entire frame with no black background or circular
    border. Return a full-image boolean mask so downstream evaluation code
    is identical to the XCT pipeline.
    """
    return np.ones(img_arr.shape[:2], dtype=bool)


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

        # --- Post-process prediction ---
        mask_arr    = np.array(pred_mask).squeeze().astype(np.float32)
        binary_mask = (mask_arr > args.threshold).astype(np.uint8)

        kernel = np.ones((args.morph_size, args.morph_size))
        if args.morph == "open":
            binary_mask = ndi.binary_opening(binary_mask, structure=kernel).astype(np.uint8)
        elif args.morph == "dilate":
            binary_mask = ndi.binary_dilation(binary_mask, structure=kernel).astype(np.uint8)

        if args.min_area > 0 or args.max_area > 0:
            labeled_mask, n_comp = ndi.label(binary_mask)
            for comp in range(1, n_comp + 1):
                area = (labeled_mask == comp).sum()
                if (args.min_area > 0 and area < args.min_area) or \
                   (args.max_area > 0 and area > args.max_area):
                    binary_mask[labeled_mask == comp] = 0

        # No circular masking — alloy images are full-frame
        masked_pred  = binary_mask
        display_mask = masked_pred * 255

        # Alloy labels are 0/255 PNG
        binary_label = (label_arr > 127).astype(np.uint8)

        iou = calculate_iou(masked_pred.astype(bool), binary_label.astype(bool))
        ious.append(iou)

        dice = dice_score(masked_pred, binary_label)
        dices.append(dice)

        prec, rec, f1_tol = tolerance_f1(
            masked_pred.astype(bool), binary_label.astype(bool), tolerance=5
        )
        precs.append(prec)
        recs.append(rec)
        f1_tols.append(f1_tol)

        has_label = binary_label.astype(bool).any()
        if has_label:
            ious_pos.append(iou)
            dices_pos.append(dice)
            f1_tols_pos.append(f1_tol)
        else:
            n_empty_total += 1
            if not masked_pred.astype(bool).any():
                n_empty_correct += 1

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(img_arr, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Label")
        plt.imshow(label_arr, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title(
            f"IOU: {iou} | Dice: {dice:.4f}\n"
            f"Tol-F1(5px): {f1_tol:.4f} [P:{prec:.3f} R:{rec:.3f}]"
        )
        plt.imshow(display_mask, cmap="gray")
        plt.axis("off")

        file = os.path.basename(img_path)
        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(args.output_dir, f"mask_{file}"),
            bbox_inches="tight", pad_inches=0,
        )
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

    print("─── Overall (all images) ────────────────────────────────")
    print(f"  Mean IoU          : {mean_iou:.4f}")
    print(f"  Mean Dice         : {mean_dice:.4f}")
    print(f"  Mean Tol-F1 (5px) : {mean_f1tol:.4f}  [P: {mean_prec:.4f}  R: {mean_rec:.4f}]")
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
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alloy class_2 intermediate domain adaptation (Stage 1)"
    )
    parser.add_argument("--seed",               type=int,   default=61)
    parser.add_argument("--lora_rank",          type=int,   default=2,   help="Conv-LoRA rank.")
    parser.add_argument("--rank",               type=int,   default=None, help="Distributed computing rank (set by launcher, not user).")
    parser.add_argument("--lr",                 type=float, default=LR,  help="Learning rate.")
    parser.add_argument("--loss_func",          type=str,   default=None,
                        choices=["structure_loss", "dice_focal_loss", "bce_loss",
                                 "focal_loss", "dice_loss", "lovasz_loss"],
                        help="Loss function. Overrides the script default if given.")
    parser.add_argument("--expert_num",         type=int,   default=8)
    parser.add_argument("--num_gpus",           type=int,   default=1)
    parser.add_argument("--dataset_dir",        type=str,   default="datasets/alloy")
    parser.add_argument("--output_dir",         type=str,   default="output/alloy/class2")
    parser.add_argument("--data_name",          type=str,   default="val_class2",
                        help="CSV stem to evaluate on after training (e.g. val_class2, train_class2).")
    
    parser.add_argument("--ckpt_path",          type=str,   default="AutogluonModels/alloy/class_2",
                        help="Path to save (train) the best model, or load (--eval) a checkpoint.")
    parser.add_argument("--ckpt_path_last",     type=str,   default="AutogluonModels/alloy/class_2_last",
                        help="Path to save the last-epoch model weights after training.")
    parser.add_argument("--per_gpu_batch_size", type=int,   default=1)
    parser.add_argument("--batch_size",         type=int,   default=4,
                        help="Effective batch size; gradient accumulation used if needed.")
    # post-processing knobs
    parser.add_argument("--threshold",  type=float, default=0.50,
                        help="Sigmoid threshold for binarising the predicted mask.")
    parser.add_argument("--morph",      type=str,   default="none",
                        choices=["none", "open", "dilate"],
                        help="Morphological post-processing operation.")
    parser.add_argument("--morph_size", type=int,   default=3,
                        help="Kernel size for the morphological operation.")
    parser.add_argument("--min_area",   type=int,   default=0,
                        help="Remove connected components smaller than this many pixels.")
    parser.add_argument("--max_area",   type=int,   default=0,
                        help="Remove connected components larger than this (pixels). 0=disabled.")
    parser.add_argument("--eval",       action="store_true",
                        help="Skip training; load --ckpt_path and evaluate on --data_name.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = expand_path(
        pd.read_csv(os.path.join(dataset_dir, "train_class2.csv")), dataset_dir
    )
    val_df = expand_path(
        pd.read_csv(os.path.join(dataset_dir, "val_class2.csv")), dataset_dir
    )

    hyperparameters = {
        "optim.lora.r":                    args.lora_rank,
        "optim.peft":                      "conv_lora",
        "optim.lora.conv_lora_expert_num": args.expert_num,
        "env.num_gpus":                    args.num_gpus,
        "optim.loss_func":                 args.loss_func if args.loss_func is not None else LOSS,
        "optim.max_epochs":                MAX_EPOCH,
        "optim.lr":                        args.lr,
        "optim.patience":                  MAX_EPOCH,
        "env.per_gpu_batch_size":          args.per_gpu_batch_size,
        "env.batch_size":                  args.batch_size,
    }

    if args.eval:
        predictor = MultiModalPredictor.load(args.ckpt_path)
    else:
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric="iou",
            eval_metric="iou",
            hyperparameters=hyperparameters,
            label="label",
        )
        predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            seed=args.seed,
            save_path=args.ckpt_path,
        )
        # Save last-epoch weights separately (best is already at ckpt_path).
        print(f"\nSaving last-epoch model to: {args.ckpt_path_last}")
        predictor.save(args.ckpt_path_last)

    # Evaluate on --data_name CSV immediately after training (or when --eval)
    eval_df = expand_path(
        pd.read_csv(os.path.join(dataset_dir, f"{args.data_name}.csv")), dataset_dir
    )
    test_df_pred  = eval_df[["image"]].copy()
    test_df_label = eval_df[["label"]].copy()

    output  = predictor.predict(test_df_pred, batch_size=1, return_all_masks=True)
    metrics = plot_output(test_df_pred, test_df_label, output, args)

    metric_file = os.path.join(args.output_dir, f"metrics_{args.data_name}.txt")
    with open(metric_file, "a") as f:
        summary = (
            f"\n=== {args.data_name} ===\n"
            f"Overall:\n"
            f"  Mean IoU          : {metrics['mean_iou']:.4f}\n"
            f"  Mean Dice         : {metrics['mean_dice']:.4f}\n"
            f"  Mean Tol-F1 (5px) : {metrics['mean_f1tol']:.4f}  "
            f"[P: {metrics['mean_prec']:.4f}  R: {metrics['mean_rec']:.4f}]\n"
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

# --- Train (intermediate domain adaptation, Stage 1) ---
# Best model → AutogluonModels/alloy/class_2
# Last model → AutogluonModels/alloy/class_2_last
# python3 run_domain_adaptation_alloy_class2.py --dataset_dir datasets/alloy --output_dir  output/alloy/class2/val --data_name   train_class2 --num_gpus 2 --batch_size 8 --rank 2


# --- Evaluate last checkpoint ---
# CUDA_VISIBLE_DEVICES=0 python3 run_domain_adaptation_alloy_class2.py.py \
#   --dataset_dir datasets/alloy \
#   --ckpt_path   AutogluonModels/alloy/class_2_last \
#   --output_dir  output/alloy/class2/val_last \
#   --data_name   val_class2 \
#   --num_gpus 1 --eval
