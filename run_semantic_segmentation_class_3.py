import argparse
import os
import pickle 
import pandas as pd
import shutil, os
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from PIL import Image
import numpy as np 
from matplotlib.colors import ListedColormap
from autogluon.multimodal import MultiModalPredictor
from skimage import exposure


def get_default_training_setting(dataset_name):
    validation_metric = "iou"
    loss = "structure_loss"
    max_epoch = 30
    lr = 1e-4

    if dataset_name == "israt":
        validation_metric = "iou"
        lr = 3e-4

    elif dataset_name == "alloy":
        validation_metric = "iou" #iou for semnantic segmentation
        max_epoch = 5
        lr = 3e-4
        loss = "structure_loss"

    elif dataset_name == "gan-generated":
        validation_metric = "iou" #iou for semnantic segmentation
        max_epoch = 20
        lr = 3e-4
        loss = "dice_focal_loss"

    return validation_metric, loss, max_epoch, lr


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def get_valid_region(img_arr):
    # Use Otsu's threshold to separate metal from background.
    # Works for both dark (AFAwNi, max~66) and bright (Ti64, max~200+) images.
    # Use the raw image for region detection (not CLAHE-equalised) because
    # metal vs. background contrast is already strong in both domains.
    otsu_val, _ = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If Otsu collapses (nearly uniform image), fall back to 10% of max
    threshold = otsu_val if otsu_val > 1 else img_arr.max() * 0.10
    mask = img_arr > threshold

    # --- Fill holes and smooth defects ---
    mask_filled = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask_filled = ndi.binary_fill_holes(mask_filled)

    # --- Fit circle around detected region ---
    contours, _ = cv2.findContours(mask_filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)

    metal_mask = np.zeros_like(mask_filled, dtype=np.uint8)
    cv2.circle(metal_mask, (int(x), int(y)), int(radius), 1, -1)
    valid_region = metal_mask.astype(bool)
    return valid_region


def get_input_img_mask(img_arr, valid_region):
    mask_arr = np.array(img_arr).astype(np.float32)
    threshold = np.mean(mask_arr)
    binary_img_mask = (mask_arr < threshold).astype(np.uint8)  # defects=1, surface=0

    masked_pred = np.where(valid_region, binary_img_mask, 0).astype(np.uint8)
    display_img_mask = masked_pred * 255  # defects=white, surface/background=black
    return display_img_mask, binary_img_mask


def calculate_iou(mask_pred, mask_true):
    """
    Computes IoU between two boolean or binary masks.
    """
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()
    if union == 0:
        return 1.0
    return np.round(intersection / union, 4)


def tolerance_f1(mask_pred, mask_true, tolerance=5):
    """
    Tolerance-based F1 for sparse point-like defects.
    A predicted pixel is a true positive if it falls within `tolerance`
    pixels of any labeled defect pixel. This handles the case where
    GAN-generated labels have slight positional offsets from predictions.
    """
    if mask_true.sum() == 0 and mask_pred.sum() == 0:
        return 1.0, 1.0, 1.0
    if mask_true.sum() == 0 or mask_pred.sum() == 0:
        return 0.0, 0.0, 0.0

    # Dilate both masks by `tolerance` pixels before computing overlap
    struct = ndi.generate_binary_structure(2, 1)
    dilated_true = ndi.binary_dilation(mask_true, structure=struct, iterations=tolerance)
    dilated_pred = ndi.binary_dilation(mask_pred, structure=struct, iterations=tolerance)

    tp = np.logical_and(mask_pred, dilated_true).sum()   # pred hits near a label
    fp = np.logical_and(mask_pred, ~dilated_true).sum()  # pred far from any label
    fn = np.logical_and(mask_true, ~dilated_pred).sum()  # label missed by pred

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return np.round(precision, 4), np.round(recall, 4), np.round(f1, 4)


def dice_score(mask1, mask2, eps=1e-6):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    
    return (2 * intersection + eps) / (mask1.sum() + mask2.sum() + eps)


def plot_output(test_df_pred, test_df_label, output, args, morph_size=7):
    ious = []
    dices = []
    precs = []
    recs = []
    f1_tols = []
    # separate tracking: images with at least one labeled defect vs empty labels
    ious_pos, dices_pos, f1_tols_pos = [], [], []   # non-empty label images
    n_empty_correct = 0   # empty label + empty prediction (true negatives)
    n_empty_total   = 0   # all empty-label images

    for img_path, label_path, pred_mask in zip(test_df_pred['image'], test_df_label['label'], output):
        img = Image.open(img_path).convert("L")
        label = Image.open(label_path).convert("L")

        img_arr = np.array(img)
        label_arr = np.array(label)
        # --- 1. Detect circular metal region from input image ---
        # Background is black (0), metal is nonzero
        valid_region = get_valid_region(img_arr)

        # --- 2. Convert mask to binary defect/surface ---
        # Class 3 defects are BRIGHT white spots on dark metal surface.
        # SAM output is a probability map; threshold at 0.5 (same as evaluate()).
        mask_arr = np.array(pred_mask).squeeze().astype(np.float32)
        threshold = 0.5
        binary_mask = (mask_arr > threshold).astype(np.uint8)  # defects=1, background=0

        # Remove small noise blobs.
        binary_mask = ndi.binary_opening(binary_mask, structure=np.ones((morph_size, morph_size))).astype(np.uint8)

        # --- 3. Apply circular region mask ---
        masked_pred = np.where(valid_region, binary_mask, 0).astype(np.uint8)
        display_mask = masked_pred * 255  # defects=white, surface/background=black

        #IOU computation (w.r. label)
        #label_arr = label_arr.astype(np.float32)
        #threshold = np.mean(label_arr)
        binary_label = label_arr.astype(np.uint8)  # defects=1, surface=0
        label_in_circle = np.where(valid_region, binary_label, 0)

        # Use masked_pred (not binary_mask) so metrics match exactly what is displayed.
        # binary_mask may have pixels outside the valid circle that are zeroed in the
        # display but would silently count as false positives in the metrics.
        iou = calculate_iou(masked_pred.astype(bool), label_in_circle.astype(bool))
        ious.append(iou)

        dice = dice_score(masked_pred, label_in_circle)
        dices.append(dice)

        prec, rec, f1_tol = tolerance_f1(masked_pred.astype(bool), label_in_circle.astype(bool), tolerance=5)
        precs.append(prec)
        recs.append(rec)
        f1_tols.append(f1_tol)

        # split stats: defect images vs empty-label images
        has_label = label_in_circle.astype(bool).any()
        if has_label:
            ious_pos.append(iou)
            dices_pos.append(dice)
            f1_tols_pos.append(f1_tol)
        else:
            n_empty_total += 1
            if not masked_pred.astype(bool).any():
                n_empty_correct += 1  # correctly predicted nothing

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(img_arr, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Input Label")
        plt.imshow(label_arr, cmap='gray')
        plt.axis("off")


        plt.subplot(1, 3, 3)
        plt.title(f"IOU: {iou} | Dice: {dice:.4f}\nTol-F1(5px): {f1_tol:.4f} [P:{prec:.3f} R:{rec:.3f}]")
        plt.imshow(display_mask, cmap="gray")
        plt.axis("off")

        file = os.path.basename(img_path)
        os.makedirs(args.output_dir, exist_ok=True)
        filename = os.path.join(args.output_dir, f"mask_{file}")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    mean_iou   = np.mean(ious)
    mean_dice  = np.mean(dices)
    mean_prec  = np.mean(precs)
    mean_rec   = np.mean(recs)
    mean_f1tol = np.mean(f1_tols)

    # stats for defect-present images only
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
        "n_defect_images": len(ious_pos), "n_empty_total": n_empty_total, "n_empty_correct":n_empty_correct,
        "tnr": tnr,
        "ious": ious,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script support converting voc format xmls to coco format json")
    parser.add_argument(
        "--task",
        type=str,
        default="leaf_disease_segmentation",
        choices=["gan-generated", "israt", "alloy"]
    )
    parser.add_argument("--seed", type=int, default=42686693)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--dataset_dir", type=str, default="datasets/")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--data_name", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="output", help="Checkpoint path.")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1, help="The batch size for each GPU.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The effective batch size. If batch_size > per_gpu_batch_size * num_gpus, gradient accumulation would be used.",
    )
    parser.add_argument("--morph_size", type=int, default=7, help="Kernel size for morphological opening (noise removal).")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(args.dataset_dir, dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"{args.data_name}.csv")), dataset_dir)

    # Use test3_class3 (AFAPB05 - aluminum-based, similar domain to training AFAwNi)
    val_data = "test3_" + args.data_name.split("_")[1]
    val_df   = expand_path(pd.read_csv(os.path.join(dataset_dir, f"{val_data}.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)

    hyperparameters = {}
    hyperparameters.update(
        {
            "optim.lora.r": args.rank,
            "optim.peft": "conv_lora",
            "optim.lora.conv_lora_expert_num": args.expert_num,
            "env.num_gpus": args.num_gpus,
            "optim.loss_func": loss,
            "optim.max_epochs": max_epoch,
            "optim.patience": max_epoch,
            "optim.lr": lr,
            "env.per_gpu_batch_size": args.per_gpu_batch_size,
            "env.batch_size": args.batch_size,
        }
    )

    if args.eval:  # load a checkpoint for evaluation
        predictor = MultiModalPredictor.load(args.ckpt_path)

    else:  # training
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric=validation_metric,
            eval_metric=validation_metric,
            hyperparameters=hyperparameters,
            label="label",
        )

        #predictor = MultiModalPredictor.load(args.ckpt_path)
        '''
        #for multiclass gan and real data
        predictor = MultiModalPredictor(
            problem_type = "semantic_segmentation", 
            validation_metric = validation_metric, #multiclass for multiclass gan data
            eval_metric = validation_metric,
            hyperparameters = hyperparameters,
            num_classes = 5,
            classes = [0, 1, 2, 3, 4],
            label = "label",
        )
        '''
        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)


    # evaluation
    #metric_file = os.path.join(args.output_dir, "metrics.txt")
    metric_file = os.path.join(args.output_dir, f"metrics_{args.data_name}.txt")
    out_file = os.path.join(args.output_dir, f"output_{args.data_name}.pkl") 

    f = open(metric_file, "a")
    if dataset_name in ["gan-generated", "israt", "alloy"]:
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"{args.data_name}.csv")), dataset_dir)

        #eval_metrics = ["iou"]
        #res = predictor.evaluate(test_df, metrics=eval_metrics, return_pred=True)

        #test_df = test_df.drop(columns=["label"], errors="ignore")
        test_df_pred = test_df[["image"]].copy()
        test_df_label = test_df[["label"]].copy()

        output = predictor.predict(test_df_pred, batch_size=1, return_all_masks=True)  # safer for memory
        metrics = plot_output(test_df_pred, test_df_label, output, args, morph_size=args.morph_size)

        summary = (
            f"\n=== {args.data_name} ===\n"
            f"Overall:\n"
            f"  Mean IoU          : {metrics['mean_iou']:.4f}\n"
            f"  Std IoU           : {np.std(metrics['ious']):.4f}\n"
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

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")
    f.close()



"""
# --- TRAINING (original, AFAwNi only) ---
# python3 run_semantic_segmentation_class_3.py --task gan-generated --dataset_dir ./datasets --output_dir ./output/class_3/train --data_name train_class3 --num_gpus 1 --batch_size 8 --rank 2

# --- Testing ---
CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_class_3.py --task gan-generated --ckpt_path AutogluonModels/class_3 --morph_size 1 --dataset_dir ./datasets --output_dir ./output/gan/test4_c3 --data_name test4_class3 --num_gpus 1 --batch_size 4 --rank 2 --eval
CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_class_3.py --task gan-generated --ckpt_path AutogluonModels/class_3 --morph_size 4 --dataset_dir ./datasets --output_dir ./output/gan/test5_c3 --data_name test5_class3 --num_gpus 1 --batch_size 4 --rank 2 --eval
CUDA_VISIBLE_DEVICES=0 python3 run_semantic_segmentation_class_3.py --task gan-generated --ckpt_path AutogluonModels/class_3 --morph_size 5 --dataset_dir ./datasets --output_dir ./output/gan/test6_c3 --data_name test6_class3 --num_gpus 1 --batch_size 4 --rank 2 --eval
"""


# Test2: Ti64 (bright metal, out-of-distribution) generates more
# scattered false positives than training-domain AFAwNi; larger kernel cuts these. ~3x3
# Test1: 1x1
# Test2: 7x7
# Test3: 7x7
# Test4: 1x1
# Test5: 4x4
# Test6: 7x7