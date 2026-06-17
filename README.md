# XCT-SAM: Sequential Parameter-Eﬃcient Domain Adaptation of SAM for Industrial XCT Defect Segmentation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Sequencing](https://img.shields.io/badge/Application-Additive--Manufacturing-blueviolet)
![Backbone](https://img.shields.io/badge/Backbone-SAM-orange)
![Model](https://img.shields.io/badge/Model-XCT--SAM-skyblue)
![Conference](https://img.shields.io/badge/Workshop-IAPR%20TC8-blue)
[![Citation](https://img.shields.io/badge/Cite%20Us-Springer--ICPR--2026-red)](https://doi.org/)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

XCT-SAM is a parameter-efficient framework for defect segmentation in
**Additive Manufacturing (AM) X-ray Computed Tomography (XCT)** images.
It addresses two major challenges in industrial XCT analysis: the large
domain gap between natural images and XCT scans, and the limited
availability of labeled defect data.

Instead of directly adapting the **Segment Anything Model (SAM)** to XCT
images, XCT-SAM introduces a **sequential domain adaptation** strategy.
The model is first adapted to an intermediate alloy microstructure
domain using lightweight **Conv-LoRA** adapters and then transferred to
the target XCT domain. This progressive adaptation effectively bridges
the distribution gap while preserving SAM's powerful segmentation
capabilities.

Using **Conv-LoRA (rank = 2)**, XCT-SAM injects convolutional spatial
inductive bias into SAM's ViT-H backbone while training only **\~4.15M
parameters**, keeping more than **99% of the original model frozen**.
This results in an efficient and scalable adaptation framework suitable
for industrial applications.

Extensive experiments on both **CycleGAN-generated out-of-distribution
XCT datasets** and **real-world NIST XCT scans** demonstrate that
XCT-SAM consistently outperforms zero-shot SAM and existing
domain-adapted SAM baselines, achieving state-of-the-art IoU and Dice
scores.

## Highlights

-   🚀 Sequential domain adaptation from **Natural Images → Alloy
    Microstructures → XCT Images**
-   ⚡ Parameter-efficient fine-tuning with **Conv-LoRA**
-   🧊 \>99% of SAM parameters remain frozen
-   🎯 Robust defect segmentation under severe class imbalance and
    domain shift
-   📊 State-of-the-art performance on synthetic and real-world
    industrial XCT datasets

XCT-SAM demonstrates that intermediate-domain adaptation combined with
parameter-efficient learning provides an effective solution for
industrial defect segmentation in additive manufacturing.

## Installation
The installation may take a while since AutoGluon Multimodal has multiple dependencies.
```shell
  conda create -n conv-lora python=3.10
  conda activate conv-lora
  pip install -U pip
  pip install -U setuptools wheel
  git clone https://github.com/autogluon/autogluon
  cd autogluon && pip install -e multimodal/[tests]
  ```

## Directory & Datasets

Enter the `autogluon/examples/automm/Conv-LoRA` directory and clone this repository.
## Dataset

The dataset used in this work can be downloaded from Google Drive:

- **NIST Real XCT Dataset with our labels:** [Download Dataset (Google Drive)](https://drive.google.com/file/d/19gmWhFFRIjvOb99ZOZFCumEcaTZZlzXB/view?usp=sharing)


## Requirements
- Python 3.8+
- torch >= 2.0, torchvision, autogluon.multimodal, transformers, Pillow, numpy, pandas, scipy, opencv-python, matplotlib


## Repository Structure

The repository is organized as follows:

```text
.
├── AutogluonModels/
├── checkpoints/
├── datasets/
├── output/
├── scripts/
├── README.md
├── run_domain_adaptation_alloy_class2.py
|── run_domain_adaptation_alloy_class3.py
├── run_semantic_segmentation_alloy_class_2.py
├── run_semantic_segmentation_alloy_class_3.py
├── run_semantic_segmentation_nist.py
├── run_df.sh
├── run_xct_focal.sh
```

### Folder Description

| Folder | Description |
|---------|-------------|
| **AutogluonModels/** | Contains saved model checkpoints during training, including Conv-LoRA adapted SAM weights and intermediate training snapshots. |
| **checkpoints/** | Contrais pretrained weights such as sam_vit_h_4b8939.pth and sam2.1_hiera_large.pt|
| **datasets/** | Stores all datasets used for training and evaluation, including alloy microstructure images, synthetic CycleGAN-XCT data, and real NIST XCT datasets. |
| **output/** | Contains experiment outputs such as predicted segmentation masks, evaluation results with performance metrics, and visualizations. |
| **scripts/** | Utility scripts for dataset preparation, preprocessing, evaluation, visualization, and other helper functions. |

---

## Main Files

#### `run_domain_adaptation_alloy_class2.py`

Trains the SAM model with Conv-LoRA adapters on the **Alloy Microstructure Class-2** dataset. This stage serves as the intermediate domain adaptation before transferring to XCT images.

---

#### `run_domain_adaptation_alloy_class3.py`

Trains the SAM model with Conv-LoRA adapters on the **Alloy Microstructure Class-3** dataset, providing an alternative intermediate adaptation setting.

---

#### `run_semantic_segmentation_alloy_class_2.py`

Finetunes and evaluates the adapted model on the **CycleGAN Synthetic XCT defect segmentation dataset** for pore segmentation (class_2), representing the final stage of sequential domain adaptation.

---

#### `run_semantic_segmentation_alloy_class_3.py`

Finetunes and evaluates the adapted model on the **CycleGAN Synthetic XCT defect segmentation dataset** for inclusion segmentation (class_3), representing the final stage of sequential domain adaptation.

---

#### `run_semantic_segmentation_nist.py`

Evaluates the adapted model on the **real NIST XCT defect segmentation dataset**, representing the final stage of sequential domain adaptation.

---

# Training

Run the following command to trained the model:

```bash
python3 run_semantic_segmentation_alloy_class_2.py \
    --dataset_dir datasets/gan-generated \
    --ckpt_path AutogluonModels/alloy_to_xct_focal/class_2 \
    --output_dir output/<value> \
    --data_name <value> \
    --alloy_ckpt_path <value> \
    --num_gpus <value> \
    --batch_size <value> \
    --rank 2 <value> 
```

### Command-line Arguments

- `dataset_dir` specifies the directory containing the training & evaluation dataset.
- `ckpt_path` specifies the path to the trained model checkpoint used for evaluation.
- `alloy_ckpt_path` specifies the path to the trained model checkpoint of stage-1 used for training on XCT images.
- `output_dir` specifies the directory where evaluation logs, predicted masks, and other outputs are saved.
- `data_name` specifies the dataset split or experiment name (e.g., `test4_class2`).
- `num_gpus` specifies the number of GPUs used for evaluation. Default is `1`.
- `batch_size` specifies the effective evaluation batch size. Default is `4`.
- `rank` specifies the rank of the Conv-LoRA adapters. Default is `2`.

# Evaluation

After running the benchmark, the evaluation results of test set are stored in "{output_dir}/metrics.txt".
You can also run the following command to evaluate a checkpoint:

`python3 run_semantic_segmentation_class_2.py --task {dataset_name} --output_dir {output_dir} --ckpt_path {ckpt_path} --data_name {data_file} --num_gpus {total_gpu} --batch_size {batch_size} --rank {rank} --eval`


# Our Related Works

### DAM-SAM
coming...


## Contact
For any questions, issues, or suggestions related to this repository, please feel free to contact @Mahedi-61 or open an issue on GitHub.


### Citation
```
@inproceedings{hasan2026xct-sam,
  title={XCT-SAM: Sequential Parameter-Efficient Domain Adaptation of SAM for Industrial XCT Defect Segmentation},
  author={Hasan, Md Mahedi and Rahaman, Md Mushfiqur and Pachkovskiy, Alan and Ahmed, Imtiaz and Dawson, Jeremy and Das, Srinjoy},
  booktitle={Proceedings of the IAPR TC8 Workshop at the International Conference on Pattern Recognition (ICPR)},
  year={2026}
}
```

```
@article{zhong2024convolution,
  title={Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model},
  author={Zhong, Zihan and Tang, Zhiqiang and He, Tong and Fang, Haoyang and Yuan, Chun},
  journal={arXiv preprint arXiv:2401.17868},
  year={2024}
}
```