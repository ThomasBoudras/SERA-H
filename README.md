# SERA-H

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Lightning](https://img.shields.io/badge/-Lightning_2.5-792ee5)](https://lightning.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2512.18128-b31b1b.svg)](https://arxiv.org/abs/2512.18128)

> **Note**: This is the official implementation of the paper **["SERA-H: Beyond Native Sentinel Spatial Limits for High-Resolution Canopy Height Mapping"](https://arxiv.org/abs/2512.18128)**.

## Abstract
High-resolution mapping of canopy height is essential for forest management and biodiversity monitoring. Although recent studies have led to the advent of deep learning methods using satellite imagery to predict height maps, these approaches often face a trade-off between data accessibility and spatial resolution. To overcome these limitations, we present SERA-H, an end-to-end model combining a super-resolution module (EDSR) and temporal attention encoding (UTAE). Trained under the supervision of high-density LiDAR data (ALS), our model generates 2.5 m resolution height maps from freely available Sentinel-1 and Sentinel-2 (10 m) time series data. Evaluated on an open-source benchmark dataset in France, SERA-H, with a MAE of 2.6 m and R² of 0.82, not only outperforms standard Sentinel-1/2 baselines but also achieves performance comparable to or better than methods relying on commercial very high-resolution imagery (SPOT-6/7, PlanetScope, Maxar). These results demonstrate that combining high-resolution supervision with the spatiotemporal information embedded in time series enables the reconstruction of details beyond the input sensors' native resolution. SERA-H opens the possibility of freely mapping forests with high revisit frequency, achieving accuracy comparable to that of costly commercial imagery.

## Workflow

![SERA-H Workflow](docs/assets/workflow.png)
*Figure 1: Overview of the SERA-H architecture combining Super-Resolution (EDSR) and Multi-temporal imaging (UTAE).*

## Results

Comparison of SERA-H with state-of-the-art methods:

| Model | Input images | Reference CHM | MAE (m) | RMSE (m) | nMAE (%) | Tree Cover IoU (%) |
|-------|--------------|------------------|---------|----------|----------|--------------------|
| Pauls | S1-S2 (10m) | GEDI (10m) | 5.13 | 7.24 | 42.06 | 45.58 |
| Schwartz | S1-S2 (10m) | GEDI (10m) | 4.47 | 6.17 | 45.96 | 35.12 |
| Tolan | Maxar (1m) | ALS (1m) | 5.49 | 7.52 | 41.74 | 70.15 |
| Liu | Planet (3m) | ALS (3m) | 4.40 | 6.20 | 37.44 | 78.42 |
| Fogel | SPOT-6/7 (1.5m) | ALS (1.5m) | 2.37 | 3.65 | 18.88 | 88.15 |
| **SERA-H** | **S1-S2 (10m)** | **ALS (2.5m)** | **2.60** | **3.86** | **20.40** | **87.25** |

*Table 1: Performance comparison on the test area at 2.5m resolution.*

> **Note**: State-of-the-art canopy height maps used for comparison were obtained either through download links provided in their respective original papers or via the Google Earth Engine API.

## Installation

### Prerequisites
- Linux or macOS
- Python 3.11
- NVIDIA GPU + CUDA

### Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/ThomasBoudras/SERA-H.git
cd SERA-H

# Create a virtual environment (optional but recommended)
conda create -n sera-h-env -c conda-forge gdal=3.8 rasterio fiona python=3.11
conda activate sera-h-env

# Install requirements
pip install -r requirements.txt
```

## Data Preparation

### 1. Download Sentinel Data
To download the required time series data (Sentinel-1, Sentinel-2, etc.), run the following command:

```bash
python -m src.preprocessing.download.download_s1_s2_via_gdf -cn=dwd_gdf_height_timeseries
```

### 2. Clean Sentinel Data

Once you have downloaded the Sentinel-1 and Sentinel-2 data, process the raw files into a clean dataset for training. This step sets up Virtual Raster Tables (VRTs) for the downloaded images, records the valid triplets (Sentinel-2, Sentinel-1 ascending, Sentinel-1 descending) directly into a GeoJSON file, and filters out regions with missing Sentinel data to ensure that only valid geometries are kept for the next steps.

```bash
python -m src.preprocessing.datasets.get_clean_dataset -cn=gcd_height_map_timeseries
```


### 3. Prepare Lidar (ALS) and Forest Mask Data

To use the lidar (ALS) images and the forest mask (`forest_mask.parquet`), please manually download these files from:

[https://huggingface.co/datasets/AI4Forest/Open-Canopy/tree/main/canopy_height](https://huggingface.co/datasets/AI4Forest/Open-Canopy/tree/main/canopy_height)

Place the downloaded `forest_mask.parquet` and canopy height (lidar) files in the following directory:

```
data/open_canopy/
```

### 4. Dataset Preparation (Presave)

To significantly speed up the training process, you need to pre-extract and save the dataset patches to disk. This preparation step iterates over the dataset and saves the processed tensors, avoiding on-the-fly raster reading during training.

```bash
python -m src.presave_dataset train=h-sera_h
```

### 5. Download Pre-trained Super-Resolution Model (EDSR)

SERA-H uses EDSR as its super-resolution backbone. You need to download the pre-trained weights from the official repository:
[https://github.com/sanghyun-son/EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch?tab=readme-ov-file)

Download the model weights (e.g., `edsr_baseline_x4.pt`) and place them in:
```
data/utils/
```

### 6. (Optional) Prepare State-of-the-Art Predictions for Comparison

To reproduce the comparison table above or generate comparison figures, you need the prediction maps of the other methods (Tolan, Liu, Fogel, Schwartz, Pauls, Lang). These can be obtained either through the download links provided in their respective original papers or via the Google Earth Engine API. Place them so that their path matches the `path` field defined in the corresponding `configs/postprocessing/metrics/h-individual_<method>.yaml` file (predictions are generally expected under `data/sota/<method>/`).

After running these steps, your data should be organized as follows:

```
data/
├── utils/         # Helper data (geojsons, pre-trained EDSR weights)
├── sentinel/      # Downloaded Sentinel-1 and Sentinel-2 images
├── open_canopy/   # Lidar and forest mask data you downloaded from Hugging Face
└── sota/          # (Optional) State-of-the-art predictions used for comparison
```

> **Note**: `configs/preprocessing/download`, `configs/preprocessing/datasets`, and `configs/predict` also include `*_test_2019` config variants, which run the same download/clean/predict workflow described above on a separate 2019 test area.

## Usage

This project uses [Hydra](https://hydra.cc/) for configuration management. All configuration files are in `configs/`.

### Training
To train the model with default configuration:

```bash
python -m src.train train=h-sera_h
```

### Ablation Study
To run ablation studies, use the configuration files ending with `_as` in `configs/train/`. These configs are pre-set for specific experiments (e.g., varying number of input images).

```bash
# Example: Train with 4 input images
python -m src.train train=h-sera_h_as_04img

# Example: Train using U-Net architecture for ablation
python -m src.train train=h-unet_as
```

You can still override individual parameters from the command line:
```bash
python -m src.train train=h-sera_h_as_04img datamodule.batch_size=32
```

### Inference / Metrics

To run inference on the entire test area, use:

```bash
python -m src.presave_dataset predict=h-sera_h
python -m src.predict predict=h-sera_h
```

After inference, compute the evaluation metrics based on the predictions. Each model/method has its own metrics config in `configs/postprocessing/metrics/`, named `h-individual_<method>.yaml`:

```bash
# Compute local + global metrics for SERA-H
python -m src.postprocessing.metrics.computes_metrics -cn=h-individual_sera

# Compute metrics for a state-of-the-art baseline (requires the corresponding data, see Data Preparation)
python -m src.postprocessing.metrics.computes_metrics -cn=h-individual_tolan2024
```

Once metrics have been computed and saved (as `.xlsx` files) for the models you want to compare, use the `h-figure_*.yaml` configs to generate comparison figures (scatter plots, boxplots, qualitative examples, etc.):

```bash
python -m src.postprocessing.metrics.computes_metrics -cn=h-figure_scatter_plot
```

## Project Structure

All outputs (logs, checkpoints, predictions, metrics) will be automatically saved in `data/logs/`.

```
SERA-H/
├── configs/                 # Hydra configuration files
│   ├── datamodule/
│   ├── module/
│   ├── callbacks/
│   ├── logger/
│   ├── trainer/
│   ├── hydra/
│   ├── train/               # Training configs (default + `_as` ablation studies)
│   ├── predict/             # Inference configs
│   ├── preprocessing/
│   │   ├── download/        # Configs for downloading Sentinel-1/2 data
│   │   └── datasets/        # Configs for cleaning/preparing the dataset
│   ├── postprocessing/
│   │   └── metrics/         # Configs for evaluation and figures (SOTA models, scatter plots, etc.)
│   └── config.yaml          # Main config file
├── data/                    # Dataset location
│   ├── logs/                # Training logs, checkpoints, predictions and metrics
│   ├── sentinel/            # Downloaded Sentinel-1 & Sentinel-2 images
│   ├── open_canopy/         # Reference Lidar and Forest Mask
│   ├── sota/                # State-of-the-art predictions from literature (Liu, Fogel, Pauls, Schwartz, etc.)
│   └── utils/               # Helper data (geojsons, etc.)
├── src/                     # Source code
│   ├── datamodule/          # LightningDataModules
│   ├── module/              # LightningModules (Model architecture)
│   ├── callbacks/           # Lightning callbacks (e.g. image logging)
│   ├── preprocessing/       # Scripts for downloading and cleaning the dataset
│   ├── postprocessing/      # Scripts for computing metrics and generating figures
│   ├── train.py             # Main training script
│   ├── predict.py           # Inference script
│   └── presave_dataset.py   # Pre-extracts and saves dataset patches to disk
├── requirements.txt         # Python dependencies
└── setup.cfg                # Project metadata
```

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@misc{boudras2026serahnativesentinelspatial,
      title={SERA-H: Beyond Native Sentinel Spatial Limits for High-Resolution Canopy Height Mapping}, 
      author={Thomas Boudras and Martin Schwartz and Rasmus Fensholt and Martin Brandt and Ibrahim Fayad and Jean-Pierre Wigneron and Gabriel Belouze and Fajwel Fogel and Philippe Ciais},
      year={2026},
      eprint={2512.18128},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.18128}, 
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions, feel free to contact:
Thomas Boudras - thomas.boudras@lsce.ipsl.fr

