# Rockimage: Domain-Selective Progressive Segmentation

This repository provides the implementation of the proposed segmentation framework for rock thin-section images, including global initialization and domain-selective refinement.

The method is designed to address challenges such as weak boundaries, low contrast, and complex textures in rock thin-section images.

---

## Method Overview

The framework consists of two main stages:

### Stage 1: Global Initialization

* PCA-LAB representation
* Transformer-based parameter prediction
* GLFIF-based level-set segmentation
* Provides a stable global structural segmentation

### Stage 2: Domain-Selective Refinement

* Background refinement using LAB-B representation
* Foreground refinement using RGB-B representation
* Enhances weak boundaries and fine-grained structures

---

## Repository Structure

```text
rockimage/
├── README.md
├── requirements.txt
│
├── main_stage1_global_segmentation.py
├── quick_test.py
│
├── stage2_background_refinement.py
├── stage2_foreground_refinement.py
│
├── data_example/
│   └── sample.png
│
├── checkpoints/
│   └── README.txt
```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended environment:

* Python 3.8+
* PyTorch 1.10+

---

## Quick Test

An example input image is provided in:

```text
data_example/sample.png
```

Run the quick test with:

```bash
python quick_test.py
```

The script performs Stage 1 global segmentation and prints the number of detected contours.

---

## Model Weights

Due to GitHub file size limitations, the pre-trained model weights are provided via an external download link:

Quark Cloud:
https://pan.quark.cn/s/b401e9ae3aa8

Downloaded file name:

```text
model_epoch_0.pth
```

Please rename the downloaded file to:

```text
stage1_model.pth
```

and place it in:

```text
checkpoints/stage1_model.pth
```

If the checkpoint file is not found, the code will run with randomly initialized weights for demonstration purposes.

---

## Notes

* This repository contains the core implementation of Stage 1 (global initialization) and Stage 2 (domain-selective refinement).
* `quick_test.py` is provided to verify that the code can run successfully.
* The Stage 2 modules are included for further experimentation and extension.

---

## Contact

For questions or issues, please contact the author.

---

## License

This project is provided for academic research and educational use only.
