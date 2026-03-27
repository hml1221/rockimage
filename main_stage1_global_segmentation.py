import os
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from skimage import measure
from torch.nn import functional as F
from torchvision import models


# =========================
# Basic configuration
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

INPUT_CHANNELS = 1
INPUT_HEIGHT = 256
INPUT_WIDTH = 256

NUM_LAYERS = 4
NUM_HEADS = 8
HIDDEN_DIM = 2048
DROPOUT_RATE = 0.1


# =========================
# Stage 1 parameter model
# =========================
class TransformerModel(nn.Module):
    """
    Parameter predictor for Stage 1 global level-set initialization.
    Input: PCA-LAB single-channel image
    Output: 4 positive parameters for GLFIF evolution
    """

    def __init__(
        self,
        input_channels=1,
        input_height=256,
        input_width=256,
        num_layers=4,
        num_heads=8,
        hidden_dim=2048,
        dropout_rate=0.1,
    ):
        super().__init__()

        resnet50 = models.resnet50(weights=None)
        resnet50.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        resnet50.fc = nn.Identity()
        self.feature_extractor = resnet50

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            features = self.feature_extractor(dummy_input)
            feature_dim = features.shape[1]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.parameter_generator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softplus(),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.unsqueeze(1)
        transformer_output = self.transformer_encoder(features)
        pooled_output = transformer_output.mean(dim=1)
        parameters = self.parameter_generator(pooled_output)
        return parameters


# =========================
# Utility functions
# =========================
def delete_small_contours(contours, min_area=200):
    filtered = []
    for contour in contours:
        poly = np.round(contour).astype(int)
        if len(poly) < 3:
            continue
        area = cv2.contourArea(poly[:, ::-1])
        if area >= min_area:
            filtered.append(contour)
    return filtered


def gaussian_blur_torch(img, kernel_size=3, sigma=1.0):
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3:
        img = img.unsqueeze(0)

    x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
    y = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
    x, y = torch.meshgrid(x, y, indexing="ij")

    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    padding = kernel_size // 2
    blurred = F.conv2d(img, kernel, padding=padding)
    return blurred.squeeze()


def torch_gradient(tensor):
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D.")

    dy = torch.zeros_like(tensor)
    dx = torch.zeros_like(tensor)

    dy[1:-1, :] = (tensor[2:, :] - tensor[:-2, :]) / 2.0
    dy[0, :] = tensor[1, :] - tensor[0, :]
    dy[-1, :] = tensor[-1, :] - tensor[-2, :]

    dx[:, 1:-1] = (tensor[:, 2:] - tensor[:, :-2]) / 2.0
    dx[:, 0] = tensor[:, 1] - tensor[:, 0]
    dx[:, -1] = tensor[:, -1] - tensor[:, -2]

    return dy, dx


def convert_to_pca_lab(image_bgr):
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    h, w, c = img_lab.shape
    data = img_lab.reshape(-1, c).astype(np.float32)

    pca = PCA(n_components=1)
    reduced = pca.fit_transform(data)
    reduced_img = reduced.reshape(h, w)

    min_val = reduced_img.min()
    max_val = reduced_img.max()
    normalized_img = (reduced_img - min_val) / (max_val - min_val + 1e-8)
    return normalized_img.astype(np.float32)


# =========================
# GLFIF / level-set
# =========================
def glfif_update(img, local_img, u0, sigma, lambda1, lambda2, alpha1, alpha2, g):
    u1 = u0 ** 2
    u2 = (1 - u0) ** 2

    iu1 = img * u1
    iu2 = img * u2
    c1 = torch.sum(iu1) / (torch.sum(u1) + 1e-8)
    c2 = torch.sum(iu2) / (torch.sum(u2) + 1e-8)

    ku1 = gaussian_blur_torch(u1, 3, sigma)
    ku2 = gaussian_blur_torch(u2, 3, sigma)
    ki1 = gaussian_blur_torch(iu1, 3, sigma)
    ki2 = gaussian_blur_torch(iu2, 3, sigma)

    s1 = ki1 / (ku1 + 1e-8)
    s2 = ki2 / (ku2 + 1e-8)

    denominator = lambda2 * (img - c2) ** 2 + (alpha1 * s1 + alpha2 * c1) + 1e-8
    numerator = lambda1 * (img - c1) ** 2 + (alpha1 * s2 + alpha2 * c2)

    un = 1 / (1 + numerator / denominator)
    un = gaussian_blur_torch(un, 3, sigma)
    return un


def evolve_level_set(img, initial_lsf, iter_num, sigma, lambda1, lambda2, alpha1, alpha2):
    if img.dim() != 2:
        raise ValueError("Input image must be a 2D grayscale tensor.")
    if img.shape != initial_lsf.shape:
        raise ValueError("Input image and initial_lsf must have the same shape.")

    img_smooth = gaussian_blur_torch(img, 3, sigma)
    dy, dx = torch_gradient(img_smooth)
    edge_strength = dy ** 2 + dx ** 2
    g = 1 / (1 + edge_strength)

    phi = initial_lsf.clone()
    for _ in range(iter_num):
        phi = glfif_update(
            img, img, phi, sigma, lambda1, lambda2, alpha1, alpha2, g
        )
    return phi


def run_stage1_global_segmentation(img_tensor, parameters, iter_num=20, sigma=0.1):
    parameters = parameters.squeeze()

    initial_lsf = torch.ones_like(img_tensor) * 0.3
    initial_lsf[0:5, 0:5] = 0.7

    phi = evolve_level_set(
        img=img_tensor,
        initial_lsf=initial_lsf,
        iter_num=iter_num,
        sigma=sigma,
        lambda1=parameters[0].float(),
        lambda2=parameters[1].float(),
        alpha1=parameters[2].float(),
        alpha2=parameters[3].float(),
    )
    return phi


# =========================
# Model loading
# =========================
def build_model(checkpoint_path=None):
    model = TransformerModel(
        input_channels=INPUT_CHANNELS,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        print("Model weights loaded successfully.")
    else:
        print("Checkpoint not found. Using randomly initialized weights.")

    model.eval()
    return model


# =========================
# Stage 1 processing
# =========================
def process_single_image_stage1(image_path, model, visualize=True):
    """
    Process one image with Stage 1 only.

    Returns:
        phi_np: level-set result
        contours: extracted contours
        contour_count: number of contours
        parameters_np: predicted Stage 1 parameters
    """
    print(f"\nProcessing image: {os.path.basename(image_path)}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pca_lab_img = convert_to_pca_lab(image)
    input_tensor = torch.tensor(pca_lab_img).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        parameters = model(input_tensor)
        phi = run_stage1_global_segmentation(
            img_tensor=torch.tensor(pca_lab_img).float().to(device),
            parameters=parameters,
            iter_num=20,
            sigma=0.1,
        )

    parameters_np = parameters.detach().cpu().numpy().squeeze()
    phi_np = phi.detach().cpu().numpy().squeeze()

    contours = measure.find_contours(phi_np, 0.5)
    contours = delete_small_contours(
        contours,
        min_area=image.shape[0] * image.shape[1] * 0.003
    )
    contour_count = len(contours)

    print(f"Stage 1 global segmentation found {contour_count} contours.")
    print(f"Predicted parameters: {parameters_np}")

    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(pca_lab_img, cmap="gray")
        axes[1].set_title("PCA-LAB Representation")
        axes[1].axis("off")

        axes[2].imshow(image_rgb)
        for contour in contours:
            axes[2].plot(contour[:, 1], contour[:, 0], linewidth=1.5)
        axes[2].set_title("Stage 1 Global Segmentation")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return phi_np, contours, contour_count, parameters_np


def save_stage1_result(image_path, model, output_dir="./results_stage1"):
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _, contours, contour_count, _ = process_single_image_stage1(
        image_path, model, visualize=False
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color="red")
    plt.title(f"Stage 1 Global Segmentation - {contour_count} contours")
    plt.axis("off")

    output_path = os.path.join(
        output_dir,
        f"stage1_{os.path.basename(image_path)}"
    )
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved result to: {output_path}")
    return output_path


def main():
    image_path = "data_example/sample.png"
    checkpoint_path = "checkpoints/stage1_model.pth"

    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    model = build_model(checkpoint_path=checkpoint_path)

    try:
        phi_np, contours, contour_count, parameters_np = process_single_image_stage1(
            image_path, model, visualize=True
        )
        save_stage1_result(image_path, model, output_dir="./results_stage1")
        print("Stage 1 processing completed successfully.")
    except Exception as e:
        print(f"Error during Stage 1 processing: {e}")
        raise


if __name__ == "__main__":
    main()
