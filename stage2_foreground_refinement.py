import cv2
import numpy as np
import torch
from skimage import measure


def contours_to_mask(contours, image_shape):
    """
    Convert Stage 1 contours to a binary foreground mask.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for contour in contours:
        contour_int = np.round(contour).astype(int)
        contour_int[:, 0] = np.clip(contour_int[:, 0], 0, h - 1)
        contour_int[:, 1] = np.clip(contour_int[:, 1], 0, w - 1)
        cv2.fillPoly(mask, [contour_int[:, ::-1]], 1)

    return mask


def delete_small_contours(contours, min_area=50):
    """
    Remove contours with area smaller than min_area.
    """
    filtered = []
    for contour in contours:
        poly = np.round(contour).astype(int)
        if len(poly) < 3:
            continue
        area = cv2.contourArea(poly[:, ::-1])
        if area >= min_area:
            filtered.append(contour)
    return filtered


def extract_rgb_b_channel(image_bgr, foreground_mask):
    """
    Extract RGB-B representation inside the foreground domain.

    Note:
    OpenCV loads images in BGR format. To match the paper description (RGB-B),
    we first convert BGR -> RGB and then select the B channel from RGB.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    b_channel = image_rgb[:, :, 2].astype(np.float32)

    # keep only foreground region
    b_channel[foreground_mask == 0] = 0.0

    # normalize to [0, 1]
    min_val = b_channel.min()
    max_val = b_channel.max()
    b_channel = (b_channel - min_val) / (max_val - min_val + 1e-8)

    return b_channel.astype(np.float32)


def run_foreground_refinement(
    image_bgr,
    contours_stage1,
    phi_stage1,
    change_lsf_func,
    device,
    parameters,
    sigma=0.1,
    iter_num=20,
    visualize=True,
):
    """
    Stage 2 foreground-domain refinement using RGB-B representation.

    Args:
        image_bgr: original input image in BGR format
        contours_stage1: contours obtained from Stage 1 global segmentation
        phi_stage1: level-set function from Stage 1
        change_lsf_func: shared level-set evolution function
        device: torch device
        parameters: [lambda1, lambda2, alpha1, alpha2] predicted in Stage 1
        sigma: smoothing parameter
        iter_num: number of refinement iterations
        visualize: whether to show visualization

    Returns:
        contours_stage2: refined contours in foreground domain
        foreground_mask: binary foreground mask
        phi_stage2_np: level-set output of Stage 2 foreground refinement
    """
    h, w = image_bgr.shape[:2]

    # --------------------------------
    # 1. build foreground mask from Stage 1 contours
    # --------------------------------
    foreground_mask = contours_to_mask(contours_stage1, image_bgr.shape)
    foreground_area = int(np.sum(foreground_mask))

    if foreground_area == 0:
        print("Stage 2 foreground refinement skipped: empty foreground mask.")
        return [], foreground_mask, None

    # --------------------------------
    # 2. use RGB-B representation (paper-consistent)
    # --------------------------------
    fg_rgb_b = extract_rgb_b_channel(image_bgr, foreground_mask)
    fg_tensor = torch.tensor(fg_rgb_b, dtype=torch.float32, device=device)

    # --------------------------------
    # 3. initialize Stage 2 level set from Stage 1 result
    # --------------------------------
    initial_lsf_sec = torch.tensor(phi_stage1, dtype=torch.float32, device=device).clone()
    mask_torch = torch.tensor(foreground_mask, dtype=torch.uint8, device=device)

    # restrict refinement to foreground domain
    initial_lsf_sec[mask_torch == 0] = 0.3
    initial_lsf_sec[mask_torch == 1] = 0.7

    # --------------------------------
    # 4. run foreground refinement
    # --------------------------------
    lambda1 = parameters[0].float()
    lambda2 = parameters[1].float()
    alpha1 = parameters[2].float()
    alpha2 = parameters[3].float()

    phi_stage2 = change_lsf_func(
        fg_tensor,
        initial_lsf_sec,
        iter_num=iter_num,
        sigma=sigma,
        lambda1=lambda1,
        lambda2=lambda2,
        alpha1=alpha1,
        alpha2=alpha2,
    )

    phi_stage2_np = phi_stage2.detach().cpu().numpy()

    # --------------------------------
    # 5. extract refined contours
    # --------------------------------
    contours_stage2 = measure.find_contours(phi_stage2_np, level=0.5)

    min_area = h * w * 0.00001
    contours_stage2 = delete_small_contours(contours_stage2, min_area=min_area)

    print(f"Stage 2 foreground refinement: {len(contours_stage2)} contours")

    # --------------------------------
    # 6. visualization
    # --------------------------------
    if visualize:
        import matplotlib.pyplot as plt

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(6, 6))
        plt.imshow(image_rgb)

        # Stage 1 contours: blue
        for contour in contours_stage1:
            plt.plot(contour[:, 1], contour[:, 0], color="blue", linewidth=1.2)

        # Stage 2 foreground refinement: yellow
        for contour in contours_stage2:
            plt.plot(contour[:, 1], contour[:, 0], color="yellow", linewidth=1.2)

        plt.title("Stage 1 (blue) + Stage 2 Foreground RGB-B (yellow)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return contours_stage2, foreground_mask, phi_stage2_np
