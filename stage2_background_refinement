import cv2
import numpy as np
import torch
from skimage import measure


def extract_background_mask(contours, image_shape):
    """
    根据第一阶段轮廓生成背景掩膜
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for contour in contours:
        contour_int = np.round(contour).astype(int)
        contour_int[:, 0] = np.clip(contour_int[:, 0], 0, mask.shape[0] - 1)
        contour_int[:, 1] = np.clip(contour_int[:, 1], 0, mask.shape[1] - 1)

        cv2.fillPoly(mask, [contour_int[:, ::-1]], 1)

    background_mask = (mask == 0).astype(np.uint8)
    return background_mask


def run_background_refinement(image, contours_stage1, levelset_func, device, visualize=True):
    """
    Stage 2: 背景区域二次分割

    参数：
    - image: 原始图像（BGR）
    - contours_stage1: 第一阶段轮廓
    - levelset_func: change_lsf函数
    - device: torch device
    """

    # ======================
    # 1. 提取背景区域
    # ======================
    background_mask = extract_background_mask(contours_stage1, image.shape)

    background_image = image.copy()
    background_image[background_mask == 0] = 0

    # ======================
    # 2. 转换到 LAB-B 通道
    # ======================
    img_lab = cv2.cvtColor(background_image, cv2.COLOR_BGR2LAB)
    b_channel = img_lab[:, :, 2].astype(np.float32)

    # 归一化
    b_channel = (b_channel - b_channel.min()) / (b_channel.max() - b_channel.min() + 1e-8)

    # 转 tensor
    input_tensor = torch.tensor(b_channel).float().to(device)

    # ======================
    # 3. 二次 level-set 分割
    # ======================
    initial_lsf = torch.ones_like(input_tensor) * 0.3
    initial_lsf[0:5, 0:5] = 0.7

    phi_sec = levelset_func(
        input_tensor,
        initial_lsf,
        iter_num=15,
        sigma=0.1,
        lambda1=1.0,
        lambda2=1.0,
        alpha1=1.0,
        alpha2=1.0,
    )

    phi_sec_np = phi_sec.detach().cpu().numpy()

    # ======================
    # 4. 提取轮廓
    # ======================
    contours_stage2 = measure.find_contours(phi_sec_np, 0.5)

    print(f"Stage 2 background refinement: {len(contours_stage2)} contours")

    # ======================
    # 5. 可视化
    # ======================
    if visualize:
        import matplotlib.pyplot as plt

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)

        # 第一阶段（蓝色）
        for contour in contours_stage1:
            plt.plot(contour[:, 1], contour[:, 0], color="blue", linewidth=1)

        # 第二阶段（红色）
        for contour in contours_stage2:
            plt.plot(contour[:, 1], contour[:, 0], color="red", linewidth=1)

        plt.title("Stage1 (blue) + Stage2 Background (red)")
        plt.axis("off")
        plt.show()

    return contours_stage2, background_mask
