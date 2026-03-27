import os
import cv2
import matplotlib.pyplot as plt
import torch

from main_stage1_global_segmentation import (
    build_model,
    process_single_image_stage1,
    evolve_level_set,
)
from stage2_background_refinement import run_background_refinement
from stage2_foreground_refinement import run_foreground_refinement


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def change_lsf_wrapper(img, initial_lsf, iter_num, sigma, lambda1, lambda2, alpha1, alpha2):
    """
    Wrapper so that Stage 2 modules can reuse the Stage 1 level-set evolution function.
    """
    return evolve_level_set(
        img=img,
        initial_lsf=initial_lsf,
        iter_num=iter_num,
        sigma=sigma,
        lambda1=lambda1,
        lambda2=lambda2,
        alpha1=alpha1,
        alpha2=alpha2,
    )


def save_stage1_result(image_bgr, contours_stage1, save_path):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)

    for contour in contours_stage1:
        plt.plot(contour[:, 1], contour[:, 0], color="red", linewidth=1.2)

    plt.title("Stage 1 Segmentation")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_stage1_plus_background(image_bgr, contours_stage1, contours_bg, save_path):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)

    # Stage 1: blue
    for contour in contours_stage1:
        plt.plot(contour[:, 1], contour[:, 0], color="blue", linewidth=1.2)

    # Background refinement: red
    for contour in contours_bg:
        plt.plot(contour[:, 1], contour[:, 0], color="red", linewidth=1.2)

    plt.title("Stage 1 + Background Refinement")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_stage1_plus_foreground(image_bgr, contours_stage1, contours_fg, save_path):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)

    # Stage 1: blue
    for contour in contours_stage1:
        plt.plot(contour[:, 1], contour[:, 0], color="blue", linewidth=1.2)

    # Foreground refinement: yellow
    for contour in contours_fg:
        plt.plot(contour[:, 1], contour[:, 0], color="yellow", linewidth=1.2)

    plt.title("Stage 1 + Foreground Refinement")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    image_path = "data_example/sample.png"
    checkpoint_path = "checkpoints/stage1_model.pth"
    os.makedirs("results", exist_ok=True)

    # 1. build model
    model = build_model(checkpoint_path=checkpoint_path)

    # 2. run Stage 1
    phi_stage1, contours_stage1, contour_count = process_single_image_stage1(
        image_path=image_path,
        model=model,
        visualize=False
    )

    print("Stage 1 completed.")
    print(f"Contours found: {contour_count}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image_bgr = cv2.resize(image_bgr, (256, 256))

    # save only Stage 1 result
    save_stage1_result(
        image_bgr=image_bgr,
        contours_stage1=contours_stage1,
        save_path="results/stage1_result.png"
    )

    # 3. run Stage 2 background refinement
    contours_bg, background_mask = run_background_refinement(
        image=image_bgr,
        contours_stage1=contours_stage1,
        levelset_func=change_lsf_wrapper,
        device=device,
        visualize=False
    )

    save_stage1_plus_background(
        image_bgr=image_bgr,
        contours_stage1=contours_stage1,
        contours_bg=contours_bg,
        save_path="results/stage1_plus_background.png"
    )

    # 4. run Stage 2 foreground refinement
    # Here we use fixed parameters for visualization/demo.
    # If you later want stricter consistency, you can modify Stage 1 code to return predicted parameters.
    parameters = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)

    contours_fg, foreground_mask, phi_stage2_fg = run_foreground_refinement(
        image_bgr=image_bgr,
        contours_stage1=contours_stage1,
        phi_stage1=phi_stage1,
        change_lsf_func=change_lsf_wrapper,
        device=device,
        parameters=parameters,
        sigma=0.1,
        iter_num=20,
        visualize=False
    )

    save_stage1_plus_foreground(
        image_bgr=image_bgr,
        contours_stage1=contours_stage1,
        contours_fg=contours_fg,
        save_path="results/stage1_plus_foreground.png"
    )

    print("All visual results have been saved to the results/ folder.")
    print("Saved files:")
    print("- results/stage1_result.png")
    print("- results/stage1_plus_background.png")
    print("- results/stage1_plus_foreground.png")


if __name__ == "__main__":
    main()
