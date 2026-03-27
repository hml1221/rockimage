from main_stage1_global_segmentation import build_model, process_single_image_stage1

def main():
    image_path = "data_example/sample.png"
    checkpoint_path = "checkpoints/stage1_model.pth"

    model = build_model(checkpoint_path=checkpoint_path)

    phi_np, contours, contour_count = process_single_image_stage1(
        image_path=image_path,
        model=model,
        visualize=False
    )

    print("Quick test completed successfully.")
    print(f"Contours found: {contour_count}")

if __name__ == "__main__":
    main()
