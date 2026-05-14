from pathlib import Path
import argparse
import cv2
import pandas as pd
from ultralytics import YOLO


def find_best_model(run_name="yolo_human_car"):
    """
    Automatically finds best.pt inside the runs/ folder.
    Useful when YOLO creates nested folders in Colab.
    """
    candidates = [
        p for p in Path("runs").rglob("best.pt")
        if run_name in str(p)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find best.pt for run name '{run_name}'. "
            "Please provide --model path manually."
        )

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def get_image_paths(source_path: Path):
    """
    Get image paths from a single image file or a folder.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    if source_path.is_file():
        if source_path.suffix.lower() in image_extensions:
            return [source_path]
        raise ValueError(f"Unsupported image file: {source_path}")

    if source_path.is_dir():
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(source_path.glob(f"*{ext}"))

        return sorted(image_paths)

    raise FileNotFoundError(f"Source path not found: {source_path}")


def draw_detection_box(image, box, class_name, confidence):
    """
    Draw bounding box and class label on image.
    """
    x1, y1, x2, y2 = map(int, box)

    # Human = green, car = blue
    if class_name == "human":
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    label = f"{class_name} {confidence:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    return image


def draw_count_panel(image, human_count, car_count):
    """
    Draw total human count and car count on top-left of image.
    """
    panel_text_1 = f"Total Humans: {human_count}"
    panel_text_2 = f"Cars Detected: {car_count}"

    # Background panel
    cv2.rectangle(image, (10, 10), (280, 90), (0, 0, 0), -1)

    cv2.putText(
        image,
        panel_text_1,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        image,
        panel_text_2,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return image


def process_image(model, image_path, output_dir, imgsz, conf):
    """
    Run detection on one image, draw boxes, count humans/cars, and save output.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        verbose=False
    )

    result = results[0]

    human_count = 0
    car_count = 0

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy()

        class_name = model.names[class_id]

        if class_name == "human":
            human_count += 1
            image = draw_detection_box(image, xyxy, class_name, confidence)

        elif class_name == "car":
            car_count += 1
            image = draw_detection_box(image, xyxy, class_name, confidence)

    image = draw_count_panel(image, human_count, car_count)

    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)

    return {
        "image_name": image_path.name,
        "human_count": human_count,
        "car_count": car_count,
        "output_path": str(output_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect humans and cars, draw bounding boxes, and count total humans."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained YOLO model best.pt"
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="yolo_human_car",
        help="YOLO run name for automatic best.pt search"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="outputs/task02/sample_inputs",
        help="Input image file or folder"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/task03/processed_images",
        help="Folder to save processed images"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    args = parser.parse_args()

    if args.model is not None and Path(args.model).exists():
        model_path = Path(args.model)
    else:
        model_path = find_best_model(args.run_name)

    source_path = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using model:", model_path)
    print("Source:", source_path)
    print("Output folder:", output_dir)

    model = YOLO(str(model_path))

    image_paths = get_image_paths(source_path)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {source_path}")

    summary_rows = []

    for image_path in image_paths:
        row = process_image(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            imgsz=args.imgsz,
            conf=args.conf
        )

        summary_rows.append(row)

        print(
            f"{row['image_name']} -> "
            f"Humans: {row['human_count']}, Cars: {row['car_count']}"
        )

    summary_df = pd.DataFrame(summary_rows)

    csv_path = output_dir.parent / "detection_counts.csv"
    summary_df.to_csv(csv_path, index=False)

    print("\nDetection and counting complete.")
    print("Processed images saved at:", output_dir)
    print("Counting summary saved at:", csv_path)


if __name__ == "__main__":
    main()