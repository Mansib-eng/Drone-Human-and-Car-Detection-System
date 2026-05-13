from pathlib import Path
import random
import shutil
import argparse
from ultralytics import YOLO


def find_best_model(run_name="yolo_human_car"):
    """
    Automatically find best.pt inside runs/ folder.
    This handles nested folders like:
    runs/detect/runs/detect/yolo_human_car/weights/best.pt
    """
    candidates = [
        p for p in Path("runs").rglob("best.pt")
        if run_name in str(p)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find best.pt for run name '{run_name}' inside runs/ folder."
        )

    # Use the most recently modified best.pt
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def collect_sample_images(source_dir, sample_input_dir, num_samples):
    image_paths = sorted(
        list(source_dir.glob("*.jpg")) +
        list(source_dir.glob("*.jpeg")) +
        list(source_dir.glob("*.png"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    sample_input_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    selected_images = random.sample(image_paths, min(num_samples, len(image_paths)))

    for img_path in selected_images:
        shutil.copy2(img_path, sample_input_dir / img_path.name)

    return sample_input_dir


def main():
    parser = argparse.ArgumentParser(description="Generate sample YOLO predictions.")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained YOLO best.pt model."
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="yolo_human_car",
        help="YOLO run name used to auto-detect best.pt."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="data/visdrone_human_car/images/val",
        help="Source image folder."
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for prediction."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold."
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sample images to predict."
    )

    parser.add_argument(
        "--project",
        type=str,
        default="outputs/task02",
        help="Output project folder."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="sample_predictions",
        help="Output prediction folder name."
    )

    args = parser.parse_args()

    if args.model is not None and Path(args.model).exists():
        model_path = Path(args.model)
    else:
        model_path = find_best_model(args.run_name)

    source_dir = Path(args.source)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    sample_input_dir = Path(args.project) / "sample_inputs"
    prediction_source = collect_sample_images(
        source_dir=source_dir,
        sample_input_dir=sample_input_dir,
        num_samples=args.num_samples
    )

    print("Using model:", model_path)
    print("Prediction source:", prediction_source)

    model = YOLO(str(model_path))

    model.predict(
        source=str(prediction_source),
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    print("\nSample predictions saved at:")
    print(Path(args.project) / args.name)


if __name__ == "__main__":
    main()