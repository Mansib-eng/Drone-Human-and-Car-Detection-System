from pathlib import Path
import argparse
import shutil
import time
import pandas as pd
import cv2


def find_best_model(run_name="yolo_human_car"):
    candidates = []
    for root in [Path("runs"), Path("."), Path("outputs")]:
        if root.exists():
            candidates += [p for p in root.rglob("best.pt") if run_name in str(p) or p.name == "best.pt"]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_training_metrics(results_csv):
    df = read_csv(results_csv)
    if df is None or df.empty:
        return {}

    last = df.iloc[-1]
    metrics = {
        "precision": float(last["metrics/precision(B)"]) if "metrics/precision(B)" in df.columns else None,
        "recall": float(last["metrics/recall(B)"]) if "metrics/recall(B)" in df.columns else None,
        "mAP50": float(last["metrics/mAP50(B)"]) if "metrics/mAP50(B)" in df.columns else None,
        "mAP50_95": float(last["metrics/mAP50-95(B)"]) if "metrics/mAP50-95(B)" in df.columns else None,
        "total_epochs": int(len(df)),
    }

    if "metrics/mAP50(B)" in df.columns:
        best_idx = df["metrics/mAP50(B)"].idxmax()
        metrics["best_epoch_by_mAP50"] = int(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else int(best_idx + 1)

    return metrics


def get_count_metrics(count_csv):
    df = read_csv(count_csv)
    if df is None or df.empty:
        return {}

    return {
        "processed_images": int(len(df)),
        "total_humans_detected": int(df["human_count"].sum()) if "human_count" in df.columns else None,
        "total_cars_detected": int(df["car_count"].sum()) if "car_count" in df.columns else None,
        "avg_humans_per_image": float(df["human_count"].mean()) if "human_count" in df.columns else None,
        "avg_cars_per_image": float(df["car_count"].mean()) if "car_count" in df.columns else None,
        "max_humans_in_image": int(df["human_count"].max()) if "human_count" in df.columns else None,
        "max_cars_in_image": int(df["car_count"].max()) if "car_count" in df.columns else None,
    }


def get_tracking_metrics(tracking_csv):
    df = read_csv(tracking_csv)
    if df is None or df.empty:
        return {"tracking_available": False}

    return {
        "tracking_available": True,
        "tracking_frames": int(len(df)),
        "unique_humans": int(df["unique_humans_so_far"].max()) if "unique_humans_so_far" in df.columns else None,
        "unique_cars": int(df["unique_cars_so_far"].max()) if "unique_cars_so_far" in df.columns else None,
    }


def copy_images(src_dir, dst_dir, limit=5):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        return []

    images = sorted(
        list(src_dir.glob("*.jpg")) +
        list(src_dir.glob("*.jpeg")) +
        list(src_dir.glob("*.png"))
    )

    copied = []
    for img in images[:limit]:
        out = dst_dir / img.name
        shutil.copy2(img, out)
        copied.append(out)

    return copied


def create_collage(image_paths, output_path, width=420):
    if not image_paths:
        return

    resized_images = []

    for p in image_paths[:3]:
        img = cv2.imread(str(p))
        if img is None:
            continue

        h, w = img.shape[:2]
        new_h = int(h * (width / w))
        img = cv2.resize(img, (width, new_h))
        resized_images.append(img)

    if not resized_images:
        return

    max_h = max(img.shape[0] for img in resized_images)
    padded = []

    for img in resized_images:
        h, w = img.shape[:2]
        if h < max_h:
            img = cv2.copyMakeBorder(img, 0, max_h - h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded.append(img)

    collage = cv2.hconcat(padded)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), collage)


def benchmark_fps(model_path, source_dir, imgsz=640, conf=0.25, max_images=10):
    if model_path is None or not Path(model_path).exists():
        return None

    source_dir = Path(source_dir)
    if not source_dir.exists():
        return None

    images = sorted(list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.jpeg")) + list(source_dir.glob("*.png")))[:max_images]
    if not images:
        return None

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    # warm-up
    model.predict(str(images[0]), imgsz=imgsz, conf=conf, verbose=False)

    start = time.time()
    for img in images:
        model.predict(str(img), imgsz=imgsz, conf=conf, verbose=False)
    elapsed = time.time() - start

    return len(images) / elapsed if elapsed > 0 else None


def fmt(x, digits=4):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def write_report(path, metrics, count_metrics, tracking_metrics, fps):
    report = f"""# Task-05: Evaluation and Visualization Report

Task-05 summarizes the final outputs of the drone human and car detection pipeline.

---

## Prediction Outputs

Prediction outputs are available in:

```txt
outputs/task02/sample_predictions/
```

These images show predicted bounding boxes for:

```txt
human
car
```

---

## Counting Visualization

Counting visualizations are available in:

```txt
outputs/task03/processed_images/
```

Each processed image displays human/car bounding boxes and the total human count.

---

## Processed Results

Selected final results are collected in:

```txt
outputs/task05/
├── selected_prediction_outputs/
├── selected_counting_visualizations/
├── selected_tracking_outputs/
├── prediction_collage.jpg
├── counting_collage.jpg
├── metrics_summary.csv
└── evaluation_summary.md
```

---

## Metrics

| Metric | Value |
|---|---:|
| Precision | {fmt(metrics.get("precision"))} |
| Recall | {fmt(metrics.get("recall"))} |
| mAP50 | {fmt(metrics.get("mAP50"))} |
| mAP50-95 | {fmt(metrics.get("mAP50_95"))} |
| Total Epochs | {fmt(metrics.get("total_epochs"))} |
| Best Epoch by mAP50 | {fmt(metrics.get("best_epoch_by_mAP50"))} |
| Approx. FPS | {fmt(fps, 2)} |

---

## Counting Summary

| Item | Value |
|---|---:|
| Processed Images | {fmt(count_metrics.get("processed_images"))} |
| Total Humans Detected | {fmt(count_metrics.get("total_humans_detected"))} |
| Total Cars Detected | {fmt(count_metrics.get("total_cars_detected"))} |
| Average Humans per Image | {fmt(count_metrics.get("avg_humans_per_image"), 2)} |
| Average Cars per Image | {fmt(count_metrics.get("avg_cars_per_image"), 2)} |
| Maximum Humans in One Image | {fmt(count_metrics.get("max_humans_in_image"))} |
| Maximum Cars in One Image | {fmt(count_metrics.get("max_cars_in_image"))} |

---

## Tracking Summary

| Item | Value |
|---|---:|
| Tracking Available | {tracking_metrics.get("tracking_available", False)} |
| Tracking Frames | {fmt(tracking_metrics.get("tracking_frames"))} |
| Unique Human IDs | {fmt(tracking_metrics.get("unique_humans"))} |
| Unique Car IDs | {fmt(tracking_metrics.get("unique_cars"))} |

---

## Strengths

- The project covers the full computer vision workflow: dataset understanding, preprocessing, model training, detection, counting, visualization, evaluation, and optional tracking.
- YOLOv8 is lightweight and suitable for fast object detection.
- The dataset was filtered into two task-specific classes: `human` and `car`.
- The counting visualization clearly displays the total human count on processed images.
- CSV summaries make the outputs easier to inspect and reproduce.
- ByteTrack adds temporal object association across consecutive frames.

---

## Limitations

- Humans are often very small in drone images, making them harder to detect than cars.
- Dense scenes may cause overlapping labels and bounding boxes.
- Occlusion, shadows, overexposure, and motion blur can reduce detection quality.
- Counting is based on detections, so missed detections reduce the final count.
- Cars are tracked more consistently than humans because they are larger and easier to detect from aerial views.

---

## Challenges Faced

- The original VisDrone dataset contains many classes, so it had to be filtered into `human` and `car`.
- Local CPU training was slow, so Google Colab GPU training was used.
- Aerial images contain strong scale variation, lighting variation, and dense object layouts.
- Tracking required ordered frames; random images are not ideal for tracking.

---

## Conclusion

The final system successfully detects humans and cars, displays bounding boxes, counts total humans, visualizes processed results, reports model metrics, and optionally tracks objects using ByteTrack.
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create Task-05 evaluation and visualization summary.")
    parser.add_argument("--results-csv", default="outputs/task02/results.csv")
    parser.add_argument("--counts-csv", default="outputs/task03/detection_counts.csv")
    parser.add_argument("--tracking-csv", default="outputs/task04/tracking_summary.csv")
    parser.add_argument("--prediction-dir", default="outputs/task02/sample_predictions")
    parser.add_argument("--counting-dir", default="outputs/task03/processed_images")
    parser.add_argument("--tracking-dir", default="outputs/task04/tracked_frames")
    parser.add_argument("--output-dir", default="outputs/task05")
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-name", default="yolo_human_car")
    parser.add_argument("--fps-source", default="outputs/task02/sample_inputs")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = get_training_metrics(args.results_csv)
    count_metrics = get_count_metrics(args.counts_csv)
    tracking_metrics = get_tracking_metrics(args.tracking_csv)

    model_path = Path(args.model) if args.model and Path(args.model).exists() else find_best_model(args.run_name)
    fps = benchmark_fps(model_path, args.fps_source, args.imgsz, args.conf, args.max_images)

    pred_imgs = copy_images(args.prediction_dir, output_dir / "selected_prediction_outputs")
    count_imgs = copy_images(args.counting_dir, output_dir / "selected_counting_visualizations")
    track_imgs = copy_images(args.tracking_dir, output_dir / "selected_tracking_outputs")

    create_collage(pred_imgs, output_dir / "prediction_collage.jpg")
    create_collage(count_imgs, output_dir / "counting_collage.jpg")
    create_collage(track_imgs, output_dir / "tracking_collage.jpg")

    summary = {
        **metrics,
        **count_metrics,
        **tracking_metrics,
        "fps": fps
    }
    pd.DataFrame([summary]).to_csv(output_dir / "metrics_summary.csv", index=False)

    write_report(output_dir / "evaluation_summary.md", metrics, count_metrics, tracking_metrics, fps)

    print("Task-05 evaluation complete.")
    print("Output folder:", output_dir)
    print("Metrics summary:", output_dir / "metrics_summary.csv")
    print("Evaluation report:", output_dir / "evaluation_summary.md")


if __name__ == "__main__":
    main()
