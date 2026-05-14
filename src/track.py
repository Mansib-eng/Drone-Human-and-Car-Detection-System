from pathlib import Path
import argparse
import cv2
import pandas as pd
from ultralytics import YOLO


def find_best_model(run_name: str = "yolo_human_car") -> Path:
    """
    Automatically finds best.pt from the project folder.
    This supports normal and nested YOLO run folders.
    """
    search_roots = [Path("runs"), Path("."), Path("outputs")]
    candidates = []

    for root in search_roots:
        if root.exists():
            candidates.extend([p for p in root.rglob("best.pt") if run_name in str(p) or p.name == "best.pt"])

    if not candidates:
        raise FileNotFoundError(
            f"Could not find best.pt for run name '{run_name}'. "
            "Please provide the model path using --model."
        )

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def get_image_paths(source_path: Path):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_paths = []

    if source_path.is_file() and source_path.suffix.lower() in image_extensions:
        return [source_path]

    if source_path.is_dir():
        for ext in image_extensions:
            image_paths.extend(source_path.glob(f"*{ext}"))
        return sorted(image_paths)

    return []


def is_video_file(source_path: Path) -> bool:
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    return source_path.is_file() and source_path.suffix.lower() in video_extensions


def draw_box(image, box_xyxy, class_name, track_id, confidence):
    x1, y1, x2, y2 = map(int, box_xyxy)

    if class_name == "human":
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    id_text = f"ID:{track_id}" if track_id is not None else "ID:N/A"
    label = f"{class_name} {id_text} {confidence:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 6, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1
    )

    return image


def draw_tracking_panel(image, frame_idx, human_count, car_count, unique_humans, unique_cars):
    panel_h = 130
    cv2.rectangle(image, (10, 10), (360, panel_h), (0, 0, 0), -1)

    lines = [
        f"Frame: {frame_idx}",
        f"Current Humans: {human_count}",
        f"Current Cars: {car_count}",
        f"Unique Human IDs: {len(unique_humans)}",
        f"Unique Car IDs: {len(unique_cars)}"
    ]

    y = 35
    for line in lines:
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )
        y += 22

    return image


def process_frame(model, frame, frame_idx, frame_name, tracker, imgsz, conf, unique_humans, unique_cars):
    results = model.track(
        source=frame,
        persist=True,
        tracker=tracker,
        imgsz=imgsz,
        conf=conf,
        verbose=False
    )

    result = results[0]

    human_count = 0
    car_count = 0
    human_track_ids = []
    car_track_ids = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            confidence = float(box.conf[0].item())

            if class_name not in ["human", "car"]:
                continue

            xyxy = box.xyxy[0].cpu().numpy()

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())

            if class_name == "human":
                human_count += 1
                if track_id is not None:
                    human_track_ids.append(track_id)
                    unique_humans.add(track_id)

            elif class_name == "car":
                car_count += 1
                if track_id is not None:
                    car_track_ids.append(track_id)
                    unique_cars.add(track_id)

            frame = draw_box(frame, xyxy, class_name, track_id, confidence)

    frame = draw_tracking_panel(
        frame,
        frame_idx=frame_idx,
        human_count=human_count,
        car_count=car_count,
        unique_humans=unique_humans,
        unique_cars=unique_cars
    )

    summary = {
        "frame_index": frame_idx,
        "frame_name": frame_name,
        "human_count": human_count,
        "car_count": car_count,
        "human_track_ids": " ".join(map(str, sorted(set(human_track_ids)))),
        "car_track_ids": " ".join(map(str, sorted(set(car_track_ids)))),
        "unique_humans_so_far": len(unique_humans),
        "unique_cars_so_far": len(unique_cars)
    }

    return frame, summary


def main():
    parser = argparse.ArgumentParser(description="Track humans and cars using YOLO + ByteTrack.")

    parser.add_argument("--model", type=str, default=None, help="Path to trained YOLO model best.pt")
    parser.add_argument("--run-name", type=str, default="yolo_human_car", help="YOLO run name for auto model search")
    parser.add_argument("--source", type=str, default="outputs/task04/tracking_input_frames", help="Input video file or image folder")
    parser.add_argument("--output-dir", type=str, default="outputs/task04", help="Output folder")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracking config: bytetrack.yaml or botsort.yaml")
    parser.add_argument("--fps", type=int, default=10, help="FPS for output video when source is image sequence")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process. 0 means all frames.")

    args = parser.parse_args()

    if args.model is not None and Path(args.model).exists():
        model_path = Path(args.model)
    else:
        model_path = find_best_model(args.run_name)

    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    tracked_frames_dir = output_dir / "tracked_frames"
    tracked_frames_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = output_dir / "tracking_output.mp4"
    csv_path = output_dir / "tracking_summary.csv"

    print("Using model:", model_path)
    print("Using tracker:", args.tracker)
    print("Source:", source_path)
    print("Output folder:", output_dir)

    model = YOLO(str(model_path))

    unique_humans = set()
    unique_cars = set()
    summary_rows = []

    video_writer = None

    if is_video_file(source_path):
        cap = cv2.VideoCapture(str(source_path))

        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {source_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = args.fps

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            tracked_frame, summary = process_frame(
                model=model,
                frame=frame,
                frame_idx=frame_idx,
                frame_name=f"frame_{frame_idx:06d}",
                tracker=args.tracker,
                imgsz=args.imgsz,
                conf=args.conf,
                unique_humans=unique_humans,
                unique_cars=unique_cars
            )

            if video_writer is None:
                h, w = tracked_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

            video_writer.write(tracked_frame)
            cv2.imwrite(str(tracked_frames_dir / f"frame_{frame_idx:06d}.jpg"), tracked_frame)
            summary_rows.append(summary)

        cap.release()

    else:
        image_paths = get_image_paths(source_path)

        if not image_paths:
            raise FileNotFoundError(f"No image/video files found at: {source_path}")

        if args.max_frames > 0:
            image_paths = image_paths[:args.max_frames]

        for frame_idx, image_path in enumerate(image_paths, start=1):
            frame = cv2.imread(str(image_path))

            if frame is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            tracked_frame, summary = process_frame(
                model=model,
                frame=frame,
                frame_idx=frame_idx,
                frame_name=image_path.name,
                tracker=args.tracker,
                imgsz=args.imgsz,
                conf=args.conf,
                unique_humans=unique_humans,
                unique_cars=unique_cars
            )

            if video_writer is None:
                h, w = tracked_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(output_video_path), fourcc, args.fps, (w, h))

            video_writer.write(tracked_frame)
            cv2.imwrite(str(tracked_frames_dir / image_path.name), tracked_frame)
            summary_rows.append(summary)

    if video_writer is not None:
        video_writer.release()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False)

    print("\nTracking complete.")
    print("Tracked frames saved at:", tracked_frames_dir)
    print("Tracking video saved at:", output_video_path)
    print("Tracking CSV saved at:", csv_path)
    print("Unique human IDs:", len(unique_humans))
    print("Unique car IDs:", len(unique_cars))


if __name__ == "__main__":
    main()
