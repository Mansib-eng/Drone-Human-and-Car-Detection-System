from pathlib import Path
import random
import yaml
import cv2
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. Set dataset path
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "VisDrone_Dataset"

TRAIN_IMG_DIR = DATA_ROOT / "VisDrone2019-DET-train" / "images"
TRAIN_LABEL_DIR = DATA_ROOT / "VisDrone2019-DET-train" / "labels"

VAL_IMG_DIR = DATA_ROOT / "VisDrone2019-DET-val" / "images"
VAL_LABEL_DIR = DATA_ROOT / "VisDrone2019-DET-val" / "labels"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task01"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Load class names
# =========================

def load_class_names(data_root: Path):
    yaml_files = list(data_root.glob("*.yaml")) + list(data_root.glob("*.yml"))

    if len(yaml_files) == 0:
        print("No YAML file found. Using fallback class names.")
        return [
            "pedestrian", "people", "bicycle", "car", "van",
            "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"
        ]

    yaml_path = yaml_files[0]
    print(f"Using YAML file: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names")

    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]

    return names


class_names = load_class_names(DATA_ROOT)
print("Class names:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# =========================
# 3. Count images and labels
# =========================

def count_files(img_dir: Path, label_dir: Path, split_name: str):
    image_files = (
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.jpeg"))
        + list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.bmp"))
    )
    label_files = list(label_dir.glob("*.txt"))

    print(f"\n{split_name} set:")
    print(f"Images: {len(image_files)}")
    print(f"Labels: {len(label_files)}")

    return image_files, label_files


train_images, train_labels = count_files(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, "Train")
val_images, val_labels = count_files(VAL_IMG_DIR, VAL_LABEL_DIR, "Validation")


# =========================
# 4. Inspect one label file
# =========================

print("\nSample label file content:")
if train_labels:
    sample_label = train_labels[0]
    print(f"File: {sample_label.name}")

    with open(sample_label, "r") as f:
        lines = f.readlines()[:5]

    for line in lines:
        print(line.strip())


# =========================
# 5. Read all labels into a DataFrame
# =========================

def read_yolo_labels(label_dir: Path, split_name: str):
    rows = []

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, width, height = parts

                class_id = int(float(class_id))
                x_center = float(x_center)
                y_center = float(y_center)
                width = float(width)
                height = float(height)

                class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)

                rows.append({
                    "split": split_name,
                    "label_file": label_file.name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "bbox_area": width * height
                })

    return pd.DataFrame(rows)


train_df = read_yolo_labels(TRAIN_LABEL_DIR, "train")
val_df = read_yolo_labels(VAL_LABEL_DIR, "val")

df = pd.concat([train_df, val_df], ignore_index=True)

print("\nDataset annotation summary:")
print(df.head())
print("\nTotal annotations:", len(df))


# =========================
# 6. Class distribution plot
# =========================

if df.empty:
    print("\nNo annotations were loaded, so plots and sample visualization were skipped.")
else:
    class_counts = df["class_name"].value_counts()

    plt.figure(figsize=(12, 6))
    class_counts.plot(kind="bar")
    plt.title("Class Distribution in VisDrone Dataset")
    plt.xlabel("Class")
    plt.ylabel("Number of Objects")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200)
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'class_distribution.png'}")


# =========================
# 7. Bounding box size distribution
# =========================

if not df.empty:
    plt.figure(figsize=(8, 5))
    plt.hist(df["bbox_area"], bins=50)
    plt.title("Bounding Box Area Distribution")
    plt.xlabel("Normalized Bounding Box Area")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bbox_area_distribution.png", dpi=200)
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'bbox_area_distribution.png'}")


# =========================
# 8. Human and car class statistics
# =========================

target_classes = ["pedestrian", "people", "car"]
target_df = df[df["class_name"].isin(target_classes)] if not df.empty else pd.DataFrame()

print("\nTarget class statistics:")
if target_df.empty:
    print("No target-class annotations found.")
else:
    print(target_df["class_name"].value_counts())


# =========================
# 9. Draw bounding boxes on sample images
# =========================

def draw_yolo_boxes(image_path: Path, label_path: Path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_h, img_w = image.shape[:2]

    if not label_path.exists():
        return image

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id, x_center, y_center, width, height = parts

        class_id = int(float(class_id))
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width)
        height = float(height)

        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)

        # Only visualize important classes for this assignment
        if class_name not in ["pedestrian", "people", "car"]:
            continue

        x1 = int((x_center - width / 2) * img_w)
        y1 = int((y_center - height / 2) * img_h)
        x2 = int((x_center + width / 2) * img_w)
        y2 = int((y_center + height / 2) * img_h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image,
            class_name,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    return image


random.seed(42)
if train_images:
    sample_images = random.sample(train_images, min(6, len(train_images)))

    plt.figure(figsize=(14, 10))

    for i, image_path in enumerate(sample_images):
        label_path = TRAIN_LABEL_DIR / f"{image_path.stem}.txt"
        annotated_image = draw_yolo_boxes(image_path, label_path)

        plt.subplot(2, 3, i + 1)
        plt.imshow(annotated_image)
        plt.title(image_path.name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_annotated_images.jpg", dpi=200)
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'sample_annotated_images.jpg'}")
else:
    print("No training images found, so sample visualization was skipped.")