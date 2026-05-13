from pathlib import Path
import shutil
import yaml
from tqdm import tqdm


# =========================
# Paths
# =========================

ORIGINAL_DATA_ROOT = Path("data/VisDrone_Dataset")
OUTPUT_DATA_ROOT = Path("data/visdrone_human_car")

TRAIN_IMG_DIR = ORIGINAL_DATA_ROOT / "VisDrone2019-DET-train" / "images"
TRAIN_LABEL_DIR = ORIGINAL_DATA_ROOT / "VisDrone2019-DET-train" / "labels"

VAL_IMG_DIR = ORIGINAL_DATA_ROOT / "VisDrone2019-DET-val" / "images"
VAL_LABEL_DIR = ORIGINAL_DATA_ROOT / "VisDrone2019-DET-val" / "labels"

# If True, images without human/car will also be copied with empty label files.
# This can help reduce false positives, but makes the prepared dataset larger.
INCLUDE_EMPTY_LABEL_IMAGES = True


# =========================
# Class loading
# =========================

def load_class_names(data_root: Path):
    yaml_files = list(data_root.glob("*.yaml")) + list(data_root.glob("*.yml"))

    if len(yaml_files) == 0:
        print("No YAML file found. Using fallback VisDrone class names.")
        return [
            "pedestrian", "people", "bicycle", "car", "van",
            "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"
        ]

    yaml_path = yaml_files[0]
    print(f"Using original YAML file: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names")

    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]

    return names


class_names = load_class_names(ORIGINAL_DATA_ROOT)

print("Original class names:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# =========================
# Target class mapping
# =========================

def map_class_to_target(class_id: int):
    class_name = class_names[class_id].lower()

    if class_name in ["pedestrian", "people"]:
        return 0  # human

    if class_name == "car":
        return 1  # car

    return None


# =========================
# Utility functions
# =========================

def reset_output_dirs():
    for split in ["train", "val"]:
        (OUTPUT_DATA_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATA_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_image(src_img_path: Path, dst_img_path: Path):
    if not dst_img_path.exists():
        shutil.copy2(src_img_path, dst_img_path)


def process_split(img_dir: Path, label_dir: Path, split: str):
    output_img_dir = OUTPUT_DATA_ROOT / "images" / split
    output_label_dir = OUTPUT_DATA_ROOT / "labels" / split

    image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    copied_images = 0
    total_target_boxes = 0
    human_boxes = 0
    car_boxes = 0

    for img_path in tqdm(image_paths, desc=f"Processing {split}"):
        label_path = label_dir / f"{img_path.stem}.txt"
        new_label_lines = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    class_id = int(float(parts[0]))
                    target_class_id = map_class_to_target(class_id)

                    if target_class_id is None:
                        continue

                    # Keep same YOLO bbox coordinates, only change class ID.
                    new_line = f"{target_class_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
                    new_label_lines.append(new_line)

                    total_target_boxes += 1
                    if target_class_id == 0:
                        human_boxes += 1
                    elif target_class_id == 1:
                        car_boxes += 1

        if len(new_label_lines) > 0 or INCLUDE_EMPTY_LABEL_IMAGES:
            dst_img_path = output_img_dir / img_path.name
            dst_label_path = output_label_dir / f"{img_path.stem}.txt"

            copy_image(img_path, dst_img_path)

            with open(dst_label_path, "w") as f:
                if len(new_label_lines) > 0:
                    f.write("\n".join(new_label_lines) + "\n")

            copied_images += 1

    print(f"\n{split.upper()} summary")
    print(f"Copied images: {copied_images}")
    print(f"Target boxes: {total_target_boxes}")
    print(f"Human boxes: {human_boxes}")
    print(f"Car boxes: {car_boxes}")


def create_dataset_yaml():
    yaml_path = OUTPUT_DATA_ROOT / "human_car.yaml"

    data = {
        "path": str(OUTPUT_DATA_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "human",
            1: "car"
        }
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"\nCreated dataset YAML: {yaml_path}")


def main():
    reset_output_dirs()

    process_split(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, "train")
    process_split(VAL_IMG_DIR, VAL_LABEL_DIR, "val")

    create_dataset_yaml()

    print("\nPrepared YOLO dataset saved at:")
    print(OUTPUT_DATA_ROOT)


if __name__ == "__main__":
    main()