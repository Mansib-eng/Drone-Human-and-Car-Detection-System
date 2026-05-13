from pathlib import Path
from ultralytics import YOLO


def main():
    data_yaml = Path("data/visdrone_human_car/human_car.yaml")

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run this first: python src/prepare_training_data.py"
        )

    # YOLOv8n is lightweight and good for internship/demo projects.
    # You can change to yolov8s.pt for better accuracy if you have a GPU.
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs/detect",
        name="yolo_human_car",
        pretrained=True,
        patience=10,
        plots=True
    )

    print("\nTraining complete.")
    print("Best model should be saved at:")
    print("runs/detect/yolo_human_car/weights/best.pt")


if __name__ == "__main__":
    main()