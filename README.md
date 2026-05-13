# Drone Human and Car Detection System

This project is developed for the **Antlings Internship AI/ML Technical Assessment**.

The goal of this project is to build a computer vision pipeline for drone/aerial images that can:

- detect humans,
- detect cars,
- count total humans,
- visualize detection outputs,
- optionally perform object tracking.

The assessment focuses on dataset understanding, preprocessing, model training, inference, visualization, evaluation, and engineering reasoning.

---

## Dataset

Dataset used: **VisDrone Dataset**

Dataset link:  
https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset?resource=download

The dataset contains drone/aerial images with object detection annotations.

For this project, the main target classes are:

| Target | Dataset Classes |
|---|---|
| Human | pedestrian, people |
| Car | car |

The total human count will be calculated as:

```txt
total_humans = pedestrian_count + people_count
```

---

```txt
drone-human-car-detection/
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   └── 02_train_yolo_colab.ipynb
│
├── src/
│   ├── explore_dataset.py
│   ├── prepare_training_data.py
│   ├── train.py
│   └── predict_samples.py
│
├── outputs/
│   ├── task01/
│   │   ├── class_distribution.png
│   │   ├── bbox_area_distribution.png
│   │   └── sample_annotated_images.jpg
│   │
│   └── task02/
│       ├── results.png
│       ├── results.csv
│       ├── confusion_matrix.png
│       ├── sample_inputs/
│       └── sample_predictions/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Ignored Files and Folders

The following folders/files are not uploaded to GitHub because they are large or generated automatically:

```txt
data/
runs/
*.pt
*.pth
*.onnx
*.mp4
*.zip
kaggle.json
```

The full dataset should be downloaded from Kaggle and placed inside the `data/` folder.

The YOLO training folder `runs/` is generated automatically during training. Selected result images are copied into `outputs/` for documentation and submission.

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate
```

```bash
# Linux/Mac
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

The main packages used in this project are:

```txt
ultralytics
opencv-python
matplotlib
pandas
pyyaml
tqdm
```

---

# Task-01: Dataset Understanding & Preprocessing

The first task focuses on understanding the VisDrone dataset before model training.

## Dataset Structure

The dataset is expected to contain train, validation, and test folders. Each split contains image files and corresponding YOLO-format label files.

Expected dataset structure:

```txt
VisDrone_Dataset/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-test-dev/
│   ├── images/
│   └── labels/
└── data.yaml
```

For this project, the most important folders are:

```txt
VisDrone2019-DET-train/images
VisDrone2019-DET-train/labels
VisDrone2019-DET-val/images
VisDrone2019-DET-val/labels
```

---

## Label Format

The annotations are in YOLO format:

```txt
class_id x_center y_center width height
```

The bounding box coordinates are normalized between `0` and `1`.

Example:

```txt
3 0.512 0.443 0.051 0.087
```

This means:

```txt
class_id = object class
x_center = normalized x center of bounding box
y_center = normalized y center of bounding box
width    = normalized bounding box width
height   = normalized bounding box height
```

---

## Target Classes

Although the VisDrone dataset contains multiple object categories, this project focuses on:

```txt
pedestrian
people
car
```

The `pedestrian` and `people` classes are treated as humans.

Final mapping:

| New Target | Original Classes |
|---|---|
| human | pedestrian, people |
| car | car |

---

## Preprocessing Steps

The preprocessing plan includes:

1. Verify dataset folder structure.
2. Check image-label pairs.
3. Read class names from the dataset YAML file.
4. Inspect YOLO annotation format.
5. Visualize sample annotations.
6. Analyze class distribution.
7. Analyze bounding box size distribution.
8. Select target classes for the final task.
9. Resize/letterbox images during training.
10. Normalize image values through the YOLO training pipeline.

---

## Augmentation Plan

The planned augmentations include:

- horizontal flip,
- scaling,
- translation,
- brightness and contrast adjustment,
- mosaic augmentation,
- mild perspective transformation.

These augmentations are useful because drone images contain objects at different scales, positions, lighting conditions, and viewpoints.

---

## Dataset Challenges

The main challenges noticed in the dataset are:

1. **Small objects**  
   Humans and cars can appear very small in drone images.

2. **Dense scenes**  
   Some images contain many humans and vehicles close together.

3. **Occlusion**  
   Objects may be partially hidden by trees, buildings, vehicles, or other people.

4. **Lighting variation**  
   Outdoor aerial images may contain shadows, bright sunlight, low contrast, or blur.

5. **Similar vehicle classes**  
   Cars, vans, trucks, and buses may look similar from aerial viewpoints.

6. **Class imbalance**  
   Some classes may appear more frequently than others, which can affect model performance.

---

## Task-01 Outputs

Task-01 visualizations are saved in:

```txt
outputs/task01/
```

Generated outputs include:

```txt
class_distribution.png
bbox_area_distribution.png
sample_annotated_images.jpg
```

Detailed Task-01 analysis is available in:

```txt
notebooks/01_dataset_understanding.ipynb
```

To run Task-01:

```bash
python src/explore_dataset.py
```

---

# Task-02: Model Training

For the mandatory model training task, I used **YOLOv8** for object detection.

The original VisDrone dataset contains multiple object categories. Since this assessment focuses on human and car detection, I prepared a filtered two-class dataset.

---

## Class Mapping

| New Class ID | New Class Name | Original VisDrone Class |
|---|---|---|
| 0 | human | pedestrian, people |
| 1 | car | car |

The `pedestrian` and `people` classes were combined into a single `human` class because the final system needs to count total humans.

---

## Training Approach

The training pipeline follows these steps:

1. Load the original VisDrone YOLO-format annotations.
2. Filter only the required classes: `pedestrian`, `people`, and `car`.
3. Remap the classes into two target classes: `human` and `car`.
4. Create a new YOLO dataset configuration file: `human_car.yaml`.
5. Fine-tune a pretrained YOLOv8 model on the filtered dataset.
6. Save training curves, model weights, and sample prediction outputs.

---

## Model Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Pretrained weights | Yes |
| Input image size | 640 |
| Epochs | 50 |
| Batch size | 8 |
| Classes | human, car |

The main training command used was:

```bash
yolo detect train model=yolov8n.pt data=data/visdrone_human_car/human_car.yaml epochs=50 imgsz=640 batch=8 project=runs/detect name=yolo_human_car patience=10 plots=True device=0
```

---

## Google Colab Training

Local CPU training was slow, so GPU training was performed using Google Colab.

The Colab training notebook is available at:

```txt
notebooks/02_train_yolo_colab.ipynb
```

This notebook includes:

- GPU check,
- dependency installation,
- dataset loading,
- dataset preparation,
- YOLO training,
- training result visualization,
- sample prediction generation,
- output saving.

---

## Task-02 Source Files

The source files used for Task-02 are:

```txt
src/prepare_training_data.py
src/train.py
src/predict_samples.py
```

### `prepare_training_data.py`

This script prepares a filtered two-class YOLO dataset.

It converts:

```txt
pedestrian + people → human
car                 → car
```

and creates:

```txt
data/visdrone_human_car/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── human_car.yaml
```

### `train.py`

This script fine-tunes the YOLOv8 model on the filtered human/car dataset.

### `predict_samples.py`

This script uses the trained model to generate sample predictions on validation images.

---

## Generated Training Outputs

During YOLO training, the following folder is generated automatically:

```txt
runs/detect/yolo_human_car/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
├── confusion_matrix.png
├── results.csv
├── confusion_matrix_normalized.png
├── labels.jpg
└── args.yaml
```

This folder is not uploaded to GitHub because it may contain large files. Selected result images are copied into `outputs/task02/` for documentation and submission.

---

## Task-02 Outputs

Selected training and prediction outputs are saved in:

```txt
outputs/task02/
```

This folder contains:

```txt
outputs/task02/
├── results.png
├── results.csv
├── confusion_matrix.png
├── sample_inputs/
└── sample_predictions/

```

### Training Results

The file below shows YOLO training curves, including training loss, validation loss, and validation performance:

```txt
outputs/task02/results.png
```

The confusion matrix shows how well the model distinguishes between the two target classes:

```txt
outputs/task02/confusion_matrix.png
```

The two target classes are:

```txt
human
car
```

---

## Sample Predictions

After training, sample predictions were generated using the best trained YOLO model:

```txt
runs/detect/yolo_human_car/weights/best.pt
```

The prediction outputs are saved in:

```txt
outputs/task02/sample_predictions/
```

These images show predicted bounding boxes for humans and cars in drone/aerial images.

The sample predictions visually verify that the trained model can detect the required target objects before moving to the counting and final visualization pipeline.

To generate sample predictions:

```bash
python src/predict_samples.py
```

---

## Running Task-02

Run the following commands in order:

```bash
python src/prepare_training_data.py
```

```bash
python src/train.py
```

```bash
python src/predict_samples.py
```

Or use the Colab notebook:

```txt
notebooks/02_train_yolo_colab.ipynb
```

---
