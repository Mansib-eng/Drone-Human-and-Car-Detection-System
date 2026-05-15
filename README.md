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

## Project Structure

```txt
drone-human-car-detection/
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_train_yolo_colab.ipynb
│   ├── 03_detection_counting.ipynb
│   ├── 04_bytetrack_tracking.ipynb
│   └── 05_evaluation_visualization.ipynb
│
├── src/
│   ├── explore_dataset.py
│   ├── prepare_training_data.py
│   ├── train.py
│   ├── predict_samples.py
│   ├── detect_count.py
│   ├── track.py
│   └── evaluate_visualize.py
│
├── outputs/
│   ├── task01/
│   │   ├── class_distribution.png
│   │   ├── bbox_area_distribution.png
│   │   └── sample_annotated_images.jpg
│   │
│   ├── task02/
│   │   ├── results.png
│   │   ├── results.csv
│   │   ├── confusion_matrix.png
│   │   ├── sample_inputs/
│   │   └── sample_predictions/
│   │
│   ├── task03/
│   │   ├── processed_images/
│   │   └── detection_counts.csv
│   │
│   ├── task04/
│   │   ├── tracking_input_frames/
│   │   ├── tracked_frames/
│   │   ├── tracking_summary.csv
│   │   └── tracking_output.mp4
│   │
│   └── task05/
│       ├── selected_prediction_outputs/
│       ├── selected_counting_visualizations/
│       ├── selected_tracking_outputs/
│       ├── prediction_collage.jpg
│       ├── counting_collage.jpg
│       ├── tracking_collage.jpg
│       ├── metrics_summary.csv
│       └── evaluation_summary.md
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

The YOLO training folder `runs/` is generated automatically during training. Selected result images are copied into `outputs/` for documentation and submission. Large videos such as `tracking_output.mp4` can be shared through Google Drive if GitHub file size becomes an issue.

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

# Task-03: Human and Car Detection with Human Counting

Task-03 uses the trained YOLO model to detect humans and cars in drone/aerial images.

The system performs three main operations:

1. Detect humans and cars.
2. Draw bounding boxes around detected objects.
3. Display the total human count on the output image.

This directly satisfies the assessment requirement for human/car detection, bounding box visualization, and human counting.

---

## Counting Logic

The trained model has two target classes:

| Class ID | Class Name |
|---|---|
| 0 | human |
| 1 | car |

The human counting logic is simple:

```txt
total_humans = number of detections where class_name == "human"
```

Cars are also detected and displayed, but the required count is focused on humans.

---

## Task-03 Source File

```txt
src/detect_count.py
```

This script performs the complete detection and counting pipeline:

- loads the trained YOLO model,
- runs inference on input images,
- detects humans and cars,
- draws bounding boxes,
- displays total human count,
- displays car count for additional context,
- saves processed images,
- saves a CSV summary of detection counts.

---

## Task-03 Notebook

```txt
notebooks/03_detection_counting.ipynb
```

This notebook is used to run the Task-03 pipeline, display processed images, and inspect the detection count results.

---

## Running Task-03

```bash
python src/detect_count.py \
    --source outputs/task02/sample_inputs \
    --output outputs/task03/processed_images \
    --imgsz 640 \
    --conf 0.25
```

If the model path is not detected automatically, provide it manually:

```bash
python src/detect_count.py \
    --model runs/detect/yolo_human_car/weights/best.pt \
    --source outputs/task02/sample_inputs \
    --output outputs/task03/processed_images \
    --imgsz 640 \
    --conf 0.25
```

---

## Task-03 Outputs

The processed images are saved in:

```txt
outputs/task03/processed_images/
```

Each processed image contains:

- bounding boxes for humans,
- bounding boxes for cars,
- total human count displayed on the image,
- car count displayed for additional context.

A CSV summary is saved at:

```txt
outputs/task03/detection_counts.csv
```

The CSV file contains:

```txt
image_name
human_count
car_count
output_path
```

---

## Task-03 Output Structure

```txt
outputs/task03/
├── processed_images/
└── detection_counts.csv
```

---

## Observation

The system successfully detects humans and cars from drone/aerial images and displays the total human count on each processed image.

In dense scenes, some labels may overlap because many small objects appear close together. This is a common challenge in aerial object detection, especially when humans and vehicles appear small or crowded.

---

# Task-04: Optional Object Tracking

Task-04 adds object tracking as a bonus feature. I used **ByteTrack** with the trained YOLO model because it integrates well with YOLO-based detection pipelines and can assign persistent IDs to detected objects across frames.

The tracking pipeline detects humans and cars in each frame and then assigns tracking IDs to objects as they move through the image sequence.

---

## Tracking Method

The selected tracking method is:

```txt
YOLOv8 + ByteTrack
```

YOLO performs object detection, and ByteTrack performs object association across frames.

The tracker is configured using:

```txt
bytetrack.yaml
```

---

## Why ByteTrack?

ByteTrack was selected because:

- it works directly with Ultralytics YOLO tracking mode,
- it is lightweight and fast,
- it is suitable for multi-object tracking,
- it can track objects across consecutive frames using detection results,
- it is practical for a short internship assessment demo.

---

## Task-04 Source File

```txt
src/track.py
```

This script performs the complete tracking pipeline:

- loads the trained YOLO model,
- reads an input video or ordered image sequence,
- detects humans and cars,
- assigns tracking IDs using ByteTrack,
- draws bounding boxes with class names and track IDs,
- displays current human/car counts,
- displays unique tracked human/car IDs,
- saves tracked frames,
- saves a tracking video,
- saves a tracking summary CSV.

---

## Task-04 Notebook

```txt
notebooks/04_bytetrack_tracking.ipynb
```

This notebook is used to prepare input frames, run the ByteTrack pipeline, display tracked frames, view the tracking summary CSV, and generate the tracking output video.

---

## Running Task-04

```bash
python src/track.py \
    --source outputs/task04/tracking_input_frames \
    --output-dir outputs/task04 \
    --imgsz 640 \
    --conf 0.25 \
    --tracker bytetrack.yaml \
    --fps 10 \
    --max-frames 60
```

If the model path is not detected automatically, provide it manually:

```bash
python src/track.py \
    --model runs/detect/yolo_human_car/weights/best.pt \
    --source outputs/task04/tracking_input_frames \
    --output-dir outputs/task04 \
    --imgsz 640 \
    --conf 0.25 \
    --tracker bytetrack.yaml \
    --fps 10 \
    --max-frames 60
```

---

## Task-04 Outputs

Task-04 outputs are saved in:

```txt
outputs/task04/
```

This folder contains:

```txt
outputs/task04/
├── tracking_input_frames/
├── tracked_frames/
├── tracking_summary.csv
└── tracking_output.mp4
```

### `tracking_input_frames/`

This folder contains the ordered image frames used as input for tracking.

### `tracked_frames/`

This folder contains the processed frames with:

- bounding boxes,
- class names,
- tracking IDs,
- current human/car counts,
- unique human/car ID counts.

### `tracking_summary.csv`

This CSV file stores tracking information for each frame, including:

```txt
frame_index
frame_name
human_count
car_count
human_track_ids
car_track_ids
unique_humans_so_far
unique_cars_so_far
```

### `tracking_output.mp4`

This video shows the tracking output across consecutive frames.

If the video file is large, it can be shared through Google Drive instead of being uploaded directly to GitHub.

---

## Task-04 Observation

The tracking output demonstrates that the system can assign persistent IDs to detected objects across frames.

In the selected demo sequence, cars were tracked more consistently than humans because humans were smaller, less frequent, and harder to detect from the aerial viewpoint. This is a common limitation in drone-based small-object tracking.

Even with this limitation, the tracking implementation adds a useful bonus feature by showing temporal object association beyond single-image detection.

---

# Task-05: Evaluation and Visualization

Task-05 summarizes the final outputs of the complete computer vision pipeline. This section shows prediction outputs, counting visualizations, processed images/results, evaluation metrics, strengths, limitations, and challenges faced.

The goal of this task is to demonstrate that the trained system is not only able to make predictions, but also that the results can be inspected, summarized, and evaluated clearly.

---

## Task-05 Source File

```txt
src/evaluate_visualize.py
```

This script collects outputs from previous tasks and generates a final evaluation folder. It summarizes:

- YOLO prediction outputs from Task-02,
- human counting visualizations from Task-03,
- optional tracking outputs from Task-04,
- training metrics such as precision, recall, and mAP,
- approximate FPS if the trained model is available locally,
- strengths, limitations, and challenges faced.

---

## Task-05 Notebook

```txt
notebooks/05_evaluation_visualization.ipynb
```

This notebook is used to run the evaluation script, inspect generated metrics, display result collages, and view the final evaluation summary.

---

## Running Task-05

```bash
python src/evaluate_visualize.py \
    --results-csv outputs/task02/results.csv \
    --counts-csv outputs/task03/detection_counts.csv \
    --tracking-csv outputs/task04/tracking_summary.csv \
    --prediction-dir outputs/task02/sample_predictions \
    --counting-dir outputs/task03/processed_images \
    --tracking-dir outputs/task04/tracked_frames \
    --output-dir outputs/task05 \
    --imgsz 640 \
    --conf 0.25 \
    --max-images 10
```

---

## Task-05 Outputs

Task-05 outputs are saved in:

```txt
outputs/task05/
```

This folder contains:

```txt
outputs/task05/
├── selected_prediction_outputs/
├── selected_counting_visualizations/
├── selected_tracking_outputs/
├── prediction_collage.jpg
├── counting_collage.jpg
├── tracking_collage.jpg
├── metrics_summary.csv
└── evaluation_summary.md
```

---

## Prediction Outputs

Prediction outputs from Task-02 are collected from:

```txt
outputs/task02/sample_predictions/
```

These images show YOLO-predicted bounding boxes for the two target classes:

```txt
human
car
```

Selected prediction examples are copied into:

```txt
outputs/task05/selected_prediction_outputs/
```

A combined prediction visualization is saved as:

```txt
outputs/task05/prediction_collage.jpg
```

---

## Counting Visualization

Counting visualizations from Task-03 are collected from:

```txt
outputs/task03/processed_images/
```

Each processed image contains:

- human bounding boxes,
- car bounding boxes,
- total human count,
- car count for additional context.

Selected counting visualizations are copied into:

```txt
outputs/task05/selected_counting_visualizations/
```

A combined counting visualization is saved as:

```txt
outputs/task05/counting_collage.jpg
```

---

## Tracking Visualization

Tracking outputs from Task-04 are collected from:

```txt
outputs/task04/tracked_frames/
```

Selected tracking outputs are copied into:

```txt
outputs/task05/selected_tracking_outputs/
```

A combined tracking visualization is saved as:

```txt
outputs/task05/tracking_collage.jpg
```

The tracking video is available at:

```txt
outputs/task04/tracking_output.mp4
```

If the video file is large, it can be submitted through Google Drive instead of being uploaded directly to GitHub.

---

## Metrics Summary

The final metrics summary is saved at:

```txt
outputs/task05/metrics_summary.csv
```

The metrics are collected from YOLO training and generated output CSV files.

Included metrics may include:

| Metric | Description |
|---|---|
| Precision | How many predicted objects were correct |
| Recall | How many ground-truth objects were detected |
| mAP50 | Mean Average Precision at IoU threshold 0.50 |
| mAP50-95 | Mean Average Precision averaged over IoU thresholds 0.50 to 0.95 |
| FPS | Approximate inference speed if the trained model is available locally |
| Total humans detected | Sum of detected humans in processed sample images |
| Total cars detected | Sum of detected cars in processed sample images |
| Unique tracked IDs | Number of unique tracked objects from ByteTrack output |

The main YOLO training metrics are read from:

```txt
outputs/task02/results.csv
```

The counting metrics are read from:

```txt
outputs/task03/detection_counts.csv
```

The tracking metrics are read from:

```txt
outputs/task04/tracking_summary.csv
```

---

## Evaluation Summary Report

A Markdown evaluation report is generated at:

```txt
outputs/task05/evaluation_summary.md
```

This report includes:

- prediction output summary,
- counting visualization summary,
- processed result summary,
- precision, recall, mAP, and optional FPS,
- strengths,
- limitations,
- challenges faced,
- final conclusion.

---

## Strengths

- The project covers the full computer vision workflow: dataset understanding, preprocessing, model training, inference, counting, visualization, evaluation, and optional tracking.
- YOLOv8 provides a lightweight and practical object detection pipeline for drone/aerial images.
- The original VisDrone labels were filtered into task-specific classes: `human` and `car`.
- The system displays bounding boxes and total human count directly on processed images.
- Results are saved in both visual form and CSV format, making the outputs easier to inspect and reproduce.
- ByteTrack adds a useful bonus tracking feature by assigning persistent IDs across consecutive frames.

---

## Limitations

- Humans are often very small in drone images, making them harder to detect than cars.
- Dense scenes can cause overlapping bounding boxes and labels.
- Occlusion, shadows, overexposure, and motion blur can reduce detection performance.
- The counting method is detection-based, so missed detections directly reduce the final human count.
- Cars are tracked more consistently than humans because cars are larger and clearer from aerial views.
- The model was trained as a lightweight internship assessment pipeline, so performance can be improved with more tuning and longer experimentation.

---

## Challenges Faced

- The original dataset contained many classes, so it had to be filtered into only `human` and `car`.
- Local CPU training was slow, so Google Colab GPU training was used.
- Drone images contain strong scale variation, lighting variation, shadows, and dense object layouts.
- Small-object detection was challenging because humans occupy very few pixels in many aerial images.
- Tracking required ordered frames; random validation images are not ideal for object tracking.
- Some tracking IDs were more stable for cars than humans due to inconsistent human detections across frames.

---

## Task-05 Conclusion

The final system successfully demonstrates prediction outputs, counting visualization, processed results, evaluation metrics, and analysis of strengths and limitations. The pipeline satisfies the final evaluation and visualization requirement by showing both quantitative and qualitative results for drone-based human and car detection.

---

## Final Status

| Task | Status |
|---|---|
| Task-01: Dataset Understanding & Preprocessing | Completed |
| Task-02: YOLO Model Training | Completed |
| Task-03: Human & Car Detection with Human Counting | Completed |
| Task-04: Optional ByteTrack Tracking | Completed |
| Task-05: Evaluation & Visualization | Completed |

---

## Demo Video

A short 3–5 minute demo video is included in the submitted Google Drive folder. The video demonstrates:

- dataset understanding,
- YOLO training results,
- human/car detection,
- human counting visualization,
- optional ByteTrack tracking,
- final evaluation outputs.


