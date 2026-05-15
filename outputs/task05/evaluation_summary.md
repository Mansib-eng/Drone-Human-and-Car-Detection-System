# Task-05: Evaluation and Visualization Report

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
| Precision | 0.6916 |
| Recall | 0.5340 |
| mAP50 | 0.5736 |
| mAP50-95 | 0.3167 |
| Total Epochs | 50 |
| Best Epoch by mAP50 | 50 |
| Approx. FPS | 66.58 |

---

## Counting Summary

| Item | Value |
|---|---:|
| Processed Images | 10 |
| Total Humans Detected | 134 |
| Total Cars Detected | 197 |
| Average Humans per Image | 13.40 |
| Average Cars per Image | 19.70 |
| Maximum Humans in One Image | 39 |
| Maximum Cars in One Image | 57 |

---

## Tracking Summary

| Item | Value |
|---|---:|
| Tracking Available | True |
| Tracking Frames | 35 |
| Unique Human IDs | 8 |
| Unique Car IDs | 52 |

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
