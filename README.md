# RF-DETR: SOTA Real-Time Object Detection Model

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

RF-DETR is a real-time, transformer-based object detection model architecture developed by Roboflow and released under the Apache 2.0 license.

RF-DETR is the first real-time model to exceed 60 AP on the [Microsoft COCO benchmark](https://cocodataset.org/#home) alongside competitive performance at base sizes. It also achieves state-of-the-art performance on [RF100-VL](https://github.com/roboflow/rf100-vl)), an object detection benchmark that measures model domain adaptability to real world problems. RF-DETR is comparable speed to current real-time objection models.

**RF-DETR is small enough to run on the edge, making it an ideal model for deployments that need both strong accuracy and real-time performance.**

## Results

We validated the performance of RF-DETR on both Microsoft COCO and the RF100-VL benchmarks.

![rf-detr-coco-rf100-vl-8](https://github.com/user-attachments/assets/99b51d31-1029-4a73-b2a2-675cd45d35f0)

| Model                | params<br><sup>(M) | Latency<br><sup>T4 b1<br>(ms) | mAP<sup>COCO val<br>50-95 | mAP<sup>RF100-VL<br>50 | mAP<sup>RF100-VL<br>50-95 | Config                                                                                                        |
|----------------------|--------------------|-------------------------------|---------------------------|------------------------|---------------------------|---------------------------------------------------------------------------------------------------------------| 
| **RF-DETR-L (ours)** | **128.0**          | **22.0**                      | **59.0**                  | **coming soon**       | **coming soon**          | [link](https://github.com/roboflow/r-flow/blob/08f012eb210b1b0e03a5aa7d4e5a4c265cb20b0e/rfdetr/config.py#L26) |
| **RF-DETR-B (ours)** | **29.0**           | **6.0**                       | **53.3**                  | **86.7**               | **60.2**                  | [link](https://github.com/roboflow/r-flow/blob/08f012eb210b1b0e03a5aa7d4e5a4c265cb20b0e/rfdetr/config.py#L37) |

<details>
<summary>Full benchmark results</summary>

| Model                | params<br><sup>(M) | Latency<br><sup>T4 b1<br>(ms) | mAP<sup>COCO val<br>50-95 | mAP<sup>RF100-VL<br>50 | mAP<sup>RF100-VL<br>50-95 | Config                                                                                                        |
|----------------------|--------------------|-------------------------------|---------------------------|------------------------|---------------------------|---------------------------------------------------------------------------------------------------------------| 
| **RF-DETR-L (ours)** | **128.0**          | **22.0**                      | **59.0**                  | **coming soon**       | **coming soon**          | [link](https://github.com/roboflow/r-flow/blob/08f012eb210b1b0e03a5aa7d4e5a4c265cb20b0e/rfdetr/config.py#L26) |
| **RF-DETR-B (ours)** | **29.0**           | **6.0**                       | **53.3**                  | **86.7**               | **60.2**                  | [link](https://github.com/roboflow/r-flow/blob/08f012eb210b1b0e03a5aa7d4e5a4c265cb20b0e/rfdetr/config.py#L37) |
| LW-DETR-X            | 118.0              | 19.1                          | 58.3                      | -                      | -                         | -                                                                                                             |
| LW-DETR-L            | 46.8               | 8.8                           | 56.1                      | -                      | -                         | -                                                                                                             |
| LW-DETR-M            | 28.2               | 5.6                           | 52.5                      | 84.0                   | 57.5                      | -                                                                                                             |
| LW-DETR-S            | 14.6               | 2.9                           | 48.0                      | 84.4                   | 57.9                      | -                                                                                                             |
| LW-DETR-T            | 12.1               | 2.0                           | 42.6                      | -                      | -                         | -                                                                                                             |
| YOLOv12x             | 59.1               | 11.8                          | 55.2                      | -                      | -                         | -                                                                                                             |
| YOLOv12l             | 26.4               | 6.8                           | 53.7                      | -                      | -                         | -                                                                                                             |
| YOLOv12m             | 20.2               | 4.9                           | 52.5                      | -                      | -                         | -                                                                                                             |
| YOLOv12s             | 9.3                | 2.6                           | 48.0                      | -                      | -                         | -                                                                                                             |
| YOLOv12n             | 2.6                | 1.6                           | 40.6                      | -                      | -                         | -                                                                                                             |
| YOLO11x              | 56.9               | 11.3                          | 54.7                      | -                      | -                         | -                                                                                                             |
| YOLO11l              | 25.3               | 6.2                           | 53.4                      | -                      | -                         | -                                                                                                             |
| YOLO11m              | 20.0               | 4.7                           | 51.5                      | 84.9                   | 59.7                      | -                                                                                                             |
| YOLO11s              | 9.4                | 2.5                           | 47.0                      | 84.7                   | 59.0                      | -                                                                                                             |
| YOLO11n              | 2.6                | 1.5                           | 39.5                      | 83.2                   | 57.3                      | -                                                                                                             |
| YOLOv8x              | 68.2               | 19.1                          | 54.5                      | -                      | -                         | -                                                                                                             |
| YOLOv8l              | 43.7               | 13.2                          | 53.3                      | -                      | -                         | -                                                                                                             |
| YOLOv8m              | 28.9               | 10.1                          | 50.6                      | -                      | 59.8                      | -                                                                                                             |
| YOLOv8s              | 11.1               | 7.0                           | 45.2                      | -                      | 59.2                      | -                                                                                                             |
| YOLOv8n              | 3.1                | 6.2                           | 37.6                      | -                      | 57.4                      | -                                                                                                             |

</details>

## News

- `2025/03/20`: We release RF-DETR real-time object detection model. **Code and checkpoint are available!**

## Installation

```bash
pip install git+https://github.com/roboflow/rf-detr.git
```

The `rfdetr` package will be distributed soon via PyPI.

</details>

## Prediction

RF-DETR comes out of the box with a checkpoint trained from the Microsoft COCO dataset.

```python
import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase

model = RFDETRBase()

url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"

image = Image.open(io.BytesIO(requests.get(url).content))
detections = model.predict(image, threshold=0.5)

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)

sv.plot_image(annotated_image)
```

![rf-detr-coco-results](https://github.com/user-attachments/assets/969ed869-3044-49a3-a00d-7cae18017325)

## Training

### Dataset structure

RF-DETR expects the dataset to be in COCO format. Divide your dataset into three subdirectories: `train`, `valid`, and `test`. Each subdirectory should contain its own `_annotations.coco.json` file that holds the annotations for that particular split, along with the corresponding image files. Below is an example of the directory structure:

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```

[Roboflow](https://roboflow.com/annotate) allows you to create object detection datasets from scratch or convert existing datasets from formats like YOLO, and then export them in COCO JSON format for training. You can also explore [Roboflow Universe](https://universe.roboflow.com/) to find pre-labeled datasets for a range of use cases.

### Fine-tuning

You can fine-tune RF-DETR from pre-trained COCO checkpoints. By default, the RF-DETR-B checkpoint will be used. To get started quickly, please refer to our fine-tuning Google Colab [notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb).

```python
from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(dataset_dir=<DATASET_PATH>, epochs=10, batch_size=4, grad_accum_steps=4, lr=1e-4)
```

### Result checkpoints

During training, two model checkpoints (the regular weights and an EMA-based set of weights) will be saved in the specified output directory. The EMA (Exponential Moving Average) file is a smoothed version of the model’s weights over time, often yielding better stability and generalization.

### Load and run fine-tuned model

```python
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

detections = model.predict(<IMAGE_PATH>)
```

## License

Both the code and the weights pretrained on the COCO dataset are released under the [Apache 2.0 license](https://github.com/roboflow/r-flow/blob/main/LICENSE).

## Acknowledgements

Our work is built upon [LW-DETR](https://arxiv.org/pdf/2406.03459), [DINOv2](https://arxiv.org/pdf/2304.07193), and [Deformable DETR](https://arxiv.org/pdf/2010.04159). Thanks to their authors for their excellent work!

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@software{rf-detr,
  author = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei},
  license = {Apache-2.0},
  title = {RF-DETR},
  howpublished = {\url{https://github.com/roboflow/rf-detr}},
  year = {2025},
  note = {SOTA Real-Time Object Detection Model}
}
```


## Contribution

We welcome and appreciate all contributions! If you notice any issues or bugs, have questions, or would like to suggest new features, please [open an issue](https://github.com/roboflow/rf-detr/issues/new) or pull request. By sharing your ideas and improvements, you help make RF-DETR better for everyone.
