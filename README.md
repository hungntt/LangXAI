# LangXAI: XAI Explanations in Human Language
## Installation
```pip install -r requirements.txt```
## Usage
- Run LangXAI platform:
```python app.py```
- Choose tasks: Segmentation, Classification, Object Detection.
## Benchmark
- GPT-Vision
- KOSMOS-2
## Tasks-to-be-done
### Classification
- [x] Download ImageNet dataset for training/validation and ImageNetV2 for test.
- [ ] Evaluate the performance of LangXAI on ImageNet dataset.
### Segmentation
- [x] Download TTPLA dataset in 400x400 resolution.
- [ ] Evaluate the performance of LangXAI on TTPLA dataset.
### Object Detection
- [ ] Implement object detection models: FasterRCNN, YOLOX.
- [ ] Implement XAI methods: D-RISE, D-CLOSE.
- [ ] Download COCO dataset.
- [ ] Evaluate the performance of LangXAI on COCO dataset.