# LangXAI: XAI Explanations in Human Language
![img.png](overview.png)
## Installation
- Install dependencies
```pip install -r requirements.txt```
- Setup OpenAI API key in `config.py`
- Download a fine-tuned DeepLabv3-ResNet segmentation models into `models` folder:
  - [DeepLabv3-ResNet50](https://drive.google.com/file/d/1NbEGJcCzKJDAKiniiHwRiXFSTDmC6GJg/view?usp=drive_link)
  - [DeepLabv3-ResNet101](https://drive.google.com/file/d/1KpW5ilZbwkuwtqw1TqPbOuSvoHPJ9w3i/view?usp=drive_link)
## Usage
- Run LangXAI platform:
```python app.py```
- Choose tasks: Semantic Segmentation, Classification, Object Detection.
## Benchmark
- GPT4-Vision

| Task                   | BLEU   | METEOR | ROUGE-L | BERTScore |
|------------------------|--------|--------|---------|-----------|
| Classification         | 0.2971 | 0.5122 | 0.5196  | 0.9341    |
| Semantic Segmentation  | 0.2552 | 0.4741 | 0.4714  | 0.8594    |
| Object Detection       | 0.2754 | 0.4904 | 0.4911  | 0.9093    |

## BibTeX
- If you use LangXAI, please cite the following paper:
```

```