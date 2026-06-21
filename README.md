# Image Classification using Convolutional Neural Network

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

## About The Project

This project is an image classification system built with deep learning. It uses TensorFlow/Keras and MobileNetV2 transfer learning to classify images into custom categories. The model is trained using user-provided image folders and can later predict the class of new images.

The project includes dataset structure, training code, prediction code, model saving, class label saving, and a clean setup for GitHub.

## Tech Stack

![Tech Stack](https://skillicons.dev/icons?i=python,tensorflow,github)

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- MobileNetV2 Transfer Learning

## Features

- Trains an image classification model from custom image folders
- Uses MobileNetV2 pretrained weights for better accuracy
- Applies data augmentation to improve generalization
- Saves the best model automatically
- Saves class names for correct prediction output
- Includes a prediction script for testing new images

## Project Structure

```text
.
|-- data/
|   |-- train/
|   `-- validation/
|-- models/
|-- src/
|   |-- train.py
|   `-- predict.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Dataset Format

Add your images inside class folders. The class folder names must match in both `train` and `validation`.

```text
data/
|-- train/
|   |-- class_1/
|   `-- class_2/
`-- validation/
    |-- class_1/
    `-- class_2/
```

Example:

```text
data/train/cat/image1.jpg
data/train/dog/image2.jpg
data/validation/cat/image3.jpg
data/validation/dog/image4.jpg
```

## Installation

```bash
pip install -r requirements.txt
```

## Train The Model

```bash
python src/train.py
```

After training, the best model will be saved here:

```text
models/best_image_classifier.keras
```

The final fine-tuned model will be saved here:

```text
models/final_image_classifier.keras
```

## Predict New Images

```bash
python src/predict.py path/to/image.jpg
```

## Output

The prediction script displays:

```text
Prediction: class_name
Confidence: 95.00%
```

## Notes

- Add your dataset before training.
- Keep the same class names in training and validation folders.
- The first training run may download MobileNetV2 pretrained weights.
- You can adjust image size, batch size, and epochs in `src/train.py`.

## Author

Created by Kaviarasan M.
