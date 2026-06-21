# Image Classification using Convolutional Neural Network

This project trains an accurate image classifier using TensorFlow/Keras, data augmentation, and MobileNetV2 transfer learning.

## Project Structure

```text
.
├── data/
│   ├── train/
│   └── validation/
├── models/
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Dataset Format

Place your images inside class folders like this:

```text
data/
├── train/
│   ├── class_1/
│   └── class_2/
└── validation/
    ├── class_1/
    └── class_2/
```

Example:

```text
data/train/cat/image1.jpg
data/train/dog/image2.jpg
data/validation/cat/image3.jpg
data/validation/dog/image4.jpg
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train.py
```

The best trained model will be saved in:

```text
models/best_image_classifier.keras
```

A final fine-tuned model will also be saved as:

```text
models/final_image_classifier.keras
```

## Predict an Image

```bash
python src/predict.py path/to/image.jpg
```

## Notes

- Add your own image dataset before training.
- Keep class folder names the same in both `train` and `validation`.
- You can change image size, batch size, and epochs in `src/train.py`.
- The first training run may download MobileNetV2 pretrained weights.
