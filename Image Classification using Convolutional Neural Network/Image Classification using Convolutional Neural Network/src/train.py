from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "train"
VALIDATION_DIR = BASE_DIR / "data" / "validation"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_image_classifier.keras"
FINAL_MODEL_PATH = MODEL_DIR / "final_image_classifier.keras"
LABELS_PATH = MODEL_DIR / "class_names.txt"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 5
SEED = 42


def validate_dataset():
    if not TRAIN_DIR.exists() or not VALIDATION_DIR.exists():
        raise FileNotFoundError(
            "Dataset folders are missing. Add images to data/train and data/validation."
        )

    train_classes = sorted(path.name for path in TRAIN_DIR.iterdir() if path.is_dir())
    validation_classes = sorted(path.name for path in VALIDATION_DIR.iterdir() if path.is_dir())

    if not train_classes:
        raise ValueError("No class folders found in data/train.")

    if train_classes != validation_classes:
        raise ValueError(
            "Train and validation class folders must match exactly. "
            f"Train: {train_classes}, Validation: {validation_classes}"
        )


def load_datasets():
    validate_dataset()

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_dataset.class_names
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, class_names


def build_model(num_classes):
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def get_callbacks():
    MODEL_DIR.mkdir(exist_ok=True)
    return [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def main():
    train_dataset, validation_dataset, class_names = load_datasets()
    model, base_model = build_model(num_classes=len(class_names))

    model.summary()
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=get_callbacks(),
    )

    model.save(FINAL_MODEL_PATH)
    LABELS_PATH.write_text("\n".join(class_names), encoding="utf-8")

    print(f"Best model saved to {MODEL_PATH}")
    print(f"Final model saved to {FINAL_MODEL_PATH}")
    print(f"Class names saved to {LABELS_PATH}")


if __name__ == "__main__":
    main()
