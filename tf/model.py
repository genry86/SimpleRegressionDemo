import tensorflow as tf
from tensorflow.keras import layers, models

def build_center_detector(input_shape=(64, 64, 1)):
    model = models.Sequential([
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),

        layers.Conv2D(128, 3, activation='relu'),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)  # координаты (x, y), без активации (регрессия)
    ])
    return model

if __name__ == '__main__':
    model = build_center_detector()
    print(model.summary())