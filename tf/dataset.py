import os
import json
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def load_paths_and_labels(image_dir, json_path):
    with open(json_path, 'r') as f:
        coord_dict = json.load(f)

    image_paths, labels = [], []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and fname in coord_dict:
            image_paths.append(os.path.join(image_dir, fname))
            labels.append(coord_dict[fname])
    return image_paths, labels

def create_tf_dataset(image_paths, labels, img_size=(64, 64), batch_size=64, shuffle=True):
    image_paths = tf.convert_to_tensor(image_paths)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(len(image_paths))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def get_dataset(image_dir, json_path, batch_size=128, img_size=(64, 64)):
    image_paths, labels = load_paths_and_labels(image_dir, json_path)

    # Split train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.1, random_state=42
    )

    train_ds = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, img_size=img_size)
    val_ds   = create_tf_dataset(val_paths, val_labels, shuffle=False)

    return train_ds, val_ds

if __name__ == '__main__':
    train_dir = 'dataset/train'
    json_path = 'dataset/coords.json'
    batch_size = 128
    img_size = (64, 64)

    train_ds, val_ds = get_dataset(train_dir, json_path, batch_size=batch_size, img_size=img_size)
    print(len(train_ds), len(val_ds))