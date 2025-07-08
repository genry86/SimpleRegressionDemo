import os
import tensorflow as tf
from dataset import get_dataset
from model import build_center_detector
from EpochTracker import EpochTracker

train_dir = '../dataset/train'
json_path = '../dataset/coords.json'
model_dir = '../training_model'
os.makedirs(model_dir, exist_ok=True)

batch_size = 128
epochs = 10
img_size = (64, 64)
lr_start = 0.001
epoch_path = os.path.join(model_dir, 'last_epoch.txt')
last_model_path = os.path.join(model_dir, 'last.keras')

train_ds, val_ds = get_dataset(train_dir, json_path, batch_size=batch_size, img_size=img_size)

initial_epoch = 0
if os.path.exists(last_model_path):
    print("🔁 Load last checkpoint model...")
    model = tf.keras.models.load_model(last_model_path)
    initial_epoch = int(open(epoch_path).read()) if os.path.exists(epoch_path) else 0
else:
    print("🆕 Create new model...")
    model = build_center_detector()
model = build_center_detector()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_start),
    loss='mse',
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

# === Callbacks ===
checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(model_dir, 'best.keras'),
    save_best_only=True,
    monitor='val_loss',
    # monitor='val_mean_absolute_error',
    mode='min',
    verbose=1
)

checkpoint_last = tf.keras.callbacks.ModelCheckpoint(
    filepath=last_model_path,
    save_best_only=False
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    min_delta=0.01,
    verbose=1
)

epoch_tracker = EpochTracker(epoch_path)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_best, checkpoint_last, lr_scheduler, epoch_tracker]
)

# model.save(os.path.join(model_dir, 'final_model'))