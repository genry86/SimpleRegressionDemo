import tensorflow as tf

class EpochTracker(tf.keras.callbacks.Callback):
    def __init__(self, path):
        self.path = path
    def on_epoch_end(self, epoch, logs=None):
        with open(self.path, 'w') as f:
            f.write(str(epoch + 1))