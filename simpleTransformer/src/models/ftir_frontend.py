import tensorflow as tf

class FTIRFrontend(tf.keras.layers.Layer):
    def __init__(self, d_model=128, target_len=200):
        super().__init__()
        self.d_model = d_model
        self.target_len = target_len

        self.conv_stack = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 7, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv1D(64, 5, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv1D(128, 5, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv1D(d_model, 3, strides=2, padding="same", activation="relu"),
        ])

        # Adaptive pooling replacement
        self.pool = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, x):
        # x: (batch, seq_len, channels)
        if len(x.shape) == 2:
            # PCA input: reshape to (batch, seq_len, 1)
            x = tf.expand_dims(x, -1)

        x = self.conv_stack(x)
        x = tf.expand_dims(self.pool(x), 1)  # (batch, 1, d_model)
        x = tf.repeat(x, self.target_len, axis=1)  # (batch, target_len, d_model)
        return x
