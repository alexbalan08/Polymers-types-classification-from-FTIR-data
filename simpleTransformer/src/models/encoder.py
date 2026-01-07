import tensorflow as tf
from .positional_encoding import PositionalEncoding

class FTIREncoder(tf.keras.layers.Layer):
    def __init__(self, d_model=128, num_heads=4, num_layers=2):
        super().__init__()

        self.proj = tf.keras.layers.Dense(d_model)
        self.pos = PositionalEncoding(d_model)

        self.layers = [
            tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads
            )
            for _ in range(num_layers)
        ]

        self.norms = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]

    def call(self, x, training=False):
        # x: (batch, seq_len, 1)
        x = self.proj(x)
        x = self.pos(x)

        for attn, norm in zip(self.layers, self.norms):
            attn_out = attn(x, x, training=training)
            x = norm(x + attn_out)

        return x
