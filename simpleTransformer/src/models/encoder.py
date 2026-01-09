from .ftir_frontend import FTIRFrontend
from .positional_encoding import PositionalEncoding
import tensorflow as tf

class FTIREncoder(tf.keras.layers.Layer):
    def __init__(self, d_model=128, num_heads=4, num_layers=2, target_len=200):
        super().__init__()
        self.frontend = FTIRFrontend(d_model=d_model, target_len=target_len)
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
        x = self.frontend(x)  # (batch, target_len, d_model)
        x = self.pos(x)

        # Multi-head self attention layer
        for attn, norm in zip(self.layers, self.norms):
            attn_out = attn(x, x, training=training)
            x = norm(x + attn_out)

        return x
