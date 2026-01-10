from .ftir_frontend import FTIRFrontend
from .positional_encoding import PositionalEncoding
import tensorflow as tf

class FTIREncoder(tf.keras.layers.Layer):
    def __init__(self, d_model=128, num_heads=4, num_layers=2, target_len=200, d_ff=512, dropout=0.1):
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

        # TODO: Feed Forward Layer needed? Add additional batch normalization for each as well. Use "dropout"

        # TODO: Use layer normalization instead of batch normalization? Check if correct

        # Feed-forward networks (one per layer)
        self.ffn = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(d_ff, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ])
            for _ in range(num_layers)
        ]

        self.norms = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]

        # LayerNorm after FFN
        self.norms_ffn = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]

        # Dropout layers
        self.dropouts_attn = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]
        self.dropouts_ffn = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]

    def call(self, x, training=False):
        x = self.frontend(x)  # (batch, target_len, d_model)
        x = self.pos(x)

        # Multi-head self attention layer
        for attn, ffn, norm1, norm2, d1, d2 in zip(
            self.layers,
            self.ffn,
            self.norms,
            self.norms_ffn,
            self.dropouts_attn,
            self.dropouts_ffn,
        ):
            # Self-attention
            attn_out = attn(x, x, training=training)
            attn_out = d1(attn_out, training=training)
            x = norm1(x + attn_out)

            # Feed-forward network
            ffn_out = ffn(x, training=training)
            ffn_out = d2(ffn_out, training=training)
            x = norm2(x + ffn_out)

        return x
