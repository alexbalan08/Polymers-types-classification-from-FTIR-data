import tensorflow as tf
from .positional_encoding import PositionalEncoding

class SMILESDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2):
        super().__init__()

        self.embed = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        self.self_attn = []
        self.cross_attn = []
        self.norms1 = []
        self.norms2 = []

        for _ in range(num_layers):
            self.self_attn.append(
                tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=d_model // num_heads
                )
            )
            self.cross_attn.append(
                tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=d_model // num_heads
                )
            )
            self.norms1.append(tf.keras.layers.LayerNormalization())
            self.norms2.append(tf.keras.layers.LayerNormalization())

        self.out = tf.keras.layers.Dense(vocab_size)

    def call(self, y, enc, training=False):
        # y: (batch, target_len)
        y = self.embed(y)
        y = self.pos(y)

        for sa, ca, n1, n2 in zip(
            self.self_attn, self.cross_attn, self.norms1, self.norms2
        ):
            sa_out = sa(y, y, use_causal_mask=True, training=training)
            y = n1(y + sa_out)

            ca_out = ca(y, enc, training=training)
            y = n2(y + ca_out)

        return self.out(y)
