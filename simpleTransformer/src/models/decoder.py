import tensorflow as tf
from .positional_encoding import PositionalEncoding

class SMILESDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=512, dropout=0.1):
        super().__init__()

        self.embed = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        self.self_attn = []
        self.cross_attn = []

        # Feed-forward networks (one per layer)
        self.ffn = []

        self.norms1 = []
        self.norms2 = []

        # LayerNorm for FFN
        self.norms3 = []

        # Dropout layers
        self.dropouts1 = []
        self.dropouts2 = []
        self.dropouts3 = []

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

            # Standard Transformer Feed-Forward Network
            self.ffn.append(
                tf.keras.Sequential([
                    tf.keras.layers.Dense(d_ff, activation="relu"),
                    tf.keras.layers.Dense(d_model),
                ])
            )

            self.norms1.append(tf.keras.layers.LayerNormalization())
            self.norms2.append(tf.keras.layers.LayerNormalization())
            self.norms3.append(tf.keras.layers.LayerNormalization())

            self.dropouts1.append(tf.keras.layers.Dropout(dropout))
            self.dropouts2.append(tf.keras.layers.Dropout(dropout))
            self.dropouts3.append(tf.keras.layers.Dropout(dropout))

        self.out = tf.keras.layers.Dense(vocab_size)

    # TODO: Padding Mask? Logic for ignoring the padding

    def call(self, y, enc, training=False):
        # y: (batch, target_len)
        y = self.embed(y)
        y = self.pos(y)

        for sa, ca, ffn, n1, n2, n3, d1, d2, d3 in zip(
            self.self_attn,
            self.cross_attn,
            self.ffn,
            self.norms1,
            self.norms2,
            self.norms3,
            self.dropouts1,
            self.dropouts2,
            self.dropouts3,
        ):
            # Masked self-attention
            sa_out = sa(y, y, use_causal_mask=True, training=training)
            sa_out = d1(sa_out, training=training)
            y = n1(y + sa_out)

            # Cross-attention (encoder-decoder)
            ca_out = ca(y, enc, training=training)
            ca_out = d2(ca_out, training=training)
            y = n2(y + ca_out)

            # Feed-forward network
            ffn_out = ffn(y, training=training)
            ffn_out = d3(ffn_out, training=training)
            y = n3(y + ffn_out)

        return self.out(y)
