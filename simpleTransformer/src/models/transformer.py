import tensorflow as tf

class FTIRToSMILESTransformer(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        if isinstance(inputs, (tuple, list)):
            x, y = inputs
        else:
            raise ValueError("Input to model must be a tuple: (FTIR, target_tokens)")

        enc = self.encoder(x, training=training)
        logits = self.decoder(y, enc, training=training)
        return logits
