# Combines FTIR + SMILES, builds (X, Y)
import tensorflow as tf

class FTIRToSMILESDataModule:
    def __init__(self, ftir_ds, monomer_map, tokenizer, max_len=256):
        self.ftir_ds = ftir_ds
        self.monomer_map = monomer_map
        self.tokenizer = tokenizer
        self.max_len = max_len

    def build(self):
        X = []
        Y = []

        for spectrum, plastic in zip(
            self.ftir_ds.get_spectra(),
            self.ftir_ds.get_plastics()
        ):
            smiles_list = self.monomer_map.get_smiles_for_plastic(plastic)
            if not smiles_list:
                continue

            target = "<SEP>".join(smiles_list)
            X.append(spectrum)
            Y.append(target)

        # Fit tokenizer
        self.tokenizer.fit(Y)

        # Encode targets
        Y_encoded = [
            self._pad(self.tokenizer.encode(y)) for y in Y
        ]

        X = tf.expand_dims(tf.constant(X), -1)
        Y = tf.constant(Y_encoded)

        return X, Y

    def _pad(self, seq):
        seq = seq[: self.max_len]
        return seq + [0] * (self.max_len - len(seq))
