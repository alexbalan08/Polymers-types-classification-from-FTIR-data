# Combines FTIR + SMILES, builds (X, Y)
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FTIRToSMILESDataModule:
    def __init__(self, ftir_ds, monomer_map, tokenizer, max_len=200, n_pca=200):
        self.ftir_ds = ftir_ds
        self.monomer_map = monomer_map
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca)

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

            for smiles in smiles_list:
                #target = "<SEP>".join(smiles_list)
                X.append(spectrum)
                Y.append(smiles)

            # TODO: Predict one monomer at a time! Loop over/ append every monomer per spectra

        # Fit tokenizer
        self.tokenizer.fit(Y)

        # Encode targets
        Y_encoded = [
            self._pad(self.tokenizer.encode(y)) for y in Y
        ]

        X = tf.constant(X)
        X_np = X.numpy()

        X_scaled = self.scaler.fit_transform(X_np)
        X_pca = self.pca.fit_transform(X_scaled)

        Y = tf.constant(Y_encoded)

        return X_pca, Y

    def _pad(self, seq):
        seq = seq[: self.max_len]
        return seq + [0] * (self.max_len - len(seq))
