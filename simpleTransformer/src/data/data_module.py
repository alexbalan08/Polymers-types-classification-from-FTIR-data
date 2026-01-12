# Combines FTIR + SMILES, builds (X, Y)
import numpy as np
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

class FTIRToSMILESDataModule:
    def __init__(self, ftir_ds, monomer_map, tokenizer, max_len=200, n_pca=200):
        self.ftir_ds = ftir_ds
        self.monomer_map = monomer_map
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.n_pca = n_pca

    def build(self):
        spectra = []
        plastics = []
        for spectrum, plastic in zip(self.ftir_ds.get_spectra(), self.ftir_ds.get_plastics()):
            if "sealing_ring" in plastic:
                continue
            smiles_list = self.monomer_map.get_smiles_for_plastic(plastic)
            if not smiles_list:
                continue
            spectra.append(spectrum)
            plastics.append(plastic)

        return np.asarray(spectra), np.asarray(plastics)

    def transform_data(self, spectra, plastics, pca_objects=None):
        X = []
        Y = []
        for spectrum, plastic in zip(spectra, plastics):
            smiles_list = self.monomer_map.get_smiles_for_plastic(plastic)
            if not smiles_list:
                continue

            for smiles in smiles_list:
                X.append(spectrum)
                Y.append(smiles)

        # Fit tokenizer on ALL targets
        self.tokenizer.fit(Y)

        # Encode + pad Y
        Y = np.asarray([self._pad(self.tokenizer.encode(y)) for y in Y])
        X = np.asarray(X)

        # Convert X → PCA
        if pca_objects is None:
            scaler = StandardScaler()
            pca = PCA(n_components=self.n_pca)
            X_scaled = scaler.fit_transform(X)
            X_pca = pca.fit_transform(X_scaled)
            return X_pca, Y, (scaler, pca)

        (scaler, pca) = pca_objects
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        return X_pca, Y

    def _pad(self, seq):
        seq = seq[: self.max_len]
        return seq + [0] * (self.max_len - len(seq))
