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

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca)

        # Stored after build
        self.X = None
        self.Y = None
        self.plastic_labels = None
        self.plastic_names_used = []

    def build(self):
        X = []
        Y = []
        plastic_names = []

        for spectrum, plastic in zip(
            self.ftir_ds.get_spectra(),
            self.ftir_ds.get_plastics()
        ):
            smiles_list = self.monomer_map.get_smiles_for_plastic(plastic)
            if not smiles_list:
                continue

            for smiles in smiles_list:
                X.append(spectrum)
                Y.append(smiles)
                plastic_names.append(plastic)

        # Fit tokenizer on ALL targets
        self.tokenizer.fit(Y)

        # Encode + pad Y
        Y_encoded = [self._pad(self.tokenizer.encode(y)) for y in Y]

        # Convert X → PCA
        X_np = np.asarray(X)
        X_scaled = self.scaler.fit_transform(X_np)
        X_pca = self.pca.fit_transform(X_scaled)

        # Plastic labels for stratification
        plastic_labels = np.array([
            self.monomer_map.get_plastic_id(p) for p in plastic_names
        ])

        # Store internally
        self.X = X_pca
        self.Y = np.asarray(Y_encoded)
        self.plastic_labels = plastic_labels
        self.plastic_names_used = plastic_names

        return self.X, self.Y


            # TODO: Predict one monomer at a time! Loop over/ append every monomer per spectra

    def get_stratified_folds(self, n_splits=3, shuffle=True, random_state=0):
        """
        EXACT equivalent of:

        StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        """
        if self.X is None or self.Y is None:
            raise RuntimeError("Call build_full_dataset() first")

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

        folds = []
        for fold_id, (train_idx, val_idx) in enumerate(
                skf.split(self.X, self.plastic_labels), start=1
        ):
            print(
                f"Fold {fold_id}: "
                f"train={len(train_idx)}, val={len(val_idx)}"
            )
            folds.append((train_idx, val_idx))

        return folds

    def _pad(self, seq):
        seq = seq[: self.max_len]
        return seq + [0] * (self.max_len - len(seq))
