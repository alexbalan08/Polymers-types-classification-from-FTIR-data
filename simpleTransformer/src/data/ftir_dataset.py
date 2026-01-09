#Loads FTIR CSV → numpy array (spectra)

import pandas as pd
import numpy as np

class FTIRDataset:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.spectra = None
        self.plastics = None

    def load(self):
        self.df = pd.read_csv(self.csv_path)

        # FTIR columns are numeric wavenumbers
        ftir_cols = [c for c in self.df.columns if self._is_float(c)]
        self.spectra = self.df[ftir_cols].values.astype(np.float32)

        # Normalize spectra
        self.spectra = (self.spectra - self.spectra.mean(axis=1, keepdims=True)) / (
            self.spectra.std(axis=1, keepdims=True) + 1e-6
        )

        self.plastics = self.df["Substance"].astype(str).values

    def _is_float(self, s):
        try:
            float(s)
            return True
        except:
            return False

    def get_spectra(self):
        return self.spectra

    def get_plastics(self):
        return self.plastics
