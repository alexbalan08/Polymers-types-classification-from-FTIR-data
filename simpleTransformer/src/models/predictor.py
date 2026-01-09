# src/models/predictor.py
import numpy as np
import tensorflow as tf
import joblib

class FTIRMonomerPredictor:
    def __init__(self, model, tokenizer, scaler_path, pca_path, max_len=64):
        """
        model       : trained FTIRToSMILESTransformer
        tokenizer   : SMILESTokenizer with token_to_id() and id_to_token() methods
        scaler_path : path to StandardScaler
        pca_path    : path to PCA
        max_len     : maximum SMILES sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Load preprocessing objects
        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)

        # Use tokenizer methods for SOS/EOS
        self.SOS = tokenizer.token_to_id("<SOS>")
        self.EOS = tokenizer.token_to_id("<EOS>")

    def preprocess(self, ftir_spectrum):
        """
        Scale and reduce FTIR spectrum to match training input shape
        ftir_spectrum : (1761, ) or (1761, 1)
        Returns : (1, 256) float32
        """
        # Flatten if needed
        if ftir_spectrum.ndim == 2 and ftir_spectrum.shape[1] == 1:
            ftir_spectrum = ftir_spectrum.flatten()

        # Scale
        ftir_scaled = self.scaler.transform([ftir_spectrum])

        # PCA
        ftir_pca = self.pca.transform(ftir_scaled)

        return ftir_pca.astype(np.float32)

    def predict(self, ftir_spectrum, debug=False):
        """
        Predict SMILES string from FTIR spectrum
        ftir_spectrum : raw FTIR spectrum
        debug         : print each decoding step
        """
        # Preprocess FTIR spectrum
        x = self.preprocess(ftir_spectrum)

        # Start decoder with SOS token
        decoder_input = [self.SOS]
        decoded = []

        for step in range(self.max_len):
            # Prepare batch inputs
            y_input = np.array([decoder_input], dtype=np.int32)

            # Forward pass
            logits = self.model((x, y_input), training=False)

            # Pick next token
            next_token = int(tf.argmax(logits[0, -1]).numpy())

            if debug:
                token_char = self.tokenizer.id_to_token(next_token)
                print(f"Step {step:02d} | Token ID: {next_token:3d} | Token: {token_char:>3}")

            # Stop at EOS
            if next_token == self.EOS:
                break

            decoded.append(next_token)
            decoder_input.append(next_token)

        # Convert token IDs to SMILES string
        smiles = self.tokenizer.decode(decoded)
        return smiles
