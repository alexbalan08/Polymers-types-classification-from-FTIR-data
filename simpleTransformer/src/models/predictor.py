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
        # Safety wrapper - this method might not even be needed anymore
        if ftir_spectrum.ndim == 1:
            ftir_spectrum = ftir_spectrum.reshape(1, -1)
        return ftir_spectrum.astype(np.float32)

    def predict(self, ftir_spectrum, threshold, debug=False):
        """
        Predict SMILES string from FTIR spectrum
        ftir_spectrum : raw FTIR spectrum
        debug         : print each decoding step
        """
        # Preprocess FTIR spectrum
        x = self.preprocess(ftir_spectrum)

        # Start decoder with SOS token
        decoder_input = [[self.SOS]]
        decodings = [[]]
        probabilities = [[]]
        finished = [False]

        for step in range(self.max_len):
            for i in range(len(decoder_input)):
                if not finished[i]:
                    # Prepare batch inputs
                    y_input = np.array([decoder_input[i]], dtype=np.int32)

                    # Forward pass
                    logits = self.model((x, y_input), training=False).numpy()
                    with np.printoptions(precision=3, suppress=True):
                        print(logits[0, -1])

                    # Pick next token
                    # next_token = int(tf.argmax(logits[0, -1]).numpy())
                    pred_last_token = logits[0, -1]
                    next_tokens = np.where(pred_last_token > threshold)[0]
                    next_tokens = next_tokens[np.argsort(pred_last_token[next_tokens])[::-1]].tolist()
                    next_probs =  pred_last_token[next_tokens].tolist()
                    print(next_tokens)

                    if debug:
                        for next_token, next_prob in zip(next_tokens, next_probs):
                            token_char = self.tokenizer.id_to_token(next_token)
                            print(f"Step {step:02d} | Token ID: {next_token:3d} | Token: {token_char:>3} | Prob: {next_prob:>3}")

                    for next_token, next_prob in zip(next_tokens[1:], next_probs[1:]):
                        decoder_input.append(decoder_input[i].copy())
                        decodings.append(decodings[i].copy())
                        probabilities.append(probabilities[i].copy())
                        finished.append(next_token == self.EOS)

                        # Stop at EOS
                        if not finished[-1]:
                            decodings[-1].append(next_token)
                            decoder_input[-1].append(next_token)
                            probabilities[-1].append(next_prob)

                    finished[i] = next_tokens[0] == self.EOS
                    if not finished[i]:
                        decodings[i].append(next_tokens[0])
                        decoder_input[i].append(next_tokens[0])
                        probabilities[i].append(next_probs[0])


        # Convert token IDs to SMILES string
        smiles = [self.tokenizer.decode(dec) for dec in decodings]
        probs = [np.array(ps).mean() for ps in probabilities]
        return smiles, probs
