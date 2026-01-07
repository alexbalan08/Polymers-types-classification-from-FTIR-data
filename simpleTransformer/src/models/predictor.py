import tensorflow as tf
from src.models.inference import FTIRToSMILESGenerator

class FTIRMonomerPredictor:
    """
    High-level class to predict monomer SMILES from FTIR spectra.
    """

    def __init__(self, model, tokenizer, max_len=64):
        """
        model: trained FTIRToSMILESTransformer
        tokenizer: SMILESTokenizer
        max_len: max length of SMILES sequence
        """
        self.generator = FTIRToSMILESGenerator(model, tokenizer, max_len=max_len)
        self.tokenizer = tokenizer

    def predict(self, ftir_input):
        """
        ftir_input: single FTIR spectrum (1D) or batch (2D)
        Returns: SMILES string or list of SMILES strings
        """
        if len(ftir_input.shape) == 1:
            # Single spectrum
            return self.generator.generate(ftir_input)
        elif len(ftir_input.shape) == 2 or len(ftir_input.shape) == 3:
            # Batch of spectra
            return self.generator.batch_generate(ftir_input)
        else:
            raise ValueError("FTIR input must be shape (seq_len) or (batch, seq_len)")

    def predict_multiple(self, ftir_batch, sep="<SEP>"):
        """
        Predict multiple monomer SMILES with optional separator
        ftir_batch: (batch, seq_len)
        sep: string to separate multiple monomers
        Returns: list of SMILES strings
        """
        raw_preds = self.predict(ftir_batch)
        if isinstance(raw_preds, str):
            return raw_preds
        # Ensure <SEP> is preserved
        return [pred.replace("<SEP>", sep) for pred in raw_preds]
