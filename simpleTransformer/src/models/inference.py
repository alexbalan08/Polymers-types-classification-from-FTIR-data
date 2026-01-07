import tensorflow as tf

class FTIRToSMILESGenerator:
    def __init__(self, model, tokenizer, max_len=64):
        """
        model: trained FTIRToSMILESTransformer
        tokenizer: SMILESTokenizer
        max_len: maximum length of SMILES output
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

    def generate(self, ftir_input):
        """
        ftir_input: numpy array or tf.Tensor of shape (seq_len,) or (1, seq_len, 1)
        Returns: predicted SMILES string
        """
        # Ensure correct shape
        if len(ftir_input.shape) == 1:
            ftir_input = ftir_input[None, :, None]  # (1, seq_len, 1)
        elif len(ftir_input.shape) == 2:
            ftir_input = ftir_input[:, :, None]  # (batch, seq_len, 1)

        output = [self.tokenizer.char2idx["<SOS>"]]

        for _ in range(self.max_len):
            y_in = tf.constant([output])  # (1, current_len)
            logits = self.model((ftir_input, y_in), training=False)
            next_token = tf.argmax(logits[0, -1]).numpy()
            if next_token == self.tokenizer.char2idx["<EOS>"]:
                break
            output.append(next_token)

        # Decode token sequence to SMILES
        return self.tokenizer.decode(output)

    def batch_generate(self, ftir_batch):
        """
        ftir_batch: numpy array of shape (batch, seq_len) or (batch, seq_len, 1)
        Returns: list of SMILES strings
        """
        results = []
        for ftir_input in ftir_batch:
            smiles = self.generate(ftir_input)
            results.append(smiles)
        return results
