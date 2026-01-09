import tensorflow as tf

class SMILESTokenizer:
    def __init__(self, d_model=128):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<SEP>"]
        self.char2idx = {}
        self.idx2char = {}
        self.d_model = d_model
        self.embedding = None

    def fit(self, smiles_list):
        """
        Build vocabulary from list of SMILES strings
        """
        chars = set()
        for s in smiles_list:
            chars.update(list(s))

        vocab = self.special_tokens + sorted(chars)
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}

        # 🔹 Define embedding layer for linear transformation
        self.embedding = tf.keras.layers.Embedding(
            input_dim=len(vocab),
            output_dim=self.d_model
        )

    def encode(self, smiles):
        """
        Convert SMILES string to list of token IDs
        """
        return (
            [self.char2idx["<SOS>"]]
            + [self.char2idx[c] for c in smiles]
            + [self.char2idx["<EOS>"]]
        )

    def decode(self, tokens):
        """
        Convert list of token IDs back to SMILES string
        """
        chars = []
        for t in tokens:
            c = self.idx2char.get(t, "")
            if c in ("<SOS>", "<EOS>", "<PAD>"):
                continue
            chars.append(c)
        return "".join(chars)

    @property
    def vocab_size(self):
        return len(self.char2idx)

    def token_to_id(self, token):
        return self.char2idx[token]

    def id_to_token(self, idx):
        return self.idx2char[idx]

    def embed(self, token_ids):
        """
        Convert list of token IDs to dense vectors using linear embedding
        token_ids: list[int] or 2D tensor (batch_size, seq_len)
        returns: tensor of shape (batch_size, seq_len, d_model)
        """
        if self.embedding is None:
            raise ValueError("Call fit() first to initialize embedding layer")
        token_ids = tf.convert_to_tensor(token_ids, dtype=tf.int32)
        return self.embedding(token_ids)
