# src/data/smiles_tokenizer.py
class SMILESTokenizer:
    def __init__(self):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<SEP>"]
        self.char2idx = {}
        self.idx2char = {}

    def fit(self, smiles_list):
        chars = set()
        for s in smiles_list:
            chars.update(list(s))

        vocab = self.special_tokens + sorted(chars)
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def encode(self, smiles):
        return (
            [self.char2idx["<SOS>"]]
            + [self.char2idx[c] for c in smiles]
            + [self.char2idx["<EOS>"]]
        )

    def decode(self, tokens):
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
