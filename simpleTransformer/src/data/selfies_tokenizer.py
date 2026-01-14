import re

# Matches full SELFIES tokens like [C], [=C], [Branch1_1], [Ring2], etc.
SELFIES_TOKEN_PATTERN = re.compile(r"(\[[^\[\]]+\])")


def tokenize_selfies(selfies):
    tokens = SELFIES_TOKEN_PATTERN.findall(selfies)
    if "".join(tokens) != selfies:
        raise ValueError(f"Failed to tokenize SELFIES: {selfies}")
    return tokens


class SELFIESTokenizer:
    def __init__(self):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.token2idx = {}
        self.idx2token = {}

    def fit(self, selfies_list):
        vocab = set()
        for s in selfies_list:
            vocab.update(tokenize_selfies(s))

        vocab = self.special_tokens + sorted(vocab)
        self.token2idx = {t: i for i, t in enumerate(vocab)}
        self.idx2token = {i: t for t, i in self.token2idx.items()}

    def encode(self, selfies):
        tokens = tokenize_selfies(selfies)
        return (
            [self.token2idx["<SOS>"]]
            + [self.token2idx.get(t, self.token2idx["<UNK>"]) for t in tokens]
            + [self.token2idx["<EOS>"]]
        )

    def decode(self, indices):
        tokens = []
        for i in indices:
            t = self.idx2token.get(i, "")
            if t in self.special_tokens:
                continue
            tokens.append(t)
        return "".join(tokens)

    @property
    def vocab_size(self):
        return len(self.token2idx)

    def token_to_id(self, token):
        return self.token2idx.get(token, self.token2idx["<UNK>"])

    def id_to_token(self, idx):
        return self.idx2token.get(idx, "<UNK>")
