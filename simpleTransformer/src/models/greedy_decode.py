import tensorflow as tf

def greedy_decode(model, ftir_input, tokenizer, max_len=64):
    """
    ftir_input: (1, seq_len, 1) tensor
    Returns: predicted SMILES string
    """
    output = [tokenizer.char2idx["<SOS>"]]

    for _ in range(max_len):
        y_in = tf.constant([output])  # (1, current_len)
        logits = model((ftir_input, y_in), training=False)
        next_token = tf.argmax(logits[0, -1]).numpy()
        if next_token == tokenizer.char2idx["<EOS>"]:
            break
        output.append(next_token)

    return tokenizer.decode(output)
