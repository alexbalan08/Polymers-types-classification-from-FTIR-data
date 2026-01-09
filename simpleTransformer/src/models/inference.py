import time
import tensorflow as tf

class FTIRToSMILESGenerator:
    def __init__(self, model, tokenizer, max_len=64):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

    def generate(self, ftir_input, debug=True):
        # Ensure shape: (1, seq_len, 1)
        if len(ftir_input.shape) == 1:
            ftir_input = ftir_input[None, :, None]
        elif len(ftir_input.shape) == 2:
            ftir_input = ftir_input[:, :, None]

        sos = self.tokenizer.char2idx["<SOS>"]
        eos = self.tokenizer.char2idx["<EOS>"]

        output = [sos]

        print("\n🔍 DEBUG: Starting autoregressive decoding")
        print("SOS token:", sos)

        for step in range(self.max_len):
            start = time.time()

            y_in = tf.expand_dims(
                tf.constant(output, dtype=tf.int32), axis=0
            )

            logits = self.model((ftir_input, y_in), training=False)

            next_token = tf.argmax(
                logits[:, -1, :], axis=-1, output_type=tf.int32
            ).numpy()[0]

            output.append(next_token)

            decoded_token = (
                self.tokenizer.idx2char[next_token]
                if next_token in self.tokenizer.idx2char
                else "<?>"
            )

            elapsed = (time.time() - start) * 1000

            print(
                f"Step {step:02d} | "
                f"Token ID: {next_token:4d} | "
                f"Token: {decoded_token:>6s} | "
                f"Len: {len(output):2d} | "
                f"{elapsed:6.1f} ms"
            )

            if next_token == eos:
                print("🛑 EOS reached, stopping")
                break

        print("✅ Decoding finished\n")

        return self.tokenizer.decode(output)
