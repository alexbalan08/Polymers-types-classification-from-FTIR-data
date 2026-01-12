import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from src.models.encoder import FTIREncoder
from src.models.decoder import SMILESDecoder
from src.models.transformer import FTIRToSMILESTransformer

def train_cross_validation(
    X, Y, plastic_labels, tokenizer, mapping,
    n_splits=3,
    batch_size=10,
    epochs=1,
    model_save_path="checkpoints/ftir_transformer.weights.h5",
    # Added hyperparameters
    d_model=32,
    num_heads=4,
    num_layers=2,
    drop_rate=0.1,
    max_len=48,
    learning_rate=1e-3
):
    """
    Performs stratified k-fold training on FTIR -> SMILES dataset.

    Returns:
        model: last trained model
        all_histories_df: pd.DataFrame with loss history
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    loss_histories = []

    for fold_id, (train_idx, val_idx) in enumerate(cv.split(X, plastic_labels), start=1):
        print(f"\nFold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")

        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # Build model
        encoder = FTIREncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            target_len=max_len,
            dropout=drop_rate
        )
        decoder = SMILESDecoder(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=drop_rate
        )

        model = FTIRToSMILESTransformer(encoder, decoder)

        # Prepare datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ((X_train, Y_train[:, :-1]), Y_train[:, 1:])
        ).shuffle(1024).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            ((X_val, Y_val[:, :-1]), Y_val[:, 1:])
        ).batch(batch_size)

        # Compile model
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            ignore_class=0
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss_fn)

        # Train model
        print(f"Training fold {fold_id} ({X_train.shape[0]} train / {X_val.shape[0]} val)")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=2
        )

        # Convert history to DataFrame
        hist_df = pd.DataFrame(history.history)
        hist_df['fold'] = fold_id
        hist_df['epoch'] = range(1, len(hist_df) + 1)
        loss_histories.append(hist_df)

    # Save model weights (last fold)
    model.save_weights(model_save_path)
    print(f"Model weights saved to {model_save_path}")

    # Combine histories
    all_histories_df = pd.concat(loss_histories, ignore_index=True)
    history_csv_path = os.path.join(os.path.dirname(model_save_path), "training_history.csv")
    all_histories_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")

    return model, all_histories_df
