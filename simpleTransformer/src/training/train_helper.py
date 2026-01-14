import os
import pickle

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.models.encoder import FTIREncoder
from src.models.decoder import SMILESDecoder
from src.models.transformer import FTIRToSMILESTransformer

def train_cross_validation(
    X, Y, tokenizer, data_module, checkpoint_dir,
    n_splits=3,
    batch_size=10,
    epochs=1,
    # Added hyperparameters
    d_model=32,
    num_heads=4,
    num_layers=2,
    drop_rate=0.1,
    max_len=48,
    learning_rate=1e-3,
    do_pretraining=False,
    X_fp=None,
    Y_fp=None,
    pretrain_epochs=3
):
    """
    Performs stratified k-fold training on FTIR -> SMILES dataset.

    Returns:
        model: last trained model
        all_histories_df: pd.DataFrame with loss history
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    loss_histories = []

    for fold_id, (train_idx, val_idx) in enumerate(cv.split(X, Y), start=1):
        fold_dir = os.path.join(checkpoint_dir, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\nFold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")

        # Data transformations
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val_plastic = Y[train_idx], Y[val_idx]
        X_train, Y_train, (scaler, pca, kmeans) = data_module.transform_data(X_train, Y_train)
        X_val, Y_val = data_module.transform_data(X_val, Y_val_plastic, pca_objects=(scaler, pca, kmeans))

        # Save scaler
        scaler_path = os.path.join(fold_dir, f"ftir_scaler.save")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Save PCA
        pca_path = os.path.join(fold_dir, "ftir_pca.save")
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)

        print(f"Scaler saved to {scaler_path}")
        print(f"PCA saved to {pca_path}")

        # Fingerprint Pretraining
        if do_pretraining:
            print(f"Fold {fold_id}: Starting fingerprint pretraining ({len(X_fp)} samples)")

            # Build encoder and decoder for pretraining
            pretrain_encoder = FTIREncoder(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=drop_rate,
                is_fp=True
            )
            decoder = SMILESDecoder(
                vocab_size=tokenizer.vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=drop_rate
            )

            pretrain_model = FTIRToSMILESTransformer(pretrain_encoder, decoder)

            # Encode + pad Y_fp
            Y_fp_encoded = np.asarray([data_module._pad(tokenizer.encode(y)) for y in Y_fp])

            # Split
            X_fp_train, X_fp_test, Y_fp_train, Y_fp_test = train_test_split(
                X_fp, Y_fp_encoded,
                test_size=0.05,  # fraction of data for test set
                random_state=67,  # for reproducibility
                shuffle=True  # shuffle before splitting
            )

            # Create tf.data.Dataset
            fp_dataset = tf.data.Dataset.from_tensor_slices(
                ((X_fp_train, Y_fp_train[:, :-1]), Y_fp_train[:, 1:])
            ).shuffle(1024).batch(batch_size)
            fp_val_dataset = tf.data.Dataset.from_tensor_slices(
                ((X_fp_test, Y_fp_test[:, :-1]), Y_fp_test[:, 1:])
            ).batch(batch_size)

            # Compile model
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, ignore_class=0
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            pretrain_model.compile(optimizer=optimizer, loss=loss_fn)

            # Train
            history = pretrain_model.fit(
                fp_dataset,
                validation_data=fp_val_dataset,
                epochs=pretrain_epochs,
                verbose=1
            )
            print("Fingerprint pretraining completed!")

            # Convert history to DataFrame
            hist_df = pd.DataFrame(history.history)
            hist_df['fold'] = f"pre{fold_id}"
            hist_df['epoch'] = range(1, len(hist_df) + 1)
            loss_histories.append(hist_df)

        # Build model
        encoder = FTIREncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=drop_rate,
            is_fp=False
        )

        if not do_pretraining:
            decoder = SMILESDecoder(
                vocab_size=tokenizer.vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=drop_rate
            )
        else:
            # freeze decoder weights
            pass

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
            verbose=1
        )

        # Convert history to DataFrame
        hist_df = pd.DataFrame(history.history)
        hist_df['fold'] = fold_id
        hist_df['epoch'] = range(1, len(hist_df) + 1)
        loss_histories.append(hist_df)

        # Save model weights
        model_save_path = os.path.join(fold_dir, "ftir_transformer.weights.h5")
        model.save_weights(model_save_path)
        print(f"Model weights saved to {model_save_path}")

    # Combine histories
    all_histories_df = pd.concat(loss_histories, ignore_index=True)
    history_csv_path = os.path.join(checkpoint_dir, "training_history.csv")
    all_histories_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")

    return model, (scaler_path, pca_path), all_histories_df, (X_val, Y_val_plastic)
