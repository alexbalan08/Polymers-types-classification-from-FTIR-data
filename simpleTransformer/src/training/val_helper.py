import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from simpleTransformer.src.models.decoder import SequenceDecoder
from simpleTransformer.src.models.encoder import FTIREncoder
from simpleTransformer.src.models.predictor import FTIRMonomerPredictor
from simpleTransformer.src.models.transformer import FTIRToSequenceTransformer


def test_cross_validation(
    X, Y, tokenizer, data_module, checkpoint_dir,
    n_splits=3,
    batch_size=10,
    d_model=32,
    num_heads=4,
    num_layers=2,
    drop_rate=0.1,
    pred_threshold=0.1,
    max_len=64,
    start_fold=1
):
    """
    Performs stratified k-fold training on FTIR -> SMILES dataset.

    Returns:
        model: last trained model
        all_histories_df: pd.DataFrame with loss history
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fold_id, (train_idx, val_idx) in enumerate(cv.split(X, Y), start=1):
        if fold_id < start_fold:
            continue
        fold_dir = os.path.join(checkpoint_dir, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\nFold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")

        # Data transformations
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val_plastic = Y[train_idx], Y[val_idx]
        X_train, Y_train, (scaler, pca, kmeans) = data_module.transform_data(X_train, Y_train)
        X_val, Y_val = data_module.transform_data(X_val, Y_val_plastic, pca_objects=(scaler, pca, kmeans))

        scaler_path = os.path.join(fold_dir, f"ftir_scaler.save")
        pca_path = os.path.join(fold_dir, "ftir_pca.save")

        # Build model
        encoder = FTIREncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=drop_rate,
            is_fp=False
        )

        decoder = SequenceDecoder(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=drop_rate
        )

        model = FTIRToSequenceTransformer(encoder, decoder)

        # Prepare datasets
        # train_dataset = tf.data.Dataset.from_tensor_slices(
        #     ((X_train, Y_train[:, :-1]), Y_train[:, 1:])
        # ).shuffle(1024).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            ((X_val, Y_val[:, :-1]), Y_val[:, 1:])
        ).batch(batch_size)

        x_input = X_val[0]
        x_input = x_input.reshape(1, -1)
        y_input = np.array([[tokenizer.token_to_id("<SOS>")]], dtype=np.int32)
        _ = model((x_input, y_input), training=False)

        # load weights
        model_load_path = os.path.join(fold_dir, "ftir_transformer.weights.h5")
        model.load_weights(model_load_path)
        print("SUCCESFULLY loaded weights for", model_load_path)

        # --------------------------
        # 3. Use predictor
        # --------------------------
        predictor = FTIRMonomerPredictor(
            model=model,
            tokenizer=tokenizer,
            scaler_path=scaler_path,
            pca_path=pca_path,
            max_len=max_len
        )

        store_x = []
        store_y = []
        store_pred = []
        store_probs = []

        X_val, Y_val = X_val[np.concatenate(([True], np.any(X_val[1:] != X_val[:-1], axis=1)))], Y_val_plastic
        start_time = time.time()
        prev_x = X_val[0]
        for i, (xv, yv) in enumerate(zip(X_val, Y_val)):
            # example_ftir = X_val[0]
            # print("Reduced form spectra:", X_val[0])

            print()
            print(i)
            print("Equal previous", np.array_equal(xv, prev_x))
            prev_x = xv
            predicted_molecules, probs = predictor.predict(xv, pred_threshold, debug=False)
            print("Num predicted", len(predicted_molecules))

            for smile, p in zip(predicted_molecules, probs):
                print(f"Predicted SMILES (conf={p:5.3f}): {smile}")
            print(f"True SMILES was:  {Y_val[0]}")

            store_x.append(xv)
            store_y.append(yv)
            store_pred.append(predicted_molecules)
            store_probs.append(probs)
        print(f"PREDICTION took {time.time() - start_time} seconds")

        prediction_df = pd.DataFrame({
            "X_val": store_x,
            "Y_val": store_y,
            "prediction": store_pred,
            "probabilities": store_probs,
        })
        prediction_df.to_csv(os.path.join(fold_dir, "prediction.csv"), index=False)