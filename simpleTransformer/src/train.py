# train.py
import tensorflow as tf

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, ignore_class=0
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=loss_fn
)

model.fit(
    x=(X, Y[:, :-1]),
    y=Y[:, 1:],
    batch_size=16,
    epochs=50
)
