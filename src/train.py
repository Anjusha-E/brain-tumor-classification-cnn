from src.dataset_builder import build_dataset
from src.model_builder import build_model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Load dataset
x_train, y_train, x_test, y_test = build_dataset()

print("Training data:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing data:", x_test.shape)
print("Testing labels:", y_test.shape)


# Build CNN model
model = build_model()


# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# Save trained model
model.save("models/brain_tumor_cnn.keras")


