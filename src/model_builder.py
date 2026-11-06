from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def build_model():

    model = Sequential()

    # First convolutional layer
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(224, 224, 3)
        )
    )

    # First pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu"
        )
    )

    # Second pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convert feature maps into 1D vector
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation="relu"))

    # Dropout to reduce overfitting
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(4, activation="softmax"))

    # Configure the model for training
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()