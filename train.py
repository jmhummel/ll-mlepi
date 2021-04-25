from tensorflow import keras
from networks import ResNet
import argparse

DATASETS = {
    'cifar10': (keras.datasets.cifar10, 3, 32, 10),
    'cifar100': (keras.datasets.cifar100, 3, 32, 100),
    'fashion_mnist': (keras.datasets.fashion_mnist, 1, 28, 10),
    'mnist': (keras.datasets.mnist, 1, 28, 10),
}

def train(
        batch_size  = 64,       # Batch size
        epochs      = 25,       # Number of epochs
        dataset     = 'mnist',  # Dataset to use
        layer_depth = 50,       # Layer depth of ResNet
):
    data_source, num_channels, resolution, label_size = DATASETS[dataset]
    (x_train, y_train), (x_test, y_test) = data_source.load_data(dataset)
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    model = ResNet(num_channels=num_channels, resolution=resolution, label_size=label_size, layer_depth=layer_depth)
    print(model.summary())

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet')
    parser.add_argument('-b', '--batch-size', type=int, action='store', dest='batch_size', help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, action='store', help='Number of epochs to train over')
    parser.add_argument('-d', '--dataset', choices=DATASETS.keys(), help='Dataset to use')
    parser.add_argument('-l', '--layer-depth', type=int, action='store', dest='layer_depth', help='ResNet layer depth')
    args = parser.parse_args()
    print(f'Training: {args}')
    train(**vars(args))
