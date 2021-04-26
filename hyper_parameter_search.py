import argparse
from hyperopt import tpe, Trials, fmin, hp

from train import DATASETS, train


def search(dataset='mnist', minimize='accuracy', iterations=10):
    def objective(
            batch_size=64,      # Batch size
            epochs=25,          # Number of epochs
            layer_depth=50,     # Layer depth of ResNet
            filter_depth=64,    # Filter depth of ResNet
    ):
        """Returns validation score from hyperparameters"""
        test_scores = train(batch_size, epochs, dataset, layer_depth, filter_depth)
        if minimize == 'accuracy':
            return test_scores[1]
        return test_scores[0]

    # Define the search space
    space = {
        'batch_size': hp.quniform('batch_size', 1, 256, 64),
        'epochs': hp.quniform('epochs', 1, 100, 25),
        'layer_depth': hp.quniform('layer_depth', 16, 256, 50),
        'filter_depth': hp.quniform('filter_depth', 1, 256, 64),
    }

    # Algorithm
    tpe_algorithm = tpe.suggest

    # Trials object to track progress
    bayes_trials = Trials()

    # Optimize
    best = fmin(fn=objective, space=space, algo=tpe_algorithm,
                max_evals=iterations, trials=bayes_trials)

    print(best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter search')
    parser.add_argument('-d', '--dataset', choices=DATASETS.keys(), help='Dataset to use')
    parser.add_argument('-m', '--minimize', choices=['accuracy', 'loss'], help='Metric to minimize')
    parser.add_argument('-i', '--iterations', type=int, action='store', help='Iterations of search to optimize over')
    args = parser.parse_args()
    print(f'Seaching: {args}')
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    search(**kwargs)
