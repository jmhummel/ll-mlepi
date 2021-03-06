import argparse
from hyperopt import tpe, Trials, fmin, hp, STATUS_OK

from train import DATASETS, train


def search(dataset='mnist', minimize='accuracy', max_epochs=100, iterations=10):
    def objective(params):
        """Returns validation score from hyperparameters"""
        test_scores = train(2**int(params.get('log2_batch_size')),
                            int(params.get('epochs')),
                            dataset,
                            int(params.get('layer_depth')),
                            2**int(params.get('log2_filter_depth')),
                            verbose=False)
        loss = test_scores[0]
        if minimize == 'accuracy':
            loss = 1 - test_scores[1]
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    # Define the search space
    space = {
        'log2_batch_size': hp.quniform('batch_size', 1, 8, 1),
        'epochs': hp.quniform('epochs', 1, max_epochs, 1),
        'layer_depth': hp.quniform('layer_depth', 16, 256, 1),
        'log2_filter_depth': hp.quniform('filter_depth', 1, 8, 1),
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
    parser.add_argument('-e', '--max-epochs', type=int, action='store', dest='max_epochs', help='Max epochs to train')
    parser.add_argument('-i', '--iterations', type=int, action='store', help='Iterations of search to optimize over')
    args = parser.parse_args()
    print(f'Seaching: {args}')
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    search(**kwargs)
