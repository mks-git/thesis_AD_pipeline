import os
import numpy as np
from abc import abstractmethod

from utils.common import MODELING_TRAIN_NAME, MODELING_VAL_NAME, get_output_path

from sklearn.neighbors import NearestNeighbors

import sys

# Add the ts2vec directory to sys.path
ts2vec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../ts2vec'))
if ts2vec_path not in sys.path:
    sys.path.insert(0, ts2vec_path)

from ts2vec import TS2Vec

class Embedder:
    """Reconstruction model base class. Aiming to reconstruct `window_size`-windows.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the model and training information to.
        model (keras.model): optional reconstruction model initialization.
    """
    def __init__(self, args, output_path, model=None):
        # number of records of the windows to reconstruct
        self.window_size = args.window_size
        self.output_path = output_path
        self.model = model
        # model hyperparameters (used along with the timestamp as the model id)
        self.hp = dict()
        # model training epoch times
        self.epoch_times = []

    @classmethod
    def from_file(cls, args, model_root_path):
        full_model_path = os.path.join(model_root_path, 'model.pt')
        print(f'loading embedding model file {full_model_path}...', end=' ', flush=True)
        ts2vec_hp = {
            'input_dims': args.ts2vec_input_dims,
            'output_dims': getattr(args, 'ts2vec_output_dims', 320),
            'hidden_dims': getattr(args, 'ts2vec_hidden_dims', 64),
            'depth': getattr(args, 'ts2vec_depth', 10),
            'device': getattr(args, 'ts2vec_device', 'cpu'),
            'lr': getattr(args, 'ts2vec_lr', 0.001),
            'batch_size': getattr(args, 'ts2vec_batch_size', 16),
            'max_train_length': getattr(args, 'ts2vec_max_train_length', None),
            'temporal_unit': getattr(args, 'ts2vec_temporal_unit', 0),
            'after_iter_callback': getattr(args, 'ts2vec_after_iter_callback', None),
            'after_epoch_callback': getattr(args, 'ts2vec_after_epoch_callback', None)
        }
        model = TS2Vec(**ts2vec_hp)
        model.load(full_model_path)
        print('done.')
        return cls(args, '', model)

    @abstractmethod
    def fit(self, X_train, X_val):
        """Fits the representation learning model to `X_train` samples, validating on `X_val` samples.

        Args:
            X_train (ndarray): training samples of shape `(n_samples, window_size, n_features)`.
            X_val (ndarray): validation samples of shape shape `(n_samples, window_size, n_features)`.
        """

    @abstractmethod
    def encode(self, X):
        """Returns the embeddings of the samples of `X` by the model.

        Args:
            X (ndarray): samples to embed of shape `(n_samples, window_size, n_features)`.

        Returns:
            ndarray: embedded samples of shape `(n_samples, embedding_dim)`.
        """


class TS2VecEmbedder(Embedder):
    """TS2Vec-based Embedder class.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the model and training information to.
        model (TS2Vec): optional TS2Vec model initialization.
    """
    def __init__(self, args, output_path, model=None):
        super().__init__(args, output_path, model)
        self.args = args
        self.ts2vec_hp = {
            'input_dims': args.ts2vec_input_dims,
            'output_dims': getattr(args, 'ts2vec_output_dims', 320),
            'hidden_dims': getattr(args, 'ts2vec_hidden_dims', 64),
            'depth': getattr(args, 'ts2vec_depth', 10),
            'device': getattr(args, 'ts2vec_device', 'cpu'),
            'lr': getattr(args, 'ts2vec_lr', 0.001),
            'batch_size': getattr(args, 'ts2vec_batch_size', 16),
            'max_train_length': getattr(args, 'ts2vec_max_train_length', None),
            'temporal_unit': getattr(args, 'ts2vec_temporal_unit', 0),
            'after_iter_callback': getattr(args, 'ts2vec_after_iter_callback', None),
            'after_epoch_callback': getattr(args, 'ts2vec_after_epoch_callback', None)
        }
        # Training hyperparameters for TS2Vec
        self.train_hp = {
            'n_epochs': getattr(args, 'ts2vec_n_epochs', None),
            'n_iters': getattr(args, 'ts2vec_n_iters', None),
            'verbose': getattr(args, 'ts2vec_verbose', True)
        }
        self.hp = dict(**self.ts2vec_hp, **self.train_hp)
        self.model = model

    def fit(self, X_train, X_val=None):
        self.ts2vec_hp['input_dims'] = X_train.shape[-1]

        # ... set self.ts2vec_hp ...
        # Update args with current hyperparameters
        for k, v in self.ts2vec_hp.items():
            setattr(self.args, f'ts2vec_{k}', v)
        # Update output path based on current args and hyperparameters
        self.output_path = get_output_path(self.args, 'train_model', 'model')
        os.makedirs(self.output_path, exist_ok=True)


        self.model = TS2Vec(**self.ts2vec_hp)
        self.model.fit(
            X_train,
            n_epochs=self.train_hp['n_epochs'],
            n_iters=self.train_hp['n_iters'],
            verbose=self.train_hp['verbose']
        )
        self.save(os.path.join(self.output_path, "model.pt"))

    def encode(self, X):
        # Step 1: Use TS2Vec to encode the input (X) into embeddings
        embeddings = self.model.encode(X, encoding_window='full_series')

        # Step 2: Calculate anomaly scores based on distance in the embedding space
        # Use KNN (K-Nearest Neighbors) to compute distances between embeddings
        # nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(embeddings)
        # distances, _ = nbrs.kneighbors(embeddings)

        # Step 3: Calculate the anomaly score for each sample
        # Use average distance to nearest neighbors as anomaly score
        # anomaly_scores = distances.mean(axis=1)

        return embeddings

    def save(self, path):
        if self.model:
            self.model.save(path)

    def load(self, path):
        if self.model:
            self.model.load(path)




# dictionary gathering references to the defined embedding methods
embedding_classes = {
    'ts2vec': TS2VecEmbedder
}