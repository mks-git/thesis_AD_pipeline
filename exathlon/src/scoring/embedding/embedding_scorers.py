from sklearn.neighbors import NearestNeighbors
import numpy as np
from scoring.scorers import Scorer

class TS2VecScorer(Scorer):
    def __init__(self, args, embedder, output_path, scorer=None):
        super().__init__(args, embedder, output_path, scorer)
        self.k = getattr(args, 'ts2vec_knn_k', 5)
        self.metric = getattr(args, 'ts2vec_knn_metric', 'cosine')

    def score_windows(self, X):
        embeddings = self.normality_model.encode(X)
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric).fit(embeddings)
        dists, _ = nn.kneighbors(embeddings, return_distance=True)
        return dists.mean(axis=1).astype(np.float32)

    def score_period(self, period):
        # Score all sliding windows in the period, then aggregate to record-wise scores
        from data.helpers import get_sliding_windows
        window_size = self.normality_model.window_size
        windows = get_sliding_windows(period, window_size, 1)
        window_scores = self.score_windows(windows)
        # Rolling average to map window scores to record scores
        rolling_scores = np.convolve(window_scores, np.ones(window_size) / window_size, mode='full')
        return rolling_scores[:len(period)]

    def score(self, periods):
        # Score each period in a batch
        return np.array([self.score_period(period) for period in periods])
    
# dictionary gathering references to the defined embedding-based scoring methods
scoring_classes = {
    'knn': TS2VecScorer
}