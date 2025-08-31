import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from utils.common import MODELING_SET_NAMES
from modeling.helpers import prepend_training_info

def save_embedding_evaluation(data_dict, embedder, modeling_task_string, config_name, spreadsheet_path, k=5):
    """Adds the embedding evaluation of this configuration to a sorted comparison spreadsheet.

    Args:
        data_dict (dict): train, val and test windows, as `{X_(modeling_set_name): ndarray}`.
        embedder: the embedding model object to evaluate.
        modeling_task_string (str): formatted modeling task arguments to compare models under the same task.
        config_name (str): unique configuration identifier serving as an index in the spreadsheet.
        spreadsheet_path (str): comparison spreadsheet path.
        k (int): number of neighbors for kNN scoring.

    Returns:
        pd.DataFrame: the 1-row evaluation DataFrame holding the computed metrics.
    """
    full_spreadsheet_path = os.path.join(spreadsheet_path, f'{modeling_task_string}_comparison.csv')

    column_names, metrics = [], []
    for set_name in MODELING_SET_NAMES:
        print(f'evaluating embedding metrics on the {set_name} set...', end=' ', flush=True)
        X = data_dict[f'X_{set_name}']
        embeddings = embedder.encode(X)
        # Example: kNN anomaly score (mean distance to k nearest neighbors)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        # Exclude self-distance (first column)
        mean_knn_dist = np.mean(distances[:, 1:])
        metrics.append(mean_knn_dist)
        column_names.append((set_name + '_MEAN_KNN_DIST').upper())
        print('done.')
    # prepend training time information
    metrics, column_names = prepend_training_info(embedder, metrics, column_names)
    evaluation_df = pd.DataFrame(columns=column_names, data=[metrics], index=[config_name])
    evaluation_df.index.name = 'method'

    try:
        comparison_df = pd.read_csv(full_spreadsheet_path, index_col=0).astype(float)
        print(f'adding evaluation of `{config_name}` to {full_spreadsheet_path}...', end=' ', flush=True)
        comparison_df.loc[evaluation_df.index[0], :] = evaluation_df.values[0]
        comparison_df.sort_values(by=list(reversed(column_names)), ascending=True, inplace=True)
        comparison_df.to_csv(full_spreadsheet_path)
        print('done.')
    except FileNotFoundError:
        print(f'creating {full_spreadsheet_path} with evaluation of `{config_name}`...', end=' ', flush=True)
        evaluation_df.to_csv(full_spreadsheet_path)
        print('done.')
    return evaluation_df