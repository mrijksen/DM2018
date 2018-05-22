# libraries
import pandas as pd
import numpy as np

def dcg_at_k(r, k, method=1):
    """
    Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain

    Source: https://github.com/benhamner/ExpediaPersonalizedSortCompetition
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return 2**r[0] - 1 + np.sum((2**r[1:]-1) / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=1):
    """
    Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain

    Source: https://github.com/benhamner/ExpediaPersonalizedSortCompetition
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def compute_ndcg(path_results):
    """
    path_results is a string containing the path to the .csv file
    which contains the results of the model. The .csv should have
    columns:

    SearchId = unique ID for each user
    Relevance = 5 if booked, 1 if clicked
    PropertyId = unique ID for each property
    """

    # load results and dataset
    results = pd.read_csv(path_results)
    ndcg_list = []

    # iterate over all queries
    id_list = results['SearchId'].unique()
    for id in id_list:

        # get relevance scores of query
        list_rscores = results['Relevance'][results.SearchId == id]

        # number of results to consider
        num_res = len(list_rscores)

        # compute ndcg for each query
        ndcg_list.append(ndcg_at_k(list_rscores,num_res))
    return np.mean(ndcg_list)
