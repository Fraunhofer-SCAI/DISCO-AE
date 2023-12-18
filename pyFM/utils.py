import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree

def knn_query(X, Y, k=1, return_distance=False, use_scipy=False, dual_tree=False, n_jobs=1):

    if use_scipy:
        tree = scipy.spatial.KDTree(X, leafsize=40)
        dists, matches = tree.query(Y, k=k, workers=-1)
    else:
        # if n_jobs == 1:
        #     tree = KDTree(X, leaf_size=40)
        #     dists, matches = tree.query(Y, k=k, return_distance=True)
        # else:
        tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
        tree.fit(X)
        dists, matches = tree.kneighbors(Y)
        if k == 1:
            dists = dists.squeeze()
            matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches

def knn_query_normals(X, Y, normals1, normals2, k_base=30, return_distance=False, n_jobs=1, verbose=False):
    """
    Compute a NN query ensuring normal consistency.
    k_base determines the number of neighbors first computed for faster computation.
    """
    final_matches = np.zeros(Y.shape[0], dtype=int)
    final_dists = np.zeros(Y.shape[0])

    # FIRST DO KNN SEARCH HOPING TO OBTAIN FAST
    # tree = KDTree(X)  # Tree on (n1,)
    # dists, matches = tree.query(Y, k=k_base, return_distance=True)  # (n2,k), (n2,k)

    dists, matches = knn_query(X, Y, k=k_base, return_distance=True, n_jobs=n_jobs)

    # Check for which vertices the solution is already computed
    isvalid = np.einsum('nkp,np->nk', normals1[matches], normals2) > 0  # (n2, k)
    valid_row = isvalid.sum(1) > 0

    valid_inds = valid_row.nonzero()[0]  # (n',)
    invalid_inds = (~valid_row).nonzero()[0]  # (n2-n')

    if verbose:
        print(f'{valid_inds.size} direct matches and {invalid_inds.size} specific indices')

    # Fill the known values
    final_matches[valid_inds] = matches[(valid_inds, isvalid[valid_inds].argmax(axis=1))]
    if return_distance:
        final_dists[valid_inds] = dists[(valid_inds, isvalid[valid_inds].argmax(axis=1))]

    # Individually check other indices
    n_other = invalid_inds.size
    myit = range(n_other)
    for inv_ind in myit:
        vert_ind = invalid_inds[inv_ind]
        possible_inds = np.nonzero(normals1 @ normals2[vert_ind] > 0)[0]

        if len(possible_inds) == 0:
            final_matches[vert_ind] = matches[vert_ind,0]
            final_dists[vert_ind] = dists[vert_ind,0]
            continue

        tree = KDTree(X[possible_inds])
        temp_dist, temp_match_red = tree.query(Y[None, vert_ind], k=1, return_distance=True)

        final_matches[vert_ind] = possible_inds[temp_match_red.item()]
        final_dists[vert_ind] = temp_dist.item()

    if return_distance:
        return final_dists, final_matches
    return final_matches


