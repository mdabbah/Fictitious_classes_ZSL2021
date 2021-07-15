import pickle

import numpy as np
from scipy import stats
from sklearn.preprocessing import normalize


def calc_accuracy(true_labels: np.ndarray, predictions: np.ndarray, per_class_acc: bool = True):
    """
    calculates per class accuracy by default, if per_class_acc is false, it calculates per sample accuracy.
    :param true_labels: true labels of the test samples.
    :param predictions: predicted labels of the test samples.
    :param per_class_acc: if true, we calculate per class accuracy, else, calculates per sample accuracy.
    :return: accuracy.
    """

    true_labels = true_labels.squeeze()[:, np.newaxis]
    predictions = predictions.squeeze()[:, np.newaxis]

    test_label_set = np.unique(true_labels)[np.newaxis, :]

    onehot_true_labels = test_label_set == true_labels
    onehot_predictions = test_label_set == predictions

    if per_class_acc:
        per_class_acc = np.sum(np.logical_and(onehot_true_labels, onehot_predictions), axis=0) / np.sum(
            onehot_true_labels, axis=0)
        # from misc.visualize_results import plot_bars
        # class_type = 'seen classes' if len(per_class_acc) ==150 else 'unseen classes'
        # if len(per_class_acc) == 40:
        #     x, y = np.unique(predictions[true_labels==15], return_counts=True)
        #     plot_bars(y, x, axis_labels=('confusing classes', 'counts'), title=f'class 15- humpback confusing classes',
        #               save_folder='./awa2_debug')
        # plot_bars(per_class_acc, axis_labels=('classes', 'accuracy'), title=f'per class accuracy - {class_type}', save_folder='./CUB_debug')
        return per_class_acc.mean()
    else:
        per_sample_acc = np.any(np.logical_and(onehot_true_labels, onehot_predictions), axis=1).mean()
        return per_sample_acc


def calc_harmonic_mean(a, b):

    if isinstance(a, np.ndarray):
        hm = 2 * (a * b) / (a + b)
        hm[a+b == 0] = 0
        return hm
    if a + b == 0:
        return 0
    return 2 * (a * b) / (a + b)


def calc_weighted_avg(weights, vec):
    weights = np.array(weights) / np.sum(weights)
    vec = np.array(vec)
    return np.sum(weights * vec)


def calc_accuracies(base_classifiers_scores, novel_classifiers_scores, true_labels, base_classes, novel_classes):
    seen_acc, unseen_acc, hm = calc_gzsl_accuracies(base_classifiers_scores, novel_classifiers_scores,
                                                    true_labels, base_classes, novel_classes)


def calc_gzsl_accuracies(seen_classifiers_scores, unseen_classifiers_scores, true_labels, seen_classes, unseen_classes):
    # shape: (#seen classes, #samples)
    seen_membership_matrix = np.equal(seen_classes[:, np.newaxis], true_labels[np.newaxis, :])
    # shape: (#unseen classes, #samples)
    unseen_membership_matrix = np.equal(unseen_classes[:, np.newaxis], true_labels[np.newaxis, :])

    scores = np.concatenate([seen_classifiers_scores, unseen_classifiers_scores], axis=1)
    label_set = np.concatenate([seen_classes, unseen_classes])
    predictions = label_set[np.argmax(scores, axis=1)]

    seen_samples_idx = np.any(seen_membership_matrix, axis=0)
    unseen_samples_idx = np.any(unseen_membership_matrix, axis=0)

    seen_acc = calc_accuracy(true_labels=true_labels[seen_samples_idx],
                             predictions=predictions[seen_samples_idx])
    unseen_acc = calc_accuracy(true_labels=true_labels[unseen_samples_idx],
                               predictions=predictions[unseen_samples_idx])
    hm = calc_harmonic_mean(seen_acc, unseen_acc)

    return seen_acc, unseen_acc, hm


def calc_gzsl_metrics(seen_classifiers_scores, unseen_classifiers_scores, true_labels, seen_classes, unseen_classes,
                      calc_ausuc, fixed_bias):
    """

    :param seen_classifiers_scores:
    :param unseen_classifiers_scores:
    :param true_labels:
    :param seen_classes:
    :param unseen_classes:
    :param calc_ausuc:
    :param fixed_bias:
    :return:
    """
    if (fixed_bias is not None) and (not calc_ausuc):
        seen_classifiers_scores = seen_classifiers_scores - fixed_bias
        seen_acc, unseen_acc, hm = calc_gzsl_accuracies(seen_classifiers_scores, unseen_classifiers_scores, true_labels,
                                                        seen_classes, unseen_classes)

        return {'acc_seen': seen_acc, 'acc_novel': unseen_acc, 'H': hm, 'bias': fixed_bias, 'auc': 0}

    # shape: (#seen classes, #samples)
    seen_membership_matrix = np.equal(seen_classes[:, np.newaxis], true_labels[np.newaxis, :])
    # shape: (#unseen classes, #samples)
    unseen_membership_matrix = np.equal(unseen_classes[:, np.newaxis], true_labels[np.newaxis, :])

    # we will use those to define the biases
    seen_prediction_scores = np.max(seen_classifiers_scores, axis=1)
    unseen_prediction_scores = np.max(unseen_classifiers_scores, axis=1)

    # we will use those to calculate the accuracies
    # prediction according to seen classes classifiers for all test samples
    seen_classes_pred_idx = np.argmax(seen_classifiers_scores, axis=1)
    seen_predictions = seen_classes[seen_classes_pred_idx]
    # prediction according to unseen classes classifiers for all test samples
    unseen_classes_pred_idx = np.argmax(unseen_classifiers_scores, axis=1)
    unseen_predictions = unseen_classes[unseen_classes_pred_idx]

    seen_classes_count = np.sum(seen_membership_matrix, axis=1)
    unseen_classes_count = np.sum(unseen_membership_matrix, axis=1)

    seen_correct_count = np.sum(np.logical_and(np.equal(seen_classes[:, np.newaxis], seen_predictions[np.newaxis, :]),
                                               seen_membership_matrix), axis=1)

    unseen_correct_count = np.sum(np.logical_and(
        np.equal(unseen_classes[:, np.newaxis], unseen_predictions[np.newaxis, :]), unseen_membership_matrix), axis=1)

    # bias calculations start
    # sample i will flip prediction from seen to unseen if it had the
    # ith bias
    biases = seen_prediction_scores - unseen_prediction_scores
    biases_sort_loc = np.argsort(biases)
    biases = np.sort(biases)

    seen_correct = np.equal(true_labels, seen_predictions) + 0.
    unseen_correct = np.equal(true_labels, unseen_predictions) + 0.

    # the change in accuracy each sample will cause due to flipping a sample from seen to unseen
    acc_change_seen = seen_correct[biases_sort_loc] / seen_classes_count[seen_classes_pred_idx[biases_sort_loc]]
    acc_change_seen /= seen_classes.shape[0]

    acc_change_unseen = unseen_correct[biases_sort_loc] / unseen_classes_count[unseen_classes_pred_idx[biases_sort_loc]]
    acc_change_unseen /= unseen_classes.shape[0]

    seen_accuracy = np.cumsum(-acc_change_seen) + np.mean(seen_correct_count / seen_classes_count)
    unseen_accuracy = np.cumsum(acc_change_unseen)

    unique_biases_idx = np.append(np.unique(biases, return_index=True)[1], [0, biases.shape[0] - 1])
    unique_biases_idx = np.unique(unique_biases_idx)

    seen_accuracy = seen_accuracy[unique_biases_idx]
    unseen_accuracy = unseen_accuracy[unique_biases_idx]
    biases = biases[unique_biases_idx]

    harmonic_mean = calc_harmonic_mean(seen_accuracy, unseen_accuracy)
    AUSUC = np.trapz(seen_accuracy, unseen_accuracy)

    if fixed_bias is None:
        best_idx = np.argmax(harmonic_mean)
        best_bias = biases[best_idx]

        seen_classifiers_scores = seen_classifiers_scores - best_bias
        seen_acc, unseen_acc, hm = calc_gzsl_accuracies(seen_classifiers_scores, unseen_classifiers_scores, true_labels,
                                                        seen_classes, unseen_classes)

        return {'acc_seen': seen_acc, 'acc_novel': unseen_acc, 'H': hm, 'bias': best_bias, 'auc': AUSUC}

    else:
        seen_classifiers_scores = seen_classifiers_scores - fixed_bias
        seen_acc, unseen_acc, hm = calc_gzsl_accuracies(seen_classifiers_scores, unseen_classifiers_scores, true_labels,
                                                        seen_classes, unseen_classes)

    return {'acc_seen': seen_acc, 'acc_novel': unseen_acc, 'H': hm, 'bias': fixed_bias, 'auc': AUSUC}


def _zero_nans_and_infs(mat: np.ndarray) -> np.ndarray:
    """
    puts zeros where there's nans or infs.
    :param mat:
    :return:
    """
    mat[np.isnan(mat)] = 0
    mat[np.isinf(mat)] = 0
    return mat


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """
    normalizes each row of the given mat by its own l2 norm,
    replaces nans and infs by 0 before and after the normalization
    :param mat: mat to be normalized
    :return: the normalized mat
    """
    mat = _zero_nans_and_infs(mat)
    mat = np.divide(mat, np.sqrt(np.sum(np.square(mat), axis=1))[:, np.newaxis])
    mat = _zero_nans_and_infs(mat)

    return mat


def calc_l2_dists(mat: np.ndarray) -> np.ndarray:
    """
    calculates the l2 distance between rows of the given matrix
    :param mat: a matrix shaped (R,C)
    :return: dists - a  matrix shaped (R,R) where in cell i,j is the l2 distance between
    row i and row j in mat
    """
    mat = mat.T
    dists = np.sum(np.square(mat[:, :, np.newaxis] - mat[:, np.newaxis, :]), axis=0)
    dists = np.sqrt(dists)

    return dists


def calc_l2_dists_fast(mat_a: np.ndarray, mat_b: np.ndarray = None) -> np.ndarray:
    """
    calculates the l2 distance between rows of mat a and rows of mat b
    if mat b is None then we calculate the distances between the row of mat a itself
    :param mat_a: a matrix shaped (R1,C)
    :param mat_b: a matrix shaped (R2,C)
    :return: dists - a  matrix shaped (R1,R2) where in cell i,j is the l2 distance between
    row i in mat a and row j in mat b (or a if mat b is None
    """
    if mat_b is None:
        mat_b = mat_a
        inner_products = np.matmul(mat_a, mat_b.T)
        mat_a_inner_product = inner_products
        mat_b_inner_product = inner_products
    else:
        inner_products = np.matmul(mat_a, mat_b.T)
        mat_a_inner_product = np.matmul(mat_a, mat_a.T)
        mat_b_inner_product = np.matmul(mat_b, mat_b.T)

    dists = np.diag(mat_a_inner_product)[:, np.newaxis] + np.diag(mat_b_inner_product)[np.newaxis,
                                                          :] - 2 * inner_products
    dists[dists < 0] = 0
    dists = np.sqrt(dists)

    return dists


def calc_bhattacharyya_distance(normal_params1: np.ndarray, normal_params2: np.ndarray = None):
    """
    calculates the l2 distance between rows of mat a and rows of mat b
    if mat b is None then we calculate the distances between the row of mat a itself
    :param mat_a: a matrix shaped (R1,C)
    :param mat_b: a matrix shaped (R2,C)
    :return: dists - a  matrix shaped (R1,R2) where in cell i,j is the l2 distance between
    row i in mat a and row j in mat b (or a if mat b is None
    """
    dim = normal_params1.shape[1] // 2

    means1 = normal_params1[:, :dim]
    means2 = normal_params2[:, :dim]

    logvars1 = normal_params1[:, dim:]
    logvars2 = normal_params2[:, dim:]

    covar_reg = 0
    vars1 = np.exp(logvars1) + covar_reg
    vars2 = np.exp(logvars2) + covar_reg

    var_com = (vars1[np.newaxis, :, :] + vars2[:, np.newaxis, :]) / 2

    mahalanobis_dist_squared = np.square(means1[np.newaxis, :, :] - means2[:, np.newaxis, :])
    mahalanobis_dist_squared = np.sum(mahalanobis_dist_squared / var_com, axis=-1)

    sigmas_term = -0.5 * (np.sum(logvars1, axis=-1)[np.newaxis, :] + np.sum(logvars2, axis=-1)[:, np.newaxis]) \
                  + np.sum(np.log(var_com), axis=-1)

    dists = 1 / 8 * mahalanobis_dist_squared + 0.5 * sigmas_term
    return dists.T


def calc_hellinger_distance(normal_params1: np.ndarray, normal_params2: np.ndarray = None):
    """
    calculates the l2 distance between rows of mat a and rows of mat b
    if mat b is None then we calculate the distances between the row of mat a itself
    :param mat_a: a matrix shaped (R1,C)
    :param mat_b: a matrix shaped (R2,C)
    :return: dists - a  matrix shaped (R1,R2) where in cell i,j is the l2 distance between
    row i in mat a and row j in mat b (or a if mat b is None
    """
    dim = normal_params1.shape[1] // 2

    means1 = normal_params1[:, :dim]  # (#classes1, d)
    means2 = normal_params2[:, :dim]  # (#classes2, d)

    logvars1 = normal_params1[:, dim:]  # (classes1, d)
    logvars2 = normal_params2[:, dim:]  # (#classes2, d)

    vars1 = np.exp(logvars1)
    vars2 = np.exp(logvars2)

    var_com = (vars1[np.newaxis, :, :] + vars2[:, np.newaxis, :]) / 2  # (classes2, classes1, d)

    mahalanobis_dist_squared = np.square(means1[np.newaxis, :, :] - means2[:, np.newaxis, :])
    mahalanobis_dist_squared = np.sum(mahalanobis_dist_squared / var_com, axis=-1)  # (classes2, classes1)

    det_var1_4throot = np.prod(np.power(vars1, 1 / 4), axis=-1)[np.newaxis, :]
    det_var2_4throot = np.prod(np.power(vars2, 1 / 4), axis=-1)[:, np.newaxis]
    det_var_com_sqrt = np.prod(np.power(var_com, 1 / 2), axis=-1)

    sigmas_term = (det_var1_4throot * det_var2_4throot) / det_var_com_sqrt  # (classes2, classes1)

    dists_squared = 1 - sigmas_term * np.exp(- 1 / 8 * mahalanobis_dist_squared)
    return np.sqrt(dists_squared).T


def calc_jensen_shanon_distance(class_attr_a, class_attr_b):
    raise NotImplemented


def calc_dists(class_attr_a: np.ndarray, class_attr_b: np.ndarray, dist_type: str = 'l2'):
    if dist_type == 'l2':
        dists = calc_l2_dists_fast(class_attr_a, class_attr_b)

    elif dist_type == 'bhattacharyya':
        dists = calc_bhattacharyya_distance(class_attr_a, class_attr_b)

    elif dist_type == 'l2_half':
        idx = class_attr_a.shape[1] // 2
        dists = calc_l2_dists_fast(class_attr_a[:, :idx], class_attr_b[:, :idx])

    elif dist_type == 'hellinger':
        dists = calc_hellinger_distance(class_attr_a, class_attr_b)

    elif dist_type == 'jensen_shanon':
        dists = calc_jensen_shanon_distance(class_attr_a, class_attr_b)

    else:
        raise ValueError(f'dist_type {dist_type} is not supported')

    return dists


dists_cache = {}


def load_dists(load_info, force_load=False):
    global dists_cache
    dists = None
    if load_info is None or load_info.get('attribute_type', None) is None:
        return dists

    gfzsl_att = 'gfzsl' in load_info['attribute_type']
    if gfzsl_att or force_load:
        attribute_type = load_info['attribute_type']
        dataset_name = load_info['dataset_name']
        feature_type = load_info['feature_type']
        dist_type = load_info['dist_type']
        if gfzsl_att:
            load_path = f"../dataset/data/dists/{dataset_name}_PS_{attribute_type}_{feature_type}_{dist_type}"
        else:
            load_path = f"../dataset/data/dists/{dataset_name}_PS_{attribute_type}_{dist_type}"
        dists = dists_cache.get(load_path, None)
        if dists is None:
            with open(load_path, 'rb') as pkl_file:
                dists = pickle.load(pkl_file)
                dists_cache[load_path] = dists
        if 'a_classes' in load_info and 'b_classes' in load_info:
            a_classes = load_info['a_classes']
            b_classes = load_info['b_classes']

            dists = dists[a_classes, :]
            dists = dists[:, b_classes]

    return dists


def calc_per_class_centroids(features, labels, reduce_regions=True, num_samples2avg=None):
    label_set = np.unique(labels)

    if len(features.shape) > 2 and reduce_regions:
        features = np.mean(np.mean(features, axis=-1), axis=-1)

    classes_centroids = []
    for c_ in label_set:
        class_samples = features[labels == c_, :]
        class_centroid = np.mean(class_samples[:num_samples2avg], axis=0)
        classes_centroids.append(class_centroid)

    return np.stack(classes_centroids, axis=0)  # shape: classes X *shape(features)


def choose_class_pairs(features, labels, num_class_pairs, class_pair_criterion):
    class_centroids = calc_per_class_centroids(features, labels, reduce_regions=True)
    class_pairs = choose_samples(class_centroids, class_centroids, k=num_class_pairs, criterion=class_pair_criterion)

    labels = np.unique(labels, return_inverse=True)[1]
    labels = labels
    num_classes = len(np.unique(labels))
    class_pairs_labels = {(cp[0], cp[1]): i + num_classes for i, cp in enumerate(class_pairs)}

    return class_pairs, class_pairs_labels


def calc_avg_percision_at_k(y_true, y_scores, k=50):
    label_set = np.unique(y_true)
    precision_per_class = []
    for cid, class_ in enumerate(label_set):
        top_k = np.argsort(y_scores[:, cid])[-k:]
        precision_per_class.append(np.mean(y_true[top_k] == class_))

    return np.mean(precision_per_class)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def choose_samples(samples_a, samples_b, k, criterion):
    if criterion == 'most_similar':
        return get_k_most_similar(samples_a, samples_b, k)
    elif criterion == 'least_similar':
        return get_k_most_similar(-samples_a, samples_b, k)
    elif criterion == 'random':
        return get_k_random_pairs(samples_a.shape[0], samples_b.shape[0], k)


def get_k_random_pairs(num_samples_a, num_samples_b, k):
    mat = np.arange(num_samples_a*num_samples_b)+0.
    np.random.shuffle(mat)
    mat = np.reshape(mat, [num_samples_a, num_samples_b])

    lower_tri_ind = np.tril_indices(mat.shape[0], k=0, m=mat.shape[1])
    mat[lower_tri_ind] = -np.inf
    chosen_paris = np.unravel_index(np.argsort(mat, axis=None), mat.shape)
    chosen_paris = np.stack(chosen_paris, axis=1)[:-k - 1:-1, :]
    return chosen_paris


def get_k_most_similar(samples_a, samples_b, k):
    # calc class similarity
    samples_a = normalize(samples_a, axis=1)
    samples_b = normalize(samples_b, axis=1)
    similarity = np.matmul(samples_a, samples_b.T)  # cosine sim,
    # similarity[i,j] means how similar is sample i in group a to sample j in group b

    # choose top k similar classes, self similarity is not included
    # similarity = np.triu(similarity, k=1)
    lower_tri_ind = np.tril_indices(similarity.shape[0], k=0, m=similarity.shape[1])
    similarity[lower_tri_ind] = -np.inf
    most_similar_samples = np.unravel_index(np.argsort(similarity, axis=None), similarity.shape)
    most_similar_samples = np.stack(most_similar_samples, axis=1)[:-k - 1:-1, :]

    return most_similar_samples