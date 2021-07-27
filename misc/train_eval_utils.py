import os
from typing import List

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, WeightedRandomSampler, BatchSampler
from torchvision.utils import save_image

from misc.ManifoldCutMixBatchSampler import ManifoldMixBatchSampler
from misc.custome_dataset import CustomedDataset, get_base_transforms
from misc.cutmix import CutMix
from misc.data_loader import ZSLDatasetEmbeddings
from misc.data_utils import load_xlsa17, norm_img_paths
from misc.input_mixup import InputMixUp
from misc.log_utils import Timer, get_human_readable_time
from misc.metrics_utils import calc_gzsl_accuracies, calc_weighted_avg, calc_gzsl_metrics, calc_avg_percision_at_k, \
    mean_confidence_interval


def train_epoch(model, optimizer, train_subset_loader, loss_fn, device):
    model.train()
    for batch_imgs, batch_labels in train_subset_loader:
        batch_imgs = batch_imgs.float().to(device)
        batch_labels = batch_labels.long().to(device)

        optimizer.zero_grad()
        scores = model(batch_imgs)
        loss = loss_fn(scores, batch_labels)
        loss.backward()

        clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        hook_args = {'batch_outputs': scores, 'batch_labels': batch_labels}
        invoke_hooks(model, hook_args, 'after_train_step_hooks')


def model_embed(model, test_subset_loader, device):
    model.eval()
    test_labels = []
    test_embeddings = []
    for batch_imgs, batch_labels in test_subset_loader:
        batch_imgs = batch_imgs.float().to(device)
        batch_labels = batch_labels.long().to(device)

        with torch.no_grad():
            outputs = model.extract(batch_imgs)

        test_labels.append(batch_labels)
        test_embeddings.append(outputs)

    test_labels = torch.cat(test_labels).detach().cpu().numpy()
    if isinstance(test_embeddings[0], torch.Tensor):
        test_embeddings = torch.cat(test_embeddings, dim=0).detach().cpu().numpy()

    if isinstance(test_embeddings[0], dict):
        test_embeddings = torch.cat([o['embedding'] for o in test_embeddings], dim=0).detach().cpu().numpy()

    return test_embeddings, test_labels


def model_predict(model, test_subset_loader, loss_fn, device, output_key=None):
    model.eval()
    test_labels = []
    test_outputs = []
    loss = []
    for batch_imgs, batch_labels in test_subset_loader:
        batch_imgs = batch_imgs.float().to(device)
        batch_labels = batch_labels.long().to(device)

        with torch.no_grad():
            outputs = model(batch_imgs)

        test_labels.append(batch_labels)
        if output_key is not None:
            test_outputs.append(outputs[output_key].detach().cpu().numpy())
        else:
            test_outputs.append(outputs)

        if loss_fn is not None:
            with torch.no_grad():
                batch_loss = loss_fn(outputs, batch_labels)
                loss.append(batch_loss)

    test_labels = torch.cat(test_labels).detach().cpu().numpy()
    if isinstance(test_outputs[0], torch.Tensor):
        test_outputs = torch.cat(test_outputs, dim=0).detach().cpu().numpy()

    if isinstance(test_outputs[0], np.ndarray):
        test_outputs = np.concatenate(test_outputs, axis=0)

    test_loss = 0
    if loss_fn is not None:
        test_loss = torch.mean(torch.stack(loss)).item()

    return test_outputs, test_labels, test_loss


def eval_epoch(model, test_subset_loader, loss_fn, metrics_fn, device):
    model.eval()
    test_outputs, test_labels, test_loss = model_predict(model, test_subset_loader, loss_fn, device)

    metrics = metrics_fn(test_outputs, test_labels)

    metrics['test_loss'] = test_loss

    return metrics


def calc_accuracy_dict_output(dicts, labels, scores_key):
    scores = [d[scores_key] for d in dicts]
    scores = torch.cat(scores).detach().cpu().numpy()
    predictions = np.argmax(scores, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {'main_metric': acc, 'test_acc': acc}


def calc_accuracy(scores, labels):
    if isinstance(scores, list):
        scores = torch.cat(scores).detach().cpu().numpy()
    predictions = np.argmax(scores, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {'main_metric': acc, 'test_acc': acc}


def calc_acc_on_base(scores, labels, fake_classes_start):
    if isinstance(scores, list):
        scores = torch.cat(scores).detach().cpu().numpy()
    true_samples = labels < fake_classes_start
    scores = scores[true_samples, :fake_classes_start]
    labels = labels[true_samples]
    acc_on_true = calc_accuracy(scores, labels)['main_metric']
    return {'main_metric': acc_on_true, 'test_acc': acc_on_true}


def calc_accuracy_wfake(scores, labels, fake_classes_start):
    if isinstance(scores, list):
        scores = torch.cat(scores).detach().cpu().numpy()
    acc_on_all = calc_accuracy(scores, labels)['main_metric']
    true_samples = labels < fake_classes_start
    scores = scores[true_samples, :fake_classes_start]
    labels = labels[true_samples]
    acc_on_true = calc_accuracy(scores, labels)['main_metric']

    return {'main_metric': acc_on_true, 'acc_on_all': acc_on_all, 'test_acc': acc_on_true}


def calc_accuracy_wfake_head(scores, labels, fake_classes_start):
    orig_head_scores = torch.cat([s[0] for s in scores]).detach().cpu().numpy()
    fake_head_scores = torch.cat([s[1] for s in scores]).detach().cpu().numpy()

    acc_on_all = calc_accuracy(fake_head_scores, labels)['main_metric']
    true_samples = labels < fake_classes_start

    scores = orig_head_scores[true_samples]
    labels = labels[true_samples]
    acc_on_true = calc_accuracy(scores, labels)['main_metric']

    print(f'num true samples: {len(labels)}')
    return {'main_metric': acc_on_true, 'acc_on_all': acc_on_all, 'test_acc': acc_on_true}


def ManMix_rebuild_batch_labels(batch_labels, new_classes_labels, num_pairs_in_batch):
    first_pairs = batch_labels[:num_pairs_in_batch]
    second_pairs = batch_labels[num_pairs_in_batch:2 * num_pairs_in_batch]

    pairs_labels = [new_classes_labels[(first_pairs[i].item(), second_pairs[i].item())] for i in
                    range(num_pairs_in_batch)]
    pairs_labels = torch.tensor(pairs_labels)
    if hasattr(batch_labels, 'device'):
        pairs_labels = pairs_labels.to(batch_labels.device)
    else:
        batch_labels = torch.from_numpy(batch_labels)

    batch_labels = torch.cat([pairs_labels, batch_labels[num_pairs_in_batch * 2:]])
    return batch_labels


def ManMix_rebuild_subset_labels(labels, new_classes_labels, num_pairs_in_batch, batch_size):
    actual_batch_size = batch_size + 2 * num_pairs_in_batch
    batches = []
    for b_id in range(0, len(labels), actual_batch_size):
        batch_labels = labels[b_id: b_id + actual_batch_size]
        batch_labels_rebuilt = ManMix_rebuild_batch_labels(batch_labels, new_classes_labels, num_pairs_in_batch)
        batches.append(batch_labels_rebuilt)

    labels = torch.cat(batches).numpy()
    return labels


def calc_accuracy_wManifoldMix(scores, labels, fake_classes_start, new_classes_labels, num_pairs_in_batch, batch_size):
    if isinstance(scores, list):
        scores = torch.cat(scores).detach().cpu().numpy()

    labels = ManMix_rebuild_subset_labels(labels, new_classes_labels, num_pairs_in_batch, batch_size)

    assert len(labels) == scores.shape[0]

    acc_on_all = calc_accuracy(scores, labels)['main_metric']
    true_samples = labels < fake_classes_start
    scores = scores[true_samples, :fake_classes_start]
    labels = labels[true_samples]
    acc_on_true = calc_accuracy(scores, labels)['main_metric']

    # print(f'num true samples: {len(labels)}')
    return {'main_metric': acc_on_all, 'acc_on_all': acc_on_all, 'test_acc': acc_on_true}


def calc_inst_accuracies(model, test_subset_loaders, loss_fn, metric2monitor):
    # eval part
    with Timer('eval epoch time:'):
        metrics = eval_expandable_model(model, test_subset_loaders, loss_fn=loss_fn,
                                        metric2monitor=metric2monitor)
    return metrics


def ManMix_xent_loss(scores, labels, new_classes_labels, num_pairs_in_batch):
    if isinstance(scores, list):
        scores = torch.cat(scores)

    labels = ManMix_rebuild_batch_labels(labels, new_classes_labels, num_pairs_in_batch)
    assert len(labels) == scores.shape[0]
    return cross_entropy(scores, labels)


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    if len(target.shape) == 1:  # if target is actual labels instead of onehot
        return torch.nn.functional.cross_entropy(input, target)

    # if target is in onehot format
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def loss_xent_wfake_head(scores, labels, fake_classes_start, alpha=1.0):
    orig_head_scores = scores[0]
    aux_head_scores = scores[1]
    aux_head_loss = F.cross_entropy(aux_head_scores, labels)

    true_samples = labels < fake_classes_start
    orig_head_scores = orig_head_scores[true_samples]
    orig_head_labels = labels[true_samples]
    if orig_head_scores.nelement() == 0:
        return aux_head_loss

    orig_head_loss = F.cross_entropy(orig_head_scores, orig_head_labels)

    total_loss = orig_head_loss + alpha * aux_head_loss
    return total_loss


def loss_xent(scores, labels, output_name=None):
    if output_name is not None:
        scores = scores[output_name]

    return cross_entropy(scores, labels)


def cosine_loss(scores, true_labels):
    scores = F.normalize(scores, dim=1)
    labels_one_hot = F.one_hot(true_labels, scores.shape[1])
    loss = -torch.sum(scores * labels_one_hot, dim=1)

    return torch.mean(loss)


def cosine_xent_loss(scored_dict, true_labels, lmbda=0.1):
    scores = scored_dict['embeddings']

    labels_one_hot = F.one_hot(true_labels, scores.shape[1])
    _cosine_loss = torch.mean(-torch.sum(scores * labels_one_hot, dim=1))

    xent_loss = F.cross_entropy(scored_dict['class_scores'], true_labels)

    total_loss = _cosine_loss + lmbda * xent_loss

    return total_loss


def invoke_hooks(model, args, which):
    model.eval()
    if not hasattr(model, which):
        model.train()
        return

    with torch.no_grad():
        for hook in getattr(model, which):
            hook(args)

    model.train()


def freeze_base_model(model, last_layer_name):
    for param in model.parameters():
        param.requires_grad = False

    if last_layer_name == 'fc':
        for param in model.fc.parameters():
            param.requires_grad = True

    if last_layer_name == 'classifier':
        for param in model.classifier.parameters():
            param.requires_grad = True

    trainable_params = get_trainable_params(model)
    return trainable_params


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def get_trainable_params(model):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return trainable_params


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return trainable_params


def load_model(model_path, model, strict=True, model_dict_key='model_state_dict', print_stats=True):
    saved_dict = torch.load(model_path)
    model.load_state_dict(saved_dict[model_dict_key], strict=strict)

    if print_stats:
        print(saved_dict['stats'])

    return model, saved_dict


def create_data_loader_from_dataset(dataset, batch_size, num_workers, shuffle, balance_data=False):
    _, labels = np.unique(dataset.labels, return_inverse=True)
    weights = 1. / np.unique(labels, return_counts=True)[1]
    weights = weights[labels]
    sampler = None
    if balance_data:
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False

    subset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                num_workers=num_workers, sampler=sampler)
    return subset_loader


def create_data_loader(dataset_name, ds_subset, batch_size, num_workers, shuffle=False, offset=0, num_shots=None,
                       transform=None, mix_type=False, sampler_opt=None, get_embeddings_loader=False,
                       fold_id=-1, balance_dataset=True, feature_type='densenet_201_T3'):

    if get_embeddings_loader:

        return ZSLDatasetEmbeddings(dataset_name, feature_type, ds_subset, batch_size, conv_features=True,
                                    split_id=fold_id, offset=offset, shuffle_batch=True, balance_dataset=balance_dataset)


    img_paths, labels, class_attributes = load_xlsa17(dataset_name, ds_subset, return_image_paths=True)
    img_paths = norm_img_paths(dataset_name, img_paths)
    _, labels = np.unique(labels, return_inverse=True)
    weights = 1. / np.unique(labels, return_counts=True)[1]
    weights = weights[labels]
    labels += offset
    if mix_type == 'cutmix':
        subset = CutMix(img_paths, labels, class_attributes, num_shots, transform)
    elif mix_type == 'input_mixup':
        subset = InputMixUp(img_paths, labels, class_attributes, num_shots, transform)
    else:
        subset = CustomedDataset(img_paths, labels, class_attributes, num_shots, transform)

    sampler = None
    batch_sampler = None
    sampler_type = sampler_opt['sampler_type'] if sampler_opt is not None else None
    if sampler_type == 'balanced_sampler':
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False

    if sampler_type == 'batched_balanced_sampler':
        batch_sampler = BatchSampler(WeightedRandomSampler(weights, len(weights)), batch_size, drop_last=True)
        shuffle = False
        batch_size = 1

    elif sampler_type == 'Manifold_batch_sampler':
        batch_sampler = ManifoldMixBatchSampler(labels, sampler_opt['class_pairs'], sampler_opt['batch_size'],
                                                sampler_opt['num_pairs_in_batch'])
        shuffle = False
        batch_size = 1

    # noinspection PyArgumentList
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                                                num_workers=num_workers, sampler=sampler,
                                                batch_sampler=batch_sampler)

    return subset_loader


def draw_shots(samples, membership_mat, num_shots):
    num_classes = membership_mat.shape[0]
    shots = []
    test_samples = []
    test_labels = []
    for c in range(num_classes):
        samples_loc = membership_mat[c, :]
        class_samples = samples[samples_loc]
        perm = np.random.permutation(len(class_samples))
        class_shots = class_samples[perm[:num_shots]]
        class_test_samples = class_samples[perm[num_shots:]]
        class_test_labels = [c] * class_test_samples.shape[0]

        shots.append(class_shots)
        test_samples.append(class_test_samples)
        test_labels.append(class_test_labels)

    shots = np.concatenate(shots, axis=0)
    shots = np.reshape(shots, [num_classes, num_shots, -1])  # new shape is: num_shots X num_classes X f
    classifiers = np.mean(shots, axis=1)  # classes X f
    classifiers = normalize(classifiers, axis=1)

    test_samples = np.concatenate(test_samples, axis=0)
    test_labels = np.concatenate(test_labels)

    return classifiers, test_samples, test_labels


def eval_expandable_model_with_confidence(expandable_model, test_dataset_loaders, test_configs):
    base_set_loader, novel_set_loader = test_dataset_loaders
    expandable_model.eval()

    # with Timer('predict seen time:'):
    base_samples_embeddings, base_labels = model_embed(expandable_model, base_set_loader, expandable_model.device)

    novel_samples_embeddings, novel_labels = model_embed(expandable_model, novel_set_loader, expandable_model.device)

    base_classes = np.unique(base_labels)
    novel_classes = np.unique(novel_labels)
    base_classifiers = expandable_model.get_classifiers(base_classes)

    # build few shot
    num_shots_to_test = test_configs['num_shots']
    num_episodes = test_configs['num_episodes']

    novel_membership_matrix = novel_classes[:, np.newaxis] == novel_labels[np.newaxis, :]
    # shape: (#novel classes, # novel samples)

    results = {}
    for num_shots in num_shots_to_test:
        metrics = {'seen_acc': [], 'unseen_acc': [], 'H': [], 'supervised_acc': [], 'zsl_acc': [],
                   'overall_weighted_acc': []}
        for episode in range(num_episodes):
            novel_classifiers, novel_test_embeddings, novel_test_labels = draw_shots(novel_samples_embeddings,
                                                                                     novel_membership_matrix, num_shots)

            novel_test_labels += len(base_classes)

            all_test_samples = np.concatenate([base_samples_embeddings, novel_test_embeddings], axis=0)
            labels = np.concatenate([base_labels, novel_test_labels])

            base_clfrs_scores = np.matmul(all_test_samples, base_classifiers.T)
            novel_clfrs_scores = np.matmul(all_test_samples, novel_classifiers.T)

            seen_acc, unseen_acc, H = calc_gzsl_accuracies(base_clfrs_scores, novel_clfrs_scores, labels,
                                                           base_classes, novel_classes)

            base_samples_scores = np.matmul(base_samples_embeddings, base_classifiers.T)
            novel_samples_scores = np.matmul(novel_samples_embeddings, novel_classifiers.T)
            supervised_acc = calc_seen_acc(base_samples_scores, base_labels)
            zsl_acc = calc_zsl_acc(novel_samples_scores, novel_labels - len(base_classes))

            all_per_class_acc = calc_weighted_avg((len(base_classes), len(novel_classes)), (seen_acc, unseen_acc))

            metrics['seen_acc'].append(seen_acc)
            metrics['unseen_acc'].append(unseen_acc)
            metrics['H'].append(H)
            metrics['supervised_acc'].append(supervised_acc)
            metrics['overall_weighted_acc'].append(all_per_class_acc)
            metrics['zsl_acc'].append(zsl_acc)

        aggregated_metrics = {}
        for k, v in metrics.items():
            mean, conf = mean_confidence_interval(v)
            aggregated_metrics[k] = mean
            aggregated_metrics[k + ' 95 conf'] = conf

        aggregated_metrics['main_metric'] = aggregated_metrics['H']

        results[num_shots] = aggregated_metrics

    return results


def eval_expandable_model(expandable_model, test_dataset_loaders: List[ZSLDatasetEmbeddings],
                          use_post_bias_correction=False, loss_fn=None, metric2monitor='H', fixed_bias=None,
                          calc_auc=True,
                          scores_key='score_per_class'):
    seen_set_loader, unseen_set_loader = test_dataset_loaders
    expandable_model.eval()

    # with Timer('predict seen time:'):
    seen_samples_scores, seen_labels, loss_seen = model_predict(expandable_model, seen_set_loader, None,
                                                                expandable_model.device, output_key=scores_key)
    if isinstance(seen_samples_scores[0], dict):
        seen_samples_scores = torch.cat([o[scores_key] for o in seen_samples_scores],
                                        dim=0).detach().cpu().numpy()

    torch.cuda.empty_cache()
    # with Timer('predict unseen time:'):
    unseen_samples_scores, unseen_labels, loss_unseen = model_predict(expandable_model, unseen_set_loader, None,
                                                                      expandable_model.device, output_key=scores_key)
    if isinstance(unseen_samples_scores[0], dict):
        unseen_samples_scores = torch.cat([o[scores_key] for o in unseen_samples_scores],
                                          dim=0).detach().cpu().numpy()
    torch.cuda.empty_cache()

    expandable_model.train()

    samples_scores = np.concatenate([seen_samples_scores, unseen_samples_scores], axis=0)
    labels = np.concatenate([seen_labels, unseen_labels])

    seen_classes, unseen_classes = expandable_model.seen_classes, expandable_model.unseen_classes
    num_seen_classes, num_novel_classes = len(seen_classes), len(unseen_classes)

    seen_clfrs_scores = samples_scores[:, seen_classes]
    unseen_clfrs_scores = samples_scores[:, unseen_classes]

    # with Timer('calc accs time:'):
    seen_acc, unseen_acc, hm = calc_gzsl_accuracies(seen_clfrs_scores, unseen_clfrs_scores, labels,
                                                    seen_classes, unseen_classes)

    supervised_acc = calc_seen_acc(seen_samples_scores[:, seen_classes], seen_labels)
    zsl_acc = calc_zsl_acc(unseen_samples_scores, unseen_labels)

    all_per_class_acc = calc_weighted_avg((num_seen_classes, num_novel_classes), (seen_acc, unseen_acc))

    all_loss = calc_weighted_avg((len(seen_labels), len(unseen_labels)), (loss_seen, loss_unseen))

    gzsl_ap50 = calc_avg_percision_at_k(labels, samples_scores)
    zsl_ap50 = calc_avg_percision_at_k(unseen_labels, unseen_clfrs_scores[len(seen_labels):, :])

    metrics = {'acc_seen': seen_acc, 'acc_novel': unseen_acc, 'H': hm, 'acc_zs': zsl_acc, 'gzsl_ap50': gzsl_ap50,
               'zsl_ap50': zsl_ap50, 'supervised_acc': supervised_acc, 'main_metric': hm,
               'overall_weighted_acc': all_per_class_acc, 'loss_seen': loss_seen, 'loss_unseen': loss_unseen,
               'test_loss': all_loss
               }

    if use_post_bias_correction:
        gzsl_metrics = calc_gzsl_metrics(seen_clfrs_scores, unseen_clfrs_scores, labels,
                                         seen_classes, unseen_classes, calc_auc, fixed_bias)
        metrics.update(gzsl_metrics)

    metrics['main_metric'] = metrics[metric2monitor]
    return metrics


def eval_ensemble(models, biasses, data_loaders, alpha, use_post_bias_correction=False, use_sftmx=False):
    seen_set_loader, unseen_set_loader = data_loaders
    scores_key = 'score_per_class'
    seen_classes, unseen_classes = models[0].seen_classes, models[0].unseen_classes

    def get_scores(model, bias):
        model.eval()

        # with Timer('predict seen time:'):
        _seen_samples_scores, _seen_labels, loss_seen = model_predict(model, seen_set_loader, None,
                                                                    model.device, output_key=scores_key)
        if isinstance(_seen_samples_scores[0], dict):
            _seen_samples_scores = torch.cat([o[scores_key] for o in _seen_samples_scores],dim=0).detach().cpu().numpy()

        torch.cuda.empty_cache()
        # with Timer('predict unseen time:'):
        _unseen_samples_scores, _unseen_labels, loss_unseen = model_predict(model, unseen_set_loader, None,
                                                                          model.device, output_key=scores_key)
        if isinstance(_unseen_samples_scores[0], dict):
            _unseen_samples_scores = torch.cat([o[scores_key] for o in _unseen_samples_scores],
                                               dim=0).detach().cpu().numpy()
        torch.cuda.empty_cache()

        _seen_samples_scores[:, unseen_classes] += bias
        _unseen_samples_scores[:, unseen_classes] += bias
        if use_sftmx:
            _seen_samples_scores = softmax(_seen_samples_scores, axis=1)
            _unseen_samples_scores = softmax(_unseen_samples_scores, axis=1)
        return _seen_samples_scores, _unseen_samples_scores, (_seen_labels, _unseen_labels)

    seen_samples_scores0, unseen_samples_scores0, (seen_labels, unseen_labels) = get_scores(models[0], biasses[0])
    seen_samples_scores1, unseen_samples_scores1, _ = get_scores(models[1], biasses[1])

    seen_samples_scores = seen_samples_scores0 * (1-alpha) + seen_samples_scores1 * alpha
    unseen_samples_scores = unseen_samples_scores0 * (1-alpha) + unseen_samples_scores1 * alpha

    samples_scores = np.concatenate([seen_samples_scores, unseen_samples_scores], axis=0)
    labels = np.concatenate([seen_labels, unseen_labels])

    num_seen_classes, num_novel_classes = len(seen_classes), len(unseen_classes)

    seen_clfrs_scores = samples_scores[:, seen_classes]
    unseen_clfrs_scores = samples_scores[:, unseen_classes]

    # with Timer('calc accs time:'):
    seen_acc, unseen_acc, hm = calc_gzsl_accuracies(seen_clfrs_scores, unseen_clfrs_scores, labels,
                                                    seen_classes, unseen_classes)

    supervised_acc = calc_seen_acc(seen_samples_scores[:, seen_classes], seen_labels)
    zsl_acc = calc_zsl_acc(unseen_samples_scores, unseen_labels)

    all_per_class_acc = calc_weighted_avg((num_seen_classes, num_novel_classes), (seen_acc, unseen_acc))

    all_loss = 0

    gzsl_ap50 = calc_avg_percision_at_k(labels, samples_scores)
    zsl_ap50 = calc_avg_percision_at_k(unseen_labels, unseen_clfrs_scores[len(seen_labels):, :])

    metrics = {'acc_seen': seen_acc, 'acc_novel': unseen_acc, 'H': hm, 'acc_zs': zsl_acc, 'gzsl_ap50': gzsl_ap50,
               'zsl_ap50': zsl_ap50, 'supervised_acc': supervised_acc, 'main_metric': hm,
               'overall_weighted_acc': all_per_class_acc, 'loss_seen': 0, 'loss_unseen': 0,
               'test_loss': all_loss
               }

    if use_post_bias_correction:
        gzsl_metrics = calc_gzsl_metrics(seen_clfrs_scores, unseen_clfrs_scores, labels,
                                         seen_classes, unseen_classes, True, None)
        metrics.update(gzsl_metrics)

    return metrics


def np_circular_push(arr, new_value):
    arr[:-1] = arr[1:]
    arr[-1] = new_value
    return arr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, optimizer, scheduler, dataset_loaders, loss_fn, metrics_fn, exp_logger, best_metrics,
                model_path, device, start_epoch=0, num_epochs=100, predicate='max', metric2monitor='H',
                custom_eval_epoch=eval_expandable_model):
    if best_metrics is None:
        best_metrics = {'main_metric': 0}
    predicate = {'max': (lambda a, b: a > b), 'min': (lambda a, b: a < b)}.get(predicate, predicate)
    train_subset_loader, test_subset_loader = dataset_loaders
    eval_time = train_time = 1.
    time_per_epoch_est = np.zeros(5)
    base_only_eval = True
    if isinstance(test_subset_loader, list):
        base_only_eval = False
    for epoch in range(start_epoch, num_epochs):

        # eval part
        with Timer('eval epoch time:') as t:
            if base_only_eval:
                metrics = eval_epoch(model, test_subset_loader, loss_fn, metrics_fn, device)
            else:
                # metrics = {'main_metric': 0}
                metrics = custom_eval_epoch(model, test_subset_loader, loss_fn=loss_fn, metric2monitor=metric2monitor)

            eval_time = t.get_time()

        # logging stuff
        main_metric = metrics['main_metric']
        stats = {'epoch': epoch, **metrics}
        if predicate(main_metric, best_metrics['main_metric']):
            best_metrics = stats
            torch.save({'model_state_dict': model.state_dict(), 'stats': stats,
                        'optimizer_state_dict': optimizer.state_dict()},
                       model_path)
        print('-' * 30)
        print(stats)
        print(f'best so far {best_metrics}')
        if exp_logger is not None:
            if 'lr' in exp_logger.headers:
                stats['lr'] = get_lr(optimizer)
            exp_logger.log(stats)
        num_epochs_left = num_epochs - epoch
        time_per_epoch_est = np_circular_push(time_per_epoch_est, train_time + eval_time)
        latency_factor = 5 / (epoch + 1) if epoch < 5 else 1
        eta_str = get_human_readable_time(num_epochs_left * np.mean(time_per_epoch_est) * latency_factor)
        print(f'ETA: {eta_str}')

        # train part
        with Timer('train epoch time:') as t:
            train_epoch(model, optimizer, train_subset_loader, loss_fn, device)
            train_time = t.get_time()

        if scheduler is not None:
            scheduler.step()

        invoke_hooks(model, None, 'after_train_epoch_hooks')

    saved_dict = torch.load(model_path)
    model.load_state_dict(saved_dict['model_state_dict'])
    try:
        optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
    except ValueError as e:
        print('optimizer not loaded')
    return best_metrics


def calc_predictive_dist_acc(test_outputs, true_labels, cov_mats, num_draws=10):
    samples_logits = torch.cat([o['score_per_class'] for o in test_outputs], dim=0).detach().cpu().numpy()

    samples_embeddings = torch.cat([o['rff_embeddings'] for o in test_outputs], dim=0)

    with torch.no_grad():
        samples_vars = torch.einsum('sf,kfd,sd->sk', samples_embeddings, cov_mats, samples_embeddings)
        samples_vars = samples_vars.detach().cpu().numpy()

    predictive_dists = []
    for i in range(samples_logits.shape[0]):
        samples_dist = np.random.multivariate_normal(samples_logits[i], np.diagflat(samples_vars[i]), size=num_draws)

        predictive_dist = np.mean(softmax(samples_dist, axis=1), axis=0)
        predictive_dists.append(predictive_dist)

    predictive_dists = np.array(predictive_dists)
    accuracy = calc_accuracy(predictive_dists, true_labels)['test_acc']
    return accuracy


def mean_field_correction(logits, variances, mean_field_factor):
    logits_scale = np.sqrt(1. + variances * mean_field_factor)

    logits = logits / logits_scale[:, np.newaxis]
    return logits


def eval_sngp_ood(model, test_loaders, loss_fn=None, metric2monitor='test_acc', mean_field_factor=12.5):
    base_set_loader, novel_set_loader = test_loaders
    model.eval()

    # with Timer('predict seen time:'):
    base_samples_outputs, base_labels, _ = model_predict(model, base_set_loader, None, model.device)
    base_samples_logits = torch.cat([o['score_per_class'] for o in base_samples_outputs], dim=0).detach().cpu().numpy()

    test_loss = cross_entropy(torch.from_numpy(base_samples_logits), torch.from_numpy(base_labels)).item()

    # with Timer('predict unseen time:'):
    novel_samples_outputs, novel_labels, _ = model_predict(model, novel_set_loader, None, model.device)
    novel_samples_logits = torch.cat([o['score_per_class'] for o in novel_samples_outputs],
                                     dim=0).detach().cpu().numpy()

    all_outputs = base_samples_outputs + novel_samples_outputs
    all_samples_variances = torch.cat([o['variances'] for o in all_outputs]).detach().cpu().numpy()
    K = len(np.unique(base_labels))
    all_samples_logits = np.concatenate([base_samples_logits, novel_samples_logits])
    corrected_logits = mean_field_correction(all_samples_logits, all_samples_variances, mean_field_factor)

    uncertainty_scores = K / (K + np.sum(np.exp(corrected_logits), axis=1))

    ood_labels = np.array([0] * len(base_labels) + [1] * len(novel_labels))

    ood_auc_score = roc_auc_score(ood_labels, uncertainty_scores)

    base_acc_score = calc_accuracy(base_samples_logits, base_labels)['main_metric']

    res = {'ood_auc_score': ood_auc_score, 'test_acc': base_acc_score,
           'test_loss': test_loss}
    res['main_metric'] = res[metric2monitor]
    return res


def calc_seen_acc(scores, labels):
    label_set = np.unique(labels)
    predictions = label_set[np.argmax(scores[:, label_set], axis=1)]

    acc = accuracy_score(labels, predictions)
    return acc


import misc.metrics_utils


def calc_zsl_acc(scores, labels):
    label_set = np.unique(labels)
    unseen_predictions = label_set[np.argmax(scores[:, label_set], axis=1)]
    zsl_acc = misc.metrics_utils.calc_accuracy(labels, unseen_predictions, per_class_acc=True)
    return zsl_acc


def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape


if __name__ == '__main__':
    dataset_name = 'CUB'
    ds_subset = 'all_train_loc'
    trans = get_base_transforms(224, 224)
    dl = create_data_loader(dataset_name, ds_subset, batch_size=32, num_workers=0, shuffle=False, offset=0,
                            num_shots=None,
                            transform=trans, mix_type='input_mixup')

    batch_size = 16
    for b_id, batch in enumerate(dl):
        batch = batch[0]
        save_folder = './aug_vis/vanilla'
        os.makedirs(save_folder, exist_ok=True)
        save_image(batch[:batch_size], os.path.join(save_folder, f'batch{b_id}_input_mixup.jpg'), nrow=4)
        break
