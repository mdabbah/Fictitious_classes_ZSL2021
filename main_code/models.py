import torch
import torch.nn as nn
import torch.nn.functional as F

from DAZLE.dazle_model import Dazle
import numpy as np
from torchvision.transforms import transforms

from misc.backbones import Extractor
from misc.train_eval_utils import get_output_shape


class DazleCC(Dazle):

    def __init__(self, features_dim, w2v_dim, lambda_, init_w2v_att, classes_attributes, seen_classes, unseen_classes,
                 num_decoders, use_dropout, device, sphere_size=10., bias=1, use_CC=True, cal_unseen=True,
                 dropout_rate=0.05):

        super(DazleCC, self).__init__(features_dim, w2v_dim, lambda_, init_w2v_att, True, classes_attributes,
                                      seen_classes, unseen_classes, normalize_V=False, num_decoders=num_decoders,
                                      summarizeing_op='sum', translation_op='no_translation', use_dropout=use_dropout,
                                      device=device, bias=bias, cal_unseen=cal_unseen, norm_instances=False,
                                      drop_rate=dropout_rate)

        self.sphere_size = nn.Parameter(torch.tensor([sphere_size]))
        self.use_cc = use_CC

    def forward(self, visual_features):

        shape = visual_features.shape
        visual_features = visual_features.reshape(shape[0], shape[1], shape[2] * shape[3])
        visual_features = F.normalize(visual_features, dim=1)

        V_n = self.compute_V()

        visual_attributes = []
        for transformer_decoder in self.visual_decoders:
            visual_attributes.append(transformer_decoder(visual_features, V_n))

        visual_attributes = self.summarizing_op(visual_attributes)

        attribute_scores = self.bilinear_score(visual_attributes, V_n)
        attribute_existence = self.bilinear_existance(visual_attributes, V_n)
        instance_descriptor = attribute_scores * attribute_existence

        if self.use_dropout:
            instance_descriptor = self.visual_dropout(instance_descriptor)

        if self.use_cc:
            instance_descriptor = F.normalize(instance_descriptor, dim=1)

        attribute_scores_in_class = self.classes_attributes

        score_per_class = torch.einsum('ki,bi->bk', attribute_scores_in_class, instance_descriptor)

        if self.use_cc:
            score_per_class = score_per_class * self.sphere_size

        score_per_class = score_per_class + self.vec_bias

        package = {'score_per_class': score_per_class,
                   'attribute_existence': attribute_existence,
                   'attribute_visual_representative': visual_attributes,
                   'attribute_score': attribute_scores,
                   'instance_descriptor': instance_descriptor,
                   'class_signatures': attribute_scores_in_class,
                   'S_pp': score_per_class}

        return package


class DazleComposer(Dazle):

    def __init__(self, features_dim, w2v_dim, lambda_, init_w2v_att, classes_attributes, seen_classes, unseen_classes,
                 num_decoders, use_dropout, device, bias=1, cal_unseen=True, dropout_rate=0.05, c2c_mat=None,
                 num_samples_per_novel=1, num_novel_in_batch=1, cache_all_classes=False):

        super(DazleComposer, self).__init__(features_dim, w2v_dim, lambda_, init_w2v_att, True, classes_attributes,
                                            seen_classes, unseen_classes, normalize_V=False, num_decoders=num_decoders,
                                            summarizeing_op='sum', translation_op='no_translation',
                                            use_dropout=use_dropout,
                                            device=device, bias=bias, cal_unseen=cal_unseen, norm_instances=False,
                                            drop_rate=dropout_rate)

        if c2c_mat is None:
            raise ValueError("use the original model if you don't wan't to compose")

        self.num_novel_in_batch = num_novel_in_batch
        self.num_novel_classes = len(self.unseen_classes)
        self.c2c_mat = c2c_mat
        self.num_samples_per_novel = num_samples_per_novel

        self.cache_samples = np.zeros([self.c2c_mat.shape[0], self.num_samples_per_novel,
                                       self.num_attributes, features_dim])
        self.class_attribute_counters = np.zeros([self.c2c_mat.shape[0], self.num_attributes]).astype(int)

        cache_classes = self.unseen_classes
        if cache_all_classes:
            cache_classes = np.arange(len(self.seen_classes) + len(self.unseen_classes))

        self.cache_labels = np.repeat(cache_classes[:, np.newaxis], [self.num_samples_per_novel])
        self.curr_batch_cache_labels = None

        self.after_train_step_hooks = [self.update_novel_cache]

    def forward(self, visual_features):

        shape = visual_features.shape
        visual_features = visual_features.reshape(shape[0], shape[1], shape[2] * shape[3])
        visual_features = F.normalize(visual_features, dim=1)

        V_n = self.compute_V()

        visual_attributes = []
        for transformer_decoder in self.visual_decoders:
            visual_attributes.append(transformer_decoder(visual_features, V_n))

        visual_attributes = self.summarizing_op(visual_attributes)

        if self.training and self.num_novel_in_batch > 0:
            novel_samples = self.pick_novel_samples_from_cache(self.num_novel_in_batch)
            if novel_samples is not None:
                visual_attributes = torch.cat([visual_attributes, novel_samples], dim=0)

        attribute_scores = self.bilinear_score(visual_attributes, V_n)
        attribute_existence = self.bilinear_existance(visual_attributes, V_n)
        instance_descriptor = attribute_scores * attribute_existence

        if self.use_dropout:
            instance_descriptor = self.visual_dropout(instance_descriptor)

        attribute_scores_in_class = self.classes_attributes

        score_per_class = torch.einsum('ki,bi->bk', attribute_scores_in_class, instance_descriptor)

        score_per_class = score_per_class + self.vec_bias

        package = {'score_per_class': score_per_class,
                   'attribute_existence': attribute_existence,
                   'visual_attributes': visual_attributes,
                   'attribute_score': attribute_scores,
                   'instance_descriptor': instance_descriptor,
                   'class_signatures': attribute_scores_in_class,
                   'S_pp': score_per_class}

        return package

    def compute_aug_cross_entropy(self, in_package, score_type='score_per_class'):
        batch_label = in_package['batch_label']
        score_per_class = in_package[score_type]

        Labels = batch_label
        # Labels = self.class_sim_distributions[in_package['batch_label_numeric']]

        score_per_class = score_per_class - self.vec_bias
        if self.bias == 0 and self.num_novel_in_batch == 0:
            if self.training:
                Prob = F.log_softmax(score_per_class[:, self.seen_classes], dim=1)
                Labels = Labels[:, self.seen_classes]
            else:
                Prob = F.log_softmax(score_per_class, dim=1)
        else:
            Prob = F.log_softmax(score_per_class, dim=1)

        loss = -torch.einsum('bk,bk->b', Prob, Labels)
        loss = torch.mean(loss)
        return loss

    def compute_loss(self, in_package):
        if self.curr_batch_cache_labels is not None:
            batch_labels = in_package['batch_label']
            in_package['batch_label'] = torch.cat([batch_labels, self.curr_batch_cache_labels])
            self.curr_batch_cache_labels = None

            assert in_package['score_per_class'].shape[0] == len(in_package['batch_label'])

        return super(DazleComposer, self).compute_loss(in_package)

    def reset_cache(self):
        self.class_attribute_counters[:] = 0
        self.cache_samples[:] = 0

    def update_novel_cache(self, hook_args):

        batch_visual_attributes = hook_args['batch_outputs']['visual_attributes']  # shape: (b,n,f)
        batch_labels = hook_args['batch_labels']  # shape: b

        batch_visual_attributes = batch_visual_attributes.detach().cpu().numpy()
        batch_labels = batch_labels.detach().cpu().numpy()

        # this matrix will tell us which incoming samples want to donate which attributes to which novel classes
        donations = self.c2c_mat[:, :, np.newaxis] == batch_labels[np.newaxis, np.newaxis, :]
        indices = np.argwhere(donations)

        target_classes = indices[:, 0]  # the classes which will receive the donation
        target_attributes = indices[:, 1]  # the attributes which will receive the donation
        donor_samples = indices[:, 2]  # which samples want to donate

        # how many donations each (novel class - attribute) pair got
        donations_counter = np.cumsum(donations, axis=-1)
        donations_counter = donations_counter[target_classes, target_attributes, donor_samples]

        place_in_cache = (self.class_attribute_counters[target_classes, target_attributes] + donations_counter - 1) \
                         % self.num_samples_per_novel

        self.cache_samples[target_classes, place_in_cache, target_attributes, :] = \
            batch_visual_attributes[donor_samples, target_attributes, :]

        self.class_attribute_counters[target_classes, target_attributes] += donations_counter

        # old slow code ... but more understandable
        # for s in range(len(batch_labels)):
        #     for n in range(self.num_attributes):
        #         target_classes = self.c2c_mat[:, n] == batch_labels[s]
        #         class_counters = self.class_attribute_counters[target_classes, n]
        #         if len(class_counters) == 0:
        #             continue
        #
        #         place_in_cache = class_counters % self.num_samples_per_novel
        #         self.novel_samples_cache[target_classes, place_in_cache, n, :] = batch_visual_attributes[s, n, :]
        #
        #         class_counters += 1
        #         self.class_attribute_counters[target_classes, n] = class_counters

    def pick_novel_samples_from_cache(self, num_samples_to_add):

        is_cache_full = np.min(self.class_attribute_counters) > self.num_samples_per_novel

        # classes_to_pull_from = np.min(self.class_counters, axis=1) > 0
        #
        # class_sample_limits = np.min(self.class_counters, axis=1)
        #
        # chosen_samples = np.random.randint(0, class_sample_limits[classes_to_pull_from])
        #
        # valid_samples = self.novel_samples_cache[classes_to_pull_from, chosen_samples]
        #
        # chosen_samples = valid_samples

        if is_cache_full:
            chosen_samples_idx = np.random.randint(self.num_novel_classes * self.num_samples_per_novel,
                                                   size=num_samples_to_add)

            chosen_samples = self.cache_samples[chosen_samples_idx // self.num_samples_per_novel,
                                                chosen_samples_idx % self.num_samples_per_novel]

            chosen_labels = self.cache_labels[chosen_samples_idx]
            self.curr_batch_cache_labels = torch.tensor(chosen_labels).to(self.device)
            return torch.tensor(chosen_samples).float().to(device=self.device)

        return None


class DazleAttributeGrouper(Dazle):

    def __init__(self, features_dim, w2v_dim, lambda_, init_w2v_att, classes_attributes, seen_classes, unseen_classes,
                 num_decoders, use_dropout, device, bias=1, cal_unseen=True, dropout_rate=0.05, attribute_groups=None,
                 use_bilinear_gate=True, attention_sftmax_temperature=1.):

        super(DazleAttributeGrouper, self).__init__(features_dim, w2v_dim, lambda_, init_w2v_att, True,
                                                    classes_attributes,
                                                    seen_classes, unseen_classes, normalize_V=False,
                                                    num_decoders=num_decoders,
                                                    summarizeing_op='sum', translation_op='no_translation',
                                                    use_dropout=use_dropout,
                                                    device=device, bias=bias, cal_unseen=cal_unseen,
                                                    norm_instances=False,
                                                    drop_rate=dropout_rate,
                                                    attention_sftmax_temperature=attention_sftmax_temperature)
        self.attribute_groups = attribute_groups
        self.use_bilinear_gate = use_bilinear_gate

    def forward(self, visual_features):

        shape = visual_features.shape
        visual_features = visual_features.reshape(shape[0], shape[1], shape[2] * shape[3])
        visual_features = F.normalize(visual_features, dim=1)

        V_n = self.compute_V()

        visual_attributes = []
        for transformer_decoder in self.visual_decoders:
            visual_attributes.append(transformer_decoder(visual_features, V_n))

        visual_attributes = self.summarizing_op(visual_attributes)

        instance_descriptor = self.bilinear_score(visual_attributes, V_n)

        if self.use_bilinear_gate:
            attribute_existence = self.bilinear_existance(visual_attributes, V_n)
            instance_descriptor = instance_descriptor * attribute_existence

        if self.use_dropout:
            instance_descriptor = self.visual_dropout(instance_descriptor)

        if self.attribute_groups is not None:
            instance_descriptor = self.apply_attribute_grouping(instance_descriptor)

        attribute_scores_in_class = self.classes_attributes

        score_per_class = torch.einsum('ki,bi->bk', attribute_scores_in_class, instance_descriptor)

        score_per_class = score_per_class + self.vec_bias

        package = {'score_per_class': score_per_class,
                   'attribute_visual_representative': visual_attributes,
                   'instance_descriptor': instance_descriptor,
                   'class_signatures': attribute_scores_in_class,
                   'S_pp': score_per_class}

        return package

    def apply_attribute_grouping(self, instance_vector):

        # out_tensor = instance_vector.clone()
        for group in self.attribute_groups:
            instance_vector[:, group] = F.softmax(instance_vector[:, group], dim=1)

        return instance_vector


class DazleFakeComposer(Dazle):

    def __init__(self, features_dim, w2v_dim, lambda_, init_w2v_att, classes_attributes, seen_classes, unseen_classes,
                 num_decoders, use_dropout, device, bias=1, cal_unseen=True, dropout_rate=0.05, num_fake_in_batch=0,
                 vis_drop_rate=0.2, backbone=None, normalize_V=False, normalize_class_defs=True):

        if backbone is not None:
            backbone = Extractor(backbone)
            features_dim = get_output_shape(backbone, [1, *features_dim])[1]

        super(DazleFakeComposer, self).__init__(features_dim, w2v_dim, lambda_, init_w2v_att, True,
                                                classes_attributes, seen_classes, unseen_classes,
                                                normalize_V=normalize_V,
                                                num_decoders=num_decoders, summarizeing_op='sum',
                                                translation_op='no_translation', use_dropout=use_dropout, device=device,
                                                bias=bias, cal_unseen=cal_unseen, norm_instances=False,
                                                drop_rate=dropout_rate, backbone=backbone,
                                                normalize_class_defs=normalize_class_defs)
        self.num_fake_in_batch = num_fake_in_batch
        self.vis_drop_rate = 1 - vis_drop_rate
        self.new_samples_mask = None

        max_num_classes = max(self.num_classes, self.num_seen_classes + num_fake_in_batch)
        self.register_buffer('weight_ce', torch.eye(max_num_classes).float())

        fake_labels = len(self.seen_classes) + torch.arange(self.num_fake_in_batch)
        self.register_buffer('fake_labels', fake_labels)

    def forward(self, visual_features):

        if self.backbone is not None:
            visual_features = self.backbone(visual_features)

        shape = visual_features.shape
        visual_features = visual_features.reshape(shape[0], shape[1], shape[2] * shape[3])
        visual_features = F.normalize(visual_features, dim=1)

        V_n = self.compute_V()

        visual_attributes = []
        for transformer_decoder in self.visual_decoders:
            visual_attributes.append(transformer_decoder(visual_features, V_n))

        visual_attributes = self.summarizing_op(visual_attributes)
        # visual_attributes = visual_attributes[0]
        if self.num_fake_in_batch > 0 and self.training:
            visual_attributes = self.add_fake_samples(visual_attributes)

        instance_score = self.bilinear_score(visual_attributes, V_n)

        attribute_existence = self.bilinear_existance(visual_attributes, V_n)
        instance_descriptor = instance_score * attribute_existence

        if self.use_dropout:
            instance_descriptor = self.visual_dropout(instance_descriptor)

        attribute_scores_in_class = self.classes_attributes

        if self.training and self.bias == 0:
            attribute_scores_in_class = self.classes_attributes[self.seen_classes]

        score_per_class = torch.einsum('ki,bi->bk', attribute_scores_in_class, instance_descriptor)

        if self.bias != 0:
            score_per_class = score_per_class + self.vec_bias

        package = {'score_per_class': score_per_class,
                   'attribute_visual_representative': visual_attributes,
                   'instance_descriptor': instance_descriptor,
                   'class_signatures': attribute_scores_in_class,
                   'S_pp': score_per_class}

        return package

    def add_fake_samples(self, visual_attributes):

        #  visual_attributes shape is (|batch|, #attributes, |embedding|)

        shape = visual_attributes.shape
        # new_samples = torch.zeros((self.num_fake_in_batch, shape[1], shape[2])).float().to(self.device)
        new_samples_mask = torch.zeros((shape[0], shape[1])).float().uniform_() > self.vis_drop_rate
        new_samples_mask = new_samples_mask.float().to(self.device)

        new_samples = visual_attributes * new_samples_mask.unsqueeze(2)

        self.new_samples_mask = new_samples_mask[:self.num_fake_in_batch]

        return torch.cat([visual_attributes, new_samples[:self.num_fake_in_batch]], dim=0)

    def compute_aug_cross_entropy(self, in_package, score_type='score_per_class'):
        batch_label = in_package['batch_label']
        score_per_class = in_package[score_type]

        Labels = batch_label
        if self.bias != 0:
            score_per_class = score_per_class - self.vec_bias

        if self.bias == 0 and self.num_fake_in_batch == 0:
            if self.training:
                Prob = F.log_softmax(score_per_class[:, self.seen_classes], dim=1)
                Labels = Labels[:, self.seen_classes]
            else:
                Prob = F.log_softmax(score_per_class, dim=1)
        else:
            Prob = F.log_softmax(score_per_class, dim=1)

        loss = -torch.einsum('bk,bk->b', Prob, Labels[:, :Prob.shape[1]])
        loss = torch.mean(loss)
        return loss

    def compute_loss(self, in_package):
        if self.new_samples_mask is not None:
            batch_labels = in_package['batch_label']

            new_classifiers = self.classes_attributes[batch_labels]
            new_classifiers = new_classifiers[:self.num_fake_in_batch]
            new_classifiers = new_classifiers * self.new_samples_mask

            instance_descriptor = in_package['instance_descriptor']
            new_classes_scores = torch.einsum('ki,bi->bk', new_classifiers, instance_descriptor)

            in_package['batch_label'] = torch.cat([in_package['batch_label'], self.fake_labels])

            in_package['score_per_class'] = torch.cat([in_package['score_per_class'], new_classes_scores], dim=1)

            self.new_samples_mask = None

            assert in_package['score_per_class'].shape[0] == len(in_package['batch_label'])

        return super(DazleFakeComposer, self).compute_loss(in_package)


class DazleFakeComposerUnlimited(DazleFakeComposer):

    def __init__(self, features_dim, w2v_dim, lambda_, init_w2v_att, classes_attributes, seen_classes, unseen_classes,
                 num_decoders, use_dropout, device, bias=1, cal_unseen=True, dropout_rate=0.05, num_fake_in_batch=0,
                 vis_drop_rate=0.2, backbone=None, normalize_V=False, normalize_class_defs=True):
        super(DazleFakeComposerUnlimited, self).__init__(features_dim=features_dim, w2v_dim=w2v_dim, lambda_=lambda_,
                                                         init_w2v_att=init_w2v_att,
                                                         classes_attributes=classes_attributes,
                                                         seen_classes=seen_classes, unseen_classes=unseen_classes,
                                                         num_decoders=num_decoders, use_dropout=use_dropout,
                                                         device=device, bias=bias, cal_unseen=cal_unseen,
                                                         dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                                                         vis_drop_rate=vis_drop_rate, backbone=backbone,
                                                         normalize_V=normalize_V,
                                                         normalize_class_defs=normalize_class_defs)
        self.augmented_samples_idx = None

        # testing code diff
        self.seenclass = torch.tensor(seen_classes).type(torch.LongTensor)
        self.unseenclass = torch.tensor(unseen_classes).type(torch.LongTensor)

    def add_fake_samples(self, visual_attributes):
        #  visual_attributes shape is (|batch|, #attributes, |embedding|)
        shape = visual_attributes.shape

        new_samples_mask = torch.zeros((self.num_fake_in_batch, shape[1])).float().uniform_() > self.vis_drop_rate
        new_samples_mask = new_samples_mask.float().to(self.device)

        samples_idx = torch.randint(shape[0], size=(self.num_fake_in_batch,)).to(self.device)

        new_samples = visual_attributes[samples_idx] * new_samples_mask.unsqueeze(2)

        self.new_samples_mask = new_samples_mask
        self.augmented_samples_idx = samples_idx

        return torch.cat([visual_attributes, new_samples], dim=0)

    def compute_loss(self, in_package):
        if self.new_samples_mask is not None:
            batch_labels = in_package['batch_label']

            new_classifiers = self.classes_attributes[batch_labels]
            new_classifiers = new_classifiers[self.augmented_samples_idx]
            new_classifiers = new_classifiers * self.new_samples_mask

            instance_descriptor = in_package['instance_descriptor']
            new_classes_scores = torch.einsum('ki,bi->bk', new_classifiers, instance_descriptor)

            in_package['batch_label'] = torch.cat([in_package['batch_label'], self.fake_labels])

            in_package['score_per_class'] = torch.cat([in_package['score_per_class'], new_classes_scores], dim=1)

            self.new_samples_mask = None
            self.augmented_samples_idx = None

            assert in_package['score_per_class'].shape[0] == len(in_package['batch_label'])

        return super(DazleFakeComposer, self).compute_loss(in_package)


class DazleFakeMixer(Dazle):

    def __init__(self, features_dim, w2v_dim, lambda_, init_w2v_att, classes_attributes, seen_classes, unseen_classes,
                 num_decoders, use_dropout, device, bias=1, cal_unseen=True, dropout_rate=0.05, num_fake_in_batch=12,
                 backbone=None, class_pairs=None):
        if backbone is not None:
            backbone = Extractor(backbone)
            features_dim = get_output_shape(backbone, [1, *features_dim])[1]

        super(DazleFakeMixer, self).__init__(features_dim, w2v_dim, lambda_, init_w2v_att, True,
                                             classes_attributes, seen_classes, unseen_classes, normalize_V=False,
                                             num_decoders=num_decoders, summarizeing_op='sum',
                                             translation_op='no_translation', use_dropout=use_dropout, device=device,
                                             bias=bias, cal_unseen=cal_unseen, norm_instances=False,
                                             drop_rate=dropout_rate, backbone=backbone)

        self.num_fake_in_batch = 0

        if class_pairs is not None:
            self.num_fake_in_batch = num_fake_in_batch

            max_num_classes = max(self.num_classes, self.num_seen_classes + len(class_pairs))
            self.register_buffer('weight_ce', torch.eye(max_num_classes).float())

            class_pairs = np.array(class_pairs)
            self.pairs_dict = {(p1, p2): i + self.num_seen_classes for i, (p1, p2) in enumerate(class_pairs)}
            p1_classes_attrs = classes_attributes[class_pairs[:, 0], :]
            p2_classes_attrs = classes_attributes[class_pairs[:, 1], :]

            fake_classes_attributes = (p1_classes_attrs + p2_classes_attrs) * 0.5
            fake_classes_attributes = F.normalize(torch.from_numpy(fake_classes_attributes), dim=1)
            self.register_buffer('fake_classes_attributes', fake_classes_attributes)

    def forward(self, x):

        if self.training and self.num_fake_in_batch > 0:
            w, h = x.shape[-2:]
            p1_samples = transforms.F.resize(x[:self.num_fake_in_batch], [w, h // 2])
            p2_samples = transforms.F.resize(x[self.num_fake_in_batch: self.num_fake_in_batch * 2], [w, h - h // 2])
            new_samples = torch.cat([p1_samples, p2_samples], dim=-1)
            x = torch.cat([x[self.num_fake_in_batch * 2:], new_samples])

        if self.backbone is not None:
            visual_features = self.backbone(x)
        else:
            visual_features = x

        shape = visual_features.shape
        visual_features = visual_features.reshape(shape[0], shape[1], shape[2] * shape[3])
        visual_features = F.normalize(visual_features, dim=1)

        V_n = self.compute_V()

        visual_attributes = []
        for transformer_decoder in self.visual_decoders:
            visual_attributes.append(transformer_decoder(visual_features, V_n))

        visual_attributes = self.summarizing_op(visual_attributes)

        instance_score = self.bilinear_score(visual_attributes, V_n)

        attribute_existence = self.bilinear_existance(visual_attributes, V_n)
        instance_descriptor = instance_score * attribute_existence

        if self.use_dropout:
            instance_descriptor = self.visual_dropout(instance_descriptor)

        attribute_scores_in_class = self.classes_attributes

        if self.training and self.num_fake_in_batch > 0:
            attribute_scores_in_class = torch.cat([self.classes_attributes[self.seen_classes],
                                                   self.fake_classes_attributes], dim=0)

        score_per_class = torch.einsum('ki,bi->bk', attribute_scores_in_class, instance_descriptor)

        if self.bias != 0:
            score_per_class = score_per_class + self.vec_bias

        package = {'score_per_class': score_per_class,
                   'attribute_visual_representative': visual_attributes,
                   'instance_descriptor': instance_descriptor,
                   'class_signatures': attribute_scores_in_class,
                   'S_pp': score_per_class}

        return package

    def compute_aug_cross_entropy(self, in_package, score_type='score_per_class'):
        batch_label = in_package['batch_label']
        score_per_class = in_package[score_type]

        Labels = batch_label
        if self.bias != 0:
            score_per_class = score_per_class - self.vec_bias

        if self.bias == 0 and self.num_fake_in_batch == 0:
            if self.training:
                Prob = F.log_softmax(score_per_class[:, self.seen_classes], dim=1)
                Labels = Labels[:, self.seen_classes]
            else:
                Prob = F.log_softmax(score_per_class, dim=1)
        else:
            Prob = F.log_softmax(score_per_class, dim=1)

        loss = -torch.einsum('bk,bk->b', Prob, Labels[:, :Prob.shape[1]])
        loss = torch.mean(loss)
        return loss

    def compute_loss(self, in_package):

        if self.training and self.num_fake_in_batch > 0:
            batch_labels = in_package['batch_label']
            p1_labels = batch_labels[:self.num_fake_in_batch]
            p2_labels = batch_labels[self.num_fake_in_batch:2 * self.num_fake_in_batch]
            fake_labels = torch.tensor(
                [self.pairs_dict[(p1.item(), p2.item())] for p1, p2 in zip(p1_labels, p2_labels)]).long().to(
                self.device)

            batch_labels = torch.cat([batch_labels[self.num_fake_in_batch * 2:], fake_labels])
            in_package['batch_label'] = batch_labels
            assert in_package['score_per_class'].shape[0] == len(in_package['batch_label'])

        return super(DazleFakeMixer, self).compute_loss(in_package)


class DazleFakeComposerUnlimitedScalarAug(DazleFakeComposerUnlimited):

    def add_fake_samples(self, visual_attributes):
        #  visual_attributes shape is (|batch|, #attributes, |embedding|)
        shape = visual_attributes.shape

        new_samples_mask = torch.zeros((self.num_fake_in_batch, shape[1])).float().uniform_()
        new_samples_mask = new_samples_mask.float().to(self.device)

        samples_idx = torch.randint(shape[0], size=(self.num_fake_in_batch,)).to(self.device)

        new_samples = visual_attributes[samples_idx] * new_samples_mask.unsqueeze(2)

        self.new_samples_mask = new_samples_mask
        self.augmented_samples_idx = samples_idx

        return torch.cat([visual_attributes, new_samples], dim=0)


class ensamble_models():
    pass
