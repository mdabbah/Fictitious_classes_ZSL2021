import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Attribute_visual_atten(torch.nn.Module):
    def __init__(self, features_dim, w2v_dim, softmax_temperature=1.):
        super(Attribute_visual_atten, self).__init__()
        self.dim_f = features_dim
        self.dim_v = w2v_dim
        self.temperature = softmax_temperature

        self.W_alpha = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v, self.dim_f)),
                                    requires_grad=True)

    def forward(self, visual_features, v):
        attribute2region_atten = torch.einsum('iv,vf,bfr->bir', v, self.W_alpha, visual_features)
        attribute2region_atten = F.softmax(attribute2region_atten * self.temperature, dim=-1)
        # h, attribute visual features
        visual_attributes = torch.einsum('bir,bfr->bif', attribute2region_atten, visual_features)

        return visual_attributes


class BilinearLayer(torch.nn.Module):
    def __init__(self, features_dim, w2v_dim, init_type, output_activation=None):
        super(BilinearLayer, self).__init__()
        self.dim_f = features_dim
        self.dim_v = w2v_dim

        if init_type == 'zero':
            self.W = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v, self.dim_f)),
                                  requires_grad=True)
        elif init_type == 'normal':
            self.W = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v, self.dim_f)),
                                  requires_grad=True)

        activations = {'sigmoid': torch.sigmoid, 'relu': torch.relu, 'elu': F.elu}
        self.output_activation = activations.get(output_activation, None)

    def forward(self, visual_attributes, v):

        score = torch.einsum('iv,vf,bif->bi', v, self.W, visual_attributes)
        if self.output_activation:
            score = self.output_activation(score)

        return score


class Dazle(torch.nn.Module):

    def __init__(self, features_dim, w2v_dim, lambda_, init_w2v_att, trainable_w2v,
                 classes_attributes, seen_classes, unseen_classes, normalize_V=False,
                 num_decoders=(1, 0), summarizeing_op='sum', translation_op='no_translation', use_dropout=False,
                 device='cpu',
                 bias=1, cal_unseen=True, norm_instances=False, gt_class_articles=None, backbone=None,
                 drop_rate=0.05, attention_sftmax_temperature=1., normalize_class_defs=True):
        super(Dazle, self).__init__()

        self.backbone = backbone


        self.normalize_class_defs = normalize_class_defs

        self.device = device
        self.dim_f = features_dim
        self.dim_v = w2v_dim
        self.dim_class_desc = classes_attributes.shape[-1]
        self.lambda_ = lambda_
        self.num_attributes = init_w2v_att.shape[0]
        self.cal_unseen = cal_unseen
        self.attention_sftmax_temperature = attention_sftmax_temperature

        self.normalize_V = normalize_V
        assert translation_op in ('no_translation', 'direct_translation', 'weighted_translation', 's2s_translation',
                                  'article_generation_translation')
        self.translation_op = translation_op

        self.classes_attributes_raw = nn.Parameter(torch.tensor(classes_attributes).float(), requires_grad=False)
        self.classes_attributes = torch.tensor(classes_attributes)
        if len(classes_attributes.shape) == 2 and normalize_class_defs:
            self.classes_attributes = F.normalize(self.classes_attributes, dim=1).float()  # should be classes X attributes
        # self.classes_attributes = torch.tensor(classes_attributes)  # should be classes X attributes
        # self.classes_attributes = F.normalize(self.classes_attributes - torch.mean(self.classes_attributes, dim=0), dim=1)
        self.classes_attributes = nn.Parameter(self.classes_attributes, requires_grad=False)

        self.seen_classes = seen_classes.squeeze().astype(int)
        self.unseen_classes = unseen_classes.squeeze().astype(int)
        self.num_classes = self.seen_classes.shape[0] + self.unseen_classes.shape[0]
        self.num_seen_classes = len(self.seen_classes)
        self.num_novel_classes = len(self.unseen_classes)

        self.bias = bias
        self.mask_bias = np.ones((1, self.num_classes)) * bias
        self.mask_bias[:, self.seen_classes] *= -self.bias
        self.mask_bias = nn.Parameter(torch.tensor(self.mask_bias).float(), requires_grad=False)
        self.vec_bias = self.mask_bias

        self.register_buffer('weight_ce', torch.eye(self.num_classes).float())


        if init_w2v_att is None:
            self.V = nn.Parameter(nn.init.normal_(torch.empty(self.num_attributes, self.dim_v)), requires_grad=True)
        else:
            self.init_w2v_att = F.normalize(torch.tensor(init_w2v_att)).float()
            self.V = nn.Parameter(self.init_w2v_att.clone(), requires_grad=trainable_w2v)

        self.num_decoders = num_decoders
        # assert num_decoders >= 1
        self.visual_decoders = []
        for idx in range(num_decoders[0]):
            transformer_decoder = Attribute_visual_atten(features_dim, w2v_dim, self.attention_sftmax_temperature)
            self.__setattr__(f'transformer_decoder{idx}', transformer_decoder)
            self.visual_decoders.append(transformer_decoder)

        if summarizeing_op == 'sum':
            self.summarizing_op = lambda tensor_list: torch.sum(torch.stack(tensor_list), dim=0)
        elif summarizeing_op == 'mean':
            self.summarizing_op = lambda tensor_list: torch.mean(torch.stack(tensor_list), dim=0)

        self.bilinear_score = BilinearLayer(features_dim, w2v_dim, 'normal')
        self.bilinear_existance = BilinearLayer(features_dim, w2v_dim, 'zero', 'sigmoid')

        if translation_op == 'weighted_translation':
            self.W_translation = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v, self.dim_class_desc)),
                                              requires_grad=True)

        if translation_op == 'article_generation_translation':
            # self.class_lin = nn.Linear(self.dim_class_desc, self.dim_v)
            self.W_translation = nn.Parameter(
                nn.init.normal_(torch.empty(self.num_attributes, self.dim_v, self.dim_class_desc)),
                requires_grad=True)
            if gt_class_articles is not None:
                self.register_buffer('gt_class_articles', F.normalize(torch.tensor(gt_class_articles), dim=-1))

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.visual_dropout = torch.nn.Dropout(drop_rate)

        self.norm_instances = norm_instances



    def compute_V(self):
        if self.normalize_V:
            V_n = F.normalize(self.V)
        else:
            V_n = self.V
        return V_n

    def forward(self, visual_features):

        if self.backbone:
            visual_features = self.backbone(visual_features)

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

        class_articles = None
        if self.norm_instances:
            instance_descriptor = F.normalize(instance_descriptor, dim=1)

        if self.translation_op == 'direct_translation':
            attribute_scores_in_class = torch.einsum('iv,kv->ki', V_n, self.classes_attributes)
            attribute_scores_in_class = F.normalize(attribute_scores_in_class, dim=1)
        elif self.translation_op == 'no_translation':
            attribute_scores_in_class = self.classes_attributes
        elif self.translation_op == 'weighted_translation':
            attribute_scores_in_class = torch.einsum('iv,vd,kd->ki', V_n, self.W_translation, self.classes_attributes)
            attribute_scores_in_class = F.normalize(attribute_scores_in_class, dim=1)
        elif self.translation_op == 's2s_translation':
            attribute_scores_in_class = torch.einsum('iv,kiv->ki', V_n, self.classes_attributes)
            attribute_scores_in_class = F.normalize(attribute_scores_in_class, dim=1)
        elif self.translation_op == 'article_generation_translation':
            # class_lin_out = self.class_lin(self.classes_attributes)
            # att_lin_out = self.att_lin(V_n)
            class_articles = torch.einsum('kj,ivj->kiv ', self.classes_attributes, self.W_translation)
            attribute_scores_in_class = torch.einsum('iv,kiv->ki', V_n, class_articles)
            attribute_scores_in_class = F.normalize(attribute_scores_in_class, dim=1)

        # score per class
        # distance_based_score = -torch.abs(instance_descriptor.unsqueeze(2) - attribute_scores_in_class.transpose(0, 1).
        #                             unsqueeze(0))
        # distance_based_score = torch.sum(distance_based_score, dim=1)
        distance_based_score = 0

        score_per_class = torch.einsum('ki,bi->bik', attribute_scores_in_class, instance_descriptor)
        score_per_class = torch.sum(score_per_class, dim=1)  # [bk] <== [bik]  # score per class

        score_per_class = score_per_class + self.vec_bias + distance_based_score

        package = {'score_per_class': score_per_class,
                   'distance_based_score': distance_based_score,
                   'attribute_existence': attribute_existence,
                   'attribute_visual_representative': visual_attributes,
                   'attribute_score': attribute_scores,
                   'instance_descriptor': instance_descriptor,
                   'class_signatures': attribute_scores_in_class,
                   'S_pp': score_per_class}

        if class_articles is not None:
            package['computed_class_articles'] = class_articles

        return package

    def compute_aug_cross_entropy(self, in_package, score_type='score_per_class'):
        batch_label = in_package['batch_label']
        score_per_class = in_package[score_type]

        Labels = batch_label
        # Labels = self.class_sim_distributions[in_package['batch_label_numeric']]

        score_per_class = score_per_class - self.vec_bias
        if self.bias == 0:
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

    def compute_loss_Self_Calibrate(self, in_package):
        score_per_class = in_package['score_per_class']

        prob_all = F.softmax(score_per_class, dim=-1)
        if self.cal_unseen:
            prob_to_cal = prob_all[:, self.unseen_classes]
            assert prob_to_cal.size(1) == len(self.unseen_classes)
        else:
            prob_to_cal = prob_all[:, self.seen_classes]
            assert prob_to_cal.size(1) == len(self.seen_classes)

        mass_unseen = torch.sum(prob_to_cal, dim=1)

        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp

    def articles_cosine_loss(self, in_package):
        seen_classes = self.seen_classes
        computed_class_articles = in_package['computed_class_articles'][seen_classes]
        computed_class_articles = F.normalize(computed_class_articles, dim=-1)
        gt_class_articles = self.gt_class_articles[seen_classes]

        loss = -torch.mean(torch.einsum('kiv,kiv->ki', computed_class_articles, gt_class_articles), dim=-1)
        return torch.mean(loss)

    def cosine_loss(self, in_package):
        # score_per_class = F.normalize(in_package['score_per_class'], dim=1)
        score_per_class = in_package['score_per_class']
        labels_one_hot = in_package['batch_label']
        loss = 1 - torch.einsum('bk,bk->b', labels_one_hot, score_per_class)
        return torch.mean(loss)

    def bce_loss(self, in_package):
        batch_attr_existence = in_package['attribute_existence']
        class_signatures = self.classes_attributes_raw

        true_class_signatures = class_signatures[in_package['batch_label_numeric']]

        loss = F.binary_cross_entropy(batch_attr_existence, true_class_signatures)

        return loss

    def compute_l1_loss(self, in_package):
        batch_samples = in_package['instance_descriptor']
        class_signatures = in_package['class_signatures']

        true_class_signatures = class_signatures[in_package['batch_label_numeric']]

        loss = F.l1_loss(batch_samples, true_class_signatures)
        return loss

    def save_backbone_chkpnt(self):
        self.orig_backbone_params = [p.clone().detach().requires_grad_(False) for p in self.backbone.parameters() if p.requires_grad]

    def compute_backbone_reg_loss(self, in_package):

        loss = [F.mse_loss(p, self.orig_backbone_params[i], reduction='sum') for i, p in enumerate(self.backbone.parameters()) if p.requires_grad]
        return torch.mean(torch.stack(loss))

    def compute_loss(self, in_package):

        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label_numeric'] = in_package['batch_label']
            labels_numeric = in_package['batch_label'].long()
            in_package['batch_label'] = self.weight_ce[labels_numeric]

        tot_loss = self.compute_aug_cross_entropy(in_package)
        loss_parts = {}

        loss_parts['xent_loss'] = tot_loss.item()
        loss_parts['loss_CE'] = loss_parts['xent_loss']
        # loss self-calibration
        if self.lambda_[0] > 0:
            loss_cal = self.lambda_[0]*self.compute_loss_Self_Calibrate(in_package)
            loss_parts['loss_cal'] = loss_cal.item()
            tot_loss += loss_cal

        if self.lambda_[1] > 0:
            loss_reg = self.lambda_[1]*self.compute_backbone_reg_loss(in_package)
            loss_parts['loss_backbone_reg'] = loss_reg.item()
            tot_loss += loss_reg

        #loss_sim = torch.tensor([0])#self.articles_cosine_loss(in_package)  #torch.tensor([0])  # self.bce_loss(in_package)

        out_package = {'loss': tot_loss, 'loss_parts': loss_parts, **loss_parts}

        return out_package

    def predict(self, dataset_loader):

        predictions = []
        labels = []
        for i, sample in enumerate(dataset_loader):
            features = sample[0].float().to(self.device)
            labels.append(sample[1])
            with torch.no_grad():
                predictions.append(self.forward(features)['score_per_class'])

        return torch.cat(predictions, dim=0), torch.cat(labels)

    def embed(self, dataset_loader):

        embeddings = []
        labels = []
        for i, sample in enumerate(dataset_loader):
            features = sample[0].float().to(self.device)
            labels.append(sample[1])
            with torch.no_grad():
                embeddings.append(self.forward(features)['instance_descriptor'])

        return torch.cat(embeddings, dim=0), torch.cat(labels)
