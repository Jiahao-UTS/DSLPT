import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Transformer import Transformer
from Transformer import interpolation_layer
from Transformer import get_roi
from backbone import get_face_alignment_net


class Dynamic_sparse_alignment_network(nn.Module):
    def __init__(self, num_point, d_model, trainable,
                 return_interm_layers, nhead, feedforward_dim,
                 initial_path, cfg):
        super(Dynamic_sparse_alignment_network, self).__init__()
        self.num_point = num_point
        self.d_model = d_model
        self.trainable = trainable
        self.return_interm_layers = return_interm_layers
        self.nhead = nhead
        self.feedforward_dim = feedforward_dim
        self.initial_path = initial_path

        self.initial_points = torch.from_numpy(np.load(initial_path)['init_face'] / 256.0).view(1, num_point, 2).float()
        self.initial_points.requires_grad = False
        self.initial_scale = torch.tensor([[[0.125, 0.125]]], dtype=torch.float32).repeat(1, num_point, 1)
        self.initial_scale.requires_grad = False

        self.Sample_num = cfg.MODEL.SAMPLE_NUM

        # ROI_creator
        self.ROI = get_roi(self.Sample_num)

        self.interpolation = interpolation_layer(num_point)

        self.feature_extractor = nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=True)

        self.feature_norm = nn.LayerNorm(d_model)

        # Transformer
        self.Transformer = Transformer(num_point, d_model, nhead, cfg.TRANSFORMER.NUM_DECODER, feedforward_dim, dropout=0.1)
        self.embedding_layer = nn.Linear(d_model, 2 * d_model)

        self.out_mean_layer = nn.Linear(2 * d_model, 2)
        self.out_std_layer = nn.Linear(2 * d_model, 2)

        self._reset_parameters()

        # backbone
        self.backbone = get_face_alignment_net(cfg)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def Gaussian_loss(self, mean, std, ground_truth):
        loss1 = torch.exp(- (mean - ground_truth) ** 2.0 / std / 2.0) / torch.sqrt(2.0 * np.pi * std)
        loss1 = -torch.log(loss1 + 1e-9)
        return torch.mean(loss1)

    def forward(self, image, ground_truth=None, is_train=False):
        bs = image.size(0)

        output_list = []

        feature_map = self.backbone(image)
        initial_landmarks = self.initial_points.repeat(bs, 1, 1).to(image.device)
        bbox_scale_1 = self.initial_scale.repeat(bs, 1, 1).to(image.device)

        ROI_anchor_1, bbox_size_1, start_anchor_1 = self.ROI(initial_landmarks.detach(), bbox_scale_1)
        ROI_anchor_1 = ROI_anchor_1.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_1 = self.interpolation(feature_map, ROI_anchor_1.detach()).view(bs, self.num_point, self.Sample_num,
                                                                            self.Sample_num, self.d_model)
        ROI_feature_1 = ROI_feature_1.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                     self.d_model).permute(0, 3, 2, 1)

        transformer_feature_1 = self.feature_extractor(ROI_feature_1).view(bs, self.num_point, self.d_model)

        offset_1 = self.Transformer(transformer_feature_1)
        offset_1 = F.relu(self.embedding_layer(offset_1))
        offset_mean_1 = self.out_mean_layer(offset_1)
        offset_std_1 = self.out_std_layer(offset_1)
        offset_std_1 = 1 / (1 + torch.exp(-offset_std_1))

        landmarks_1 = start_anchor_1.unsqueeze(1) + bbox_size_1.unsqueeze(1) * offset_mean_1
        output_list.append(landmarks_1)

        if is_train:
            ground_truth_1 = ((ground_truth - start_anchor_1) / bbox_size_1).unsqueeze(1)
            loss1 = self.Gaussian_loss(offset_mean_1, offset_std_1, ground_truth_1)

        bbox_scale_2 = bbox_scale_1 * torch.clamp(
            torch.max(offset_std_1[:, -1, :, :].detach() * 12, dim=2, keepdim=True)[0], 0.5, 0.7)
        ROI_anchor_2, bbox_size_2, start_anchor_2 = self.ROI(landmarks_1[:, -1, :, :].detach(), bbox_scale_2)
        ROI_anchor_2 = ROI_anchor_2.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_2 = self.interpolation(feature_map, ROI_anchor_2.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                 self.Sample_num, self.d_model)
        ROI_feature_2 = ROI_feature_2.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                           self.d_model).permute(0, 3, 2, 1)

        transformer_feature_2 = self.feature_extractor(ROI_feature_2).view(bs, self.num_point, self.d_model)

        offset_2 = self.Transformer(transformer_feature_2)
        offset_2 = F.relu(self.embedding_layer(offset_2))
        offset_mean_2 = self.out_mean_layer(offset_2)
        offset_std_2 = self.out_std_layer(offset_2)
        offset_std_2 = 1 / (1 + torch.exp(-offset_std_2))
        landmarks_2 = start_anchor_2.unsqueeze(1) + bbox_size_2.unsqueeze(1) * offset_mean_2
        output_list.append(landmarks_2)

        if is_train:
            ground_truth_2 = ((ground_truth - start_anchor_2) / bbox_size_2).unsqueeze(1)
            loss2 = self.Gaussian_loss(offset_mean_2, offset_std_2, ground_truth_2)

        bbox_scale_3 = bbox_scale_2 * torch.clamp(
            torch.max(offset_std_2[:, -1, :, :].detach() * 12, dim=2, keepdim=True)[0], 0.5, 0.7)
        ROI_anchor_3, bbox_size_3, start_anchor_3 = self.ROI(landmarks_2[:, -1, :, :].detach(), bbox_scale_3)
        ROI_anchor_3 = ROI_anchor_3.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)
        ROI_feature_3= self.interpolation(feature_map, ROI_anchor_3.detach()).view(bs, self.num_point, self.Sample_num,
                                                                                   self.Sample_num, self.d_model)
        ROI_feature_3 = ROI_feature_3.view(bs * self.num_point, self.Sample_num, self.Sample_num,
                                           self.d_model).permute(0, 3, 2, 1)

        transformer_feature_3 = self.feature_extractor(ROI_feature_3).view(bs, self.num_point, self.d_model)

        offset_3 = self.Transformer(transformer_feature_3)
        offset_3 = F.relu(self.embedding_layer(offset_3))
        offset_mean_3 = self.out_mean_layer(offset_3)
        offset_std_3 = self.out_std_layer(offset_3)
        offset_std_3 = 1 / (1 + torch.exp(-offset_std_3))

        landmarks_3 = start_anchor_3.unsqueeze(1) + bbox_size_3.unsqueeze(1) * offset_mean_3
        output_list.append(landmarks_3)

        if is_train:
            ground_truth_3 = ((ground_truth - start_anchor_3) / bbox_size_3).unsqueeze(1)
            loss3 = self.Gaussian_loss(offset_mean_3, offset_std_3, ground_truth_3)

        if is_train:
            return output_list, [loss1, loss2, loss3]
        else:
            return output_list, [offset_std_1, offset_std_2, offset_std_3], start_anchor_3, bbox_size_3

