import math
import torch
import torch.nn as nn


class Alignment_Loss(nn.Module):
    def __init__(self, cfg):
        super(Alignment_Loss, self).__init__()
        self.num_point = cfg.WFLW.NUM_POINT
        self.flag = cfg.DATASET.DATASET

    def forward(self, input_tensor, ground_truth):
        ground_truth = ground_truth.unsqueeze(1)
        L2_Loss = torch.mean(torch.norm((input_tensor - ground_truth), dim=3), dim=2)
        if self.flag == 'WFLW':
            L2_norm = torch.norm(ground_truth[:, :, 60, :] - ground_truth[:, :, 72, :], dim=2)
        elif self.flag == '300W':
            L2_norm = torch.norm(ground_truth[:, :, 36, :] - ground_truth[:, :, 45, :], dim=2)
        elif self.flag == 'COFW':
            L2_norm = torch.norm(ground_truth[:, :, 8, :] - ground_truth[:, :, 9, :], dim=2)
        L2_Loss = L2_Loss / L2_norm
        return torch.mean(L2_Loss)


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2, img_size=255):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.img_size = img_size
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)

    def forward(self, pred, target):
        bs, num, point, _ = pred.size()
        y = target.view(bs, 1, point * 2) * self.img_size
        y_hat = pred.view(bs, num, point * 2) * self.img_size
        delta_y = (y_hat - y).abs()
        zero_mask = torch.zeros_like(delta_y, dtype=torch.float32)
        one_mask = torch.ones_like(delta_y, dtype=torch.float32)
        delta_y1_mask = torch.where(delta_y < self.omega, one_mask, zero_mask)
        delta_y2_mask = torch.where(delta_y >= self.omega, one_mask, zero_mask)
        loss1 = self.omega * torch.log(1 + delta_y / self.epsilon) * delta_y1_mask
        loss2 = (delta_y - self.C) * delta_y2_mask
        return (loss1 + loss2).mean() / self.img_size


class Softwing(nn.Module):
    def __init__(self, thres1=2, thres2=20, curvature=2, img_size=255):
        super(Softwing, self).__init__()
        self.thres1 = thres1
        self.thres2 = thres2
        self.img_size = img_size
        self.curvature = curvature
        self.B = thres1 - thres2 * math.log(1 + thres1 / curvature)
        self.C = thres2 - thres2 * math.log(1 + thres2 / curvature) - self.B

    def forward(self, pred, target):
        bs, num, point, _ = pred.size()
        target = target.view(bs, 1, point * 2) * self.img_size
        pred = pred.view(bs, num, point * 2) * self.img_size

        loss = (target - pred).abs()

        # idx_small = loss < self.thres1
        idx_normal = (loss >= self.thres1) * (loss < self.thres2)
        idx_big = loss >= self.thres2

        loss[idx_normal] = self.thres2 * torch.log(1 + loss[idx_normal] / self.curvature) + self.B
        loss[idx_big] = loss[idx_big] - self.C

        return loss.mean() / self.img_size


class L1_loss(nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()

    def forward(self, pred, target):
        bs, num, point, _ = pred.size()
        target = target.view(bs, 1, point * 2)
        pred = pred.view(bs, num, point * 2)
        loss = (target - pred).abs()
        return loss.mean()