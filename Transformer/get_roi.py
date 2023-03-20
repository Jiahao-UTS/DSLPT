import torch
import torch.nn as nn

class get_roi(nn.Module):

    def __init__(self, num_points):
        super(get_roi, self).__init__()
        self.num_points = num_points

    def forward(self, anchor, bbox_scale):
        bounding_min = torch.clamp(anchor - bbox_scale, 0.0, 1.0)
        bounding_min = torch.where(bounding_min > (1 - 2 * bbox_scale), (1 - 2 * bbox_scale), bounding_min)
        bounding_max = bounding_min + 2 * bbox_scale
        bounding_box = torch.cat((bounding_min, bounding_max), dim=2)
        bounding_length = bounding_max - bounding_min

        bounding_xs = torch.nn.functional.interpolate(bounding_box[:,:,0::2], size=self.num_points,
                                                      mode='linear', align_corners=True)
        bounding_ys = torch.nn.functional.interpolate(bounding_box[:,:,1::2], size=self.num_points,
                                                      mode='linear', align_corners=True)
        bounding_xs, bounding_ys = bounding_xs.unsqueeze(3).repeat_interleave(self.num_points, dim=3), \
                                   bounding_ys.unsqueeze(2).repeat_interleave(self.num_points, dim=2)

        meshgrid = torch.stack([bounding_xs, bounding_ys], dim=-1)

        return meshgrid, bounding_length, bounding_min

