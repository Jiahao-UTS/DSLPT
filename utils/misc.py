import os
import torch

from torch import Tensor
import torch.distributed as dist

from typing import Optional, List


# 特征图打包
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


# 计算分布式集群
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# 将tensor转换为nestedtensor
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    # 判断是否为三维张量，即是RGB
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        # 输入img的shape大小，寻找这个list中图像最大的宽与高
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        # 得到一个batch性质[batch, max_dim, max_height, max_width]
        batch_shape = [len(tensor_list)] + max_size
        # 获得batch_shape
        b, c, h, w = batch_shape
        # 获取tensor的类型
        dtype = tensor_list[0].dtype
        # 获取tensor所在设备
        device = tensor_list[0].device
        # 生成一个batch_shape那么大的全0矩阵，类型设备与原tensor相同
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        # 生成一个全一的mask，大小为[batch, height, width]
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        # 把图像粘贴到新生成的tensor中，将mask存在图像的区域都改变为False
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    # 否则报错
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)
