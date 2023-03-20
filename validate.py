import argparse
import math

from Config import cfg
from Config import update_config

from utils import create_logger
from model import Dynamic_sparse_alignment_network
from Dataloader import WFLW_Dataset

import torch
import cv2
import numpy as np
import pprint
import os

import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

    # philly
    parser.add_argument('--modelDir', help='model directory', type=str, default='./weights')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='DSLPT_WFLW_6_layers.pth')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--target', help='',
                        type=str, default='alignment')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    args = parser.parse_args()

    return args


def calculate_loss(data, input_tensor, ground_truth, box_size=None):

    L2_Loss = np.mean(np.linalg.norm((input_tensor - ground_truth), axis=1), axis=0)

    if data == 'WFLW':
        L2_norm = np.linalg.norm(ground_truth[60, :] - ground_truth[72, :], axis=0)
    else:
        raise NotImplementedError

    L2_Loss = L2_Loss / L2_norm
    return np.mean(L2_Loss)


def transform_pixel_v2(pt, trans, inverse=False):
    if inverse is False:
        pt = pt @ (trans[:,0:2].T) + trans[:,2]
    else:
        pt = (pt - trans[:,2]) @ np.linalg.inv(trans[:,0:2].T)
    return pt


def main_function():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.TARGET)
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.DATASET.DATASET == 'WFLW':
        model = Dynamic_sparse_alignment_network(cfg.WFLW.NUM_POINT, cfg.MODEL.OUT_DIM, cfg.MODEL.TRAINABLE,
                                                 cfg.MODEL.INTER_LAYER, cfg.TRANSFORMER.NHEAD, cfg.TRANSFORMER.FEED_DIM,
                                                 cfg.WFLW.INITIAL_PATH, cfg)
    else:
        raise NotImplementedError

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if cfg.DATASET.DATASET == 'WFLW':
        valid_dataset = WFLW_Dataset(
            cfg, cfg.WFLW.ROOT, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise NotImplementedError

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint_file = os.path.join(args.modelDir, args.checkpoint)
    checkpoint = torch.load(checkpoint_file)

    model.module.load_state_dict(checkpoint)

    landmark_list1 = []
    landmark_list2 = []
    landmark_list3 = []

    deviation_list1 = []
    deviation_list2 = []
    deviation_list3 = []

    loss_list1 = []
    loss_list2 = []
    loss_list3 = []

    anchor_list = []
    size_list = []

    ground_truth_list = []

    trans_list = []

    model.eval()

    with torch.no_grad():
        for i, (input, meta) in enumerate(valid_loader):
            outputs, deviation, box_achor, box_size = model(input.cuda())

            ground_truth = meta['initial'].cpu().numpy()[0]
            trans = meta['trans'].cpu().numpy()[0]

            output_stage1 = outputs[0][0, -1, :, :].cpu().numpy() * 256.0
            output_stage1 = transform_pixel_v2(output_stage1, trans, inverse=True)
            output_stage2 = outputs[1][0, -1, :, :].cpu().numpy() * 256.0
            output_stage2 = transform_pixel_v2(output_stage2, trans, inverse=True)
            output_stage3 = outputs[2][0, -1, :, :].cpu().numpy() * 256.0
            output_stage3 = transform_pixel_v2(output_stage3, trans, inverse=True)

            deviation_stage1 = deviation[0][0, -1, :, :].cpu().numpy()
            deviation_stage2 = deviation[1][0, -1, :, :].cpu().numpy()
            deviation_stage3 = deviation[2][0, -1, :, :].cpu().numpy()

            box_achor = box_achor[0].cpu().numpy()
            box_size = box_size[0].cpu().numpy()

            loss1 = calculate_loss(cfg.DATASET.DATASET, output_stage1, ground_truth)
            loss2 = calculate_loss(cfg.DATASET.DATASET, output_stage2, ground_truth)
            loss3 = calculate_loss(cfg.DATASET.DATASET, output_stage3, ground_truth)

            loss_list1.append(loss1)
            loss_list2.append(loss2)
            loss_list3.append(loss3)

            landmark_list1.append(output_stage1)
            landmark_list2.append(output_stage2)
            landmark_list3.append(output_stage3)

            deviation_list1.append(deviation_stage1)
            deviation_list2.append(deviation_stage2)
            deviation_list3.append(deviation_stage3)

            ground_truth_list.append(ground_truth)

            trans_list.append(trans)
            anchor_list.append(box_achor)
            size_list.append(box_size)

            print(loss3)

        # np.savez('./'+'WFLW_without_chin'+'.npz', trans_list=trans_list, loss_list1=loss_list1, loss_list2=loss_list2,
        #          loss_list3=loss_list3, deviation_list1=deviation_list1, deviation_list2=deviation_list2,
        #          deviation_list3=deviation_list3, landmark_list1=landmark_list1, landmark_list2=landmark_list2,
        #          landmark_list3=landmark_list3, ground_truth_list=ground_truth_list, anchor_list=anchor_list,
        #          size_list=size_list)
        print('Finished')
        print(np.mean(loss_list3))



if __name__ == '__main__':
    main_function()

