import copy
import random
import logging
import json

import cv2
import numpy
import torch
import numpy as np
import os
import utils

from torch.utils.data import Dataset

from utils import generate_target


logger = logging.getLogger(__name__)


class WFLW_Dataset(Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = root
        self.number_landmarks = cfg.WFLW.NUM_POINT
        self.flip_index = np.genfromtxt(os.path.join(self.root, "Mirror.txt"),
                                        dtype=int, delimiter=',')

        self.Fraction = cfg.WFLW.FRACTION
        self.Translation_Factor = cfg.WFLW.TRANSLATION
        self.Rotation_Factor = cfg.WFLW.ROTATION
        self.Scale_Factor = cfg.WFLW.SCALE
        self.Occlusion_Mean = cfg.WFLW.OCCLUSION_MEAN
        self.Occlusion_Std = cfg.WFLW.OCCLUSION_STD
        self.Flip = cfg.WFLW.FLIP
        self.Occlusion = cfg.WFLW.OCCLUSION
        self.Transfer = cfg.WFLW.CHANNEL_TRANSFER
        self.Shear = cfg.WFLW.SHEAR
        self.Brightness = cfg.WFLW.BRIGHTNESS

        self.Data_Format = cfg.WFLW.DATA_FORMAT

        self.Transform = transform

        if is_train:
            self.annotation_file = os.path.join(root, 'WFLW_annotations', 'list_98pt_rect_attr_train_test',
                                                'list_98pt_rect_attr_train.txt')
        else:
            # self.annotation_file = os.path.join(root, 'WFLW_annotations', 'list_98pt_test',
            #                                     'list_98pt_test_occlusion.txt')
            self.annotation_file = os.path.join(root, 'WFLW_annotations', 'list_98pt_rect_attr_train_test',
                                                'list_98pt_rect_attr_test.txt')

        self.database = self.get_file_information()

    def get_file_information(self):
        Data_base = []

        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_point = []
            temp_info = temp_info.split(' ')
            for i in range(2 * self.number_landmarks):
                temp_point.append(float(temp_info[i]))
            point_coord = np.array(temp_point, dtype=np.float).reshape(self.number_landmarks, 2)
            max_index = np.max(point_coord, axis=0)
            min_index = np.min(point_coord, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])
            temp_name = os.path.join(self.root, 'WFLW_images', temp_info[-1])
            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': point_coord})

        return Data_base

    def Image_Flip(self, Img, GT):
        Mirror_GT = []
        width = Img.shape[1]
        for i in self.flip_index:
            Mirror_GT.append([width - GT[i][0], GT[i][1]])
        Img = cv2.flip(Img, 1)
        return Img, numpy.array(Mirror_GT)

    def Channel_Transfer(self, Img, Flag):
        if Flag == 1:
            Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        return Img

    def img_shear(self, img, points):
        shear_factor = np.clip(np.random.normal(0, 0.2), -0.2, 0.2)
        M = np.array([[1, shear_factor, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)
        for i in range(self.number_landmarks):
            points[i, 0:2] = utils.affine_transform(points[i, 0:2], M)
        return img, points

    def Brightness_adjust(self, img):
        brightness = np.clip(np.random.normal(0, 30.0), -30.0, 30.0)
        contrast = np.clip(np.random.normal(1.0, 0.3), 0.3, 1.3)
        img = img.astype(np.float)
        img = contrast * img + brightness
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        return img

    def Create_Occlusion(self, Img):
        Occlusion_width = int(self.Image_size * np.random.normal(self.Occlusion_Mean, self.Occlusion_Std))
        Occlusion_high = int(self.Image_size * np.random.normal(self.Occlusion_Mean, self.Occlusion_Std))
        Occlusion_x = np.random.randint(0, self.Image_size - Occlusion_width)
        Occlusion_y = np.random.randint(0, self.Image_size - Occlusion_high)

        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 0] = \
            np.random.randint(0, 256)
        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 1] = \
            np.random.randint(0, 256)
        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 2] = \
            np.random.randint(0, 256)

        return Img

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])

        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['point']
        initial = Points.copy()

        Img = cv2.imread(Img_path)

        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        if self.is_train == True:
            Rotation_Factor = self.Rotation_Factor * np.pi / 180.0
            Scale_Factor = self.Scale_Factor
            Translation_X_Factor = self.Translation_Factor
            Translation_Y_Factor = self.Translation_Factor

            angle = np.clip(np.random.normal(0, Rotation_Factor), -2 * Rotation_Factor, 2 * Rotation_Factor)
            Scale = np.clip(np.random.normal(self.Fraction, Scale_Factor), self.Fraction - Scale_Factor, self.Fraction + Scale_Factor)

            Translation_X = np.clip(np.random.normal(0, Translation_X_Factor), -Translation_X_Factor, Translation_X_Factor)
            Translation_Y = np.clip(np.random.normal(0, Translation_Y_Factor), -Translation_Y_Factor, Translation_Y_Factor)

            trans = utils.get_transforms(BBox, Scale, angle, self.Image_size, shift_factor=[Translation_X, Translation_Y])

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            for i in range(self.number_landmarks):
                Points[i,0:2] = utils.affine_transform(Points[i,0:2], trans)

            if self.Flip is True:
                Flip_Flag = np.random.randint(0, 2)
                if Flip_Flag == 1:
                    input, Points = self.Image_Flip(input, Points)

            if self.Transfer is True:
                Transfer_Flag = np.random.randint(0, 5)
                input = self.Channel_Transfer(input, Transfer_Flag)

            if self.Brightness is True:
                Brightness_Flag = np.random.randint(0, 2)
                if Brightness_Flag == 1:
                    input = self.Brightness_adjust(input)

            if self.Occlusion is True:
                Occlusion_Flag = np.random.randint(0, 2)
                if Occlusion_Flag == 1:
                    input = self.Create_Occlusion(input)

            if self.Shear is True:
                shear_flag = np.random.randint(0,3)
                if shear_flag == 1:
                    input, Points = self.img_shear(input, Points)

            if self.Transform is not None:
                input = self.Transform(input)

            meta = {'Img_path': Img_path,
                    'Points': Points / (self.Image_size),
                    'BBox': BBox,
                    'trans': trans,
                    'Scale': Scale,
                    'angle': angle,
                    'Translation': [Translation_X, Translation_Y]}

            return input, meta

        else:
            trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            for i in range(self.number_landmarks):
                Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)

            meta = {
                'initial': initial,
                'Img_path': Img_path,
                'Points': Points / (self.Image_size),
                'BBox': BBox,
                'trans': trans,
                'Scale': self.Fraction,
                'angle': 0.0,
                'Translation': [0.0, 0.0],
            }

            if self.Transform is not None:
                input = self.Transform(input)

            return input, meta

