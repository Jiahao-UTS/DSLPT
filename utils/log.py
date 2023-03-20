import os
import logging
import time
from collections import namedtuple
from pathlib import Path


def create_logger(cfg, phase='train'):
    # 获取输出文件的根目录
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    # 如果输出目录不存在，就创建相应的输出目录
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    # 数据集
    dataset = cfg.DATASET.DATASET

    model = cfg.MODEL.NAME

    # 定义最后的输出目录
    # example "./Model/WFLW/UNet"
    final_output_dir = root_output_dir / dataset / model

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # 输出时间
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(model, phase, time_str)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    # 定义日志文件
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # 声明Tensorboard的存储路径
    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (phase + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    # 返回日志文件，最终文件输出地址以及tensorboard输出地址
    return logger, str(final_output_dir), str(tensorboard_log_dir)