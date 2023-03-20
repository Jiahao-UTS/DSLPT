from torch import optim

def get_optimizer(cfg, model):
    optimizer = None

    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )

    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.TRAIN.LR,betas=(0.9, 0.95)
        )

    return optimizer
