from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # ---------------------------------------------------------------------------- #
    # Resume
    # ---------------------------------------------------------------------------- #
    # Automatically resume weights from last checkpoints
    cfg.AUTO_RESUME = True
    # Whether to resume the optimizer and the scheduler
    cfg.RESUME_STATES = True
    # Path of weights to resume
    cfg.RESUME_PATH = ''
    # Whether to resume weights strictly
    cfg.RESUME_STRICT = True

    # ---------------------------------------------------------------------------- #
    # Optimizer
    # ---------------------------------------------------------------------------- #
    cfg.OPTIMIZER = CN()
    cfg.OPTIMIZER.TYPE = ''

    # Basic parameters of the optimizer
    # Note that the learning rate should be changed according to batch size
    cfg.OPTIMIZER.LR = 1e-3
    cfg.OPTIMIZER.WEIGHT_DECAY = 0.0
    # Maximum norm of gradients. Non-positive for disable
    cfg.OPTIMIZER.MAX_GRAD_NORM = 0.0

    # Specific parameters of optimizers
    cfg.OPTIMIZER.SGD = CN()
    cfg.OPTIMIZER.SGD.momentum = 0.9

    cfg.OPTIMIZER.Adam = CN()
    cfg.OPTIMIZER.Adam.betas = (0.9, 0.999)

    # ---------------------------------------------------------------------------- #
    # Scheduler (learning rate schedule)
    # ---------------------------------------------------------------------------- #
    cfg.LR_SCHEDULER = CN()
    cfg.LR_SCHEDULER.TYPE = ''

    # Specific parameters of schedulers
    cfg.LR_SCHEDULER.StepLR = CN()
    cfg.LR_SCHEDULER.StepLR.step_size = 0
    cfg.LR_SCHEDULER.StepLR.gamma = 0.1

    cfg.LR_SCHEDULER.MultiStepLR = CN()
    cfg.LR_SCHEDULER.MultiStepLR.milestones = ()
    cfg.LR_SCHEDULER.MultiStepLR.gamma = 0.1

    # ---------------------------------------------------------------------------- #
    # Specific train options
    # ---------------------------------------------------------------------------- #
    cfg.TRAIN = CN()

    # Batch size
    cfg.TRAIN.BATCH_SIZE = 1
    # Number of workers (dataloader)
    cfg.TRAIN.NUM_WORKERS = 0
    # Period to save checkpoints. 0 for disable
    cfg.TRAIN.CHECKPOINT_PERIOD = 0
    # Period to log training status. 0 for disable
    cfg.TRAIN.LOG_PERIOD = 0
    # Period to summary training status. 0 for disable
    cfg.TRAIN.SUMMARY_PERIOD = 0
    # Max number of checkpoints to keep
    cfg.TRAIN.MAX_TO_KEEP = 0
    # Max number of iteration
    cfg.TRAIN.MAX_ITER = 1

    # ---------------------------------------------------------------------------- #
    # Specific validation options
    # ---------------------------------------------------------------------------- #
    cfg.VAL = CN()

    # Batch size
    cfg.VAL.BATCH_SIZE = 1
    # Number of workers (dataloader)
    cfg.VAL.NUM_WORKERS = 0
    # Period to validate. 0 for disable
    cfg.VAL.PERIOD = 0
    # Period to log validation status. 0 for disable
    cfg.VAL.LOG_PERIOD = 0

    # The metric for best validation performance
    cfg.VAL.METRIC = ''
    cfg.VAL.METRIC_ASCEND = True

    # ---------------------------------------------------------------------------- #
    # Misc options
    # ---------------------------------------------------------------------------- #
    # if set to @, the filename of config will be used by default
    cfg.OUTPUT_DIR = '@'

    # For reproducibility...but not really because modern fast GPU libraries use
    # non-deterministic op implementations
    # -1 means use time seed.
    cfg.RNG_SEED = -1

    return cfg


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #
def purge_config(cfg: CN):
    """Purge config for clean logs and logical check.

    Notes:
        If a CfgNode has 'TYPE' attribute,
        its CfgNode children the key of which do not contain 'TYPE' will be removed.
    """
    target_key = cfg.get('TYPE', None)
    removed_keys = []
    for k, v in cfg.items():
        if isinstance(v, CN):
            if target_key is not None and (k != target_key):
                removed_keys.append(k)
            else:
                purge_config(v)
    for k in removed_keys:
        del cfg[k]
