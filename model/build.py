import torch
from fvcore.common.registry import Registry
from fvcore.nn import parameter_count
import utils.logging as logging
MODEL_REGISTRY = Registry("MODEL")

logger = logging.get_logger(__name__)
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


BACKBONE_REGISTRY = Registry("MODEL")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbone model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.DDP.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.DDP.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    logger.info(f"Model Parameters: {parameter_count(model)[''] / 10 ** 6}M")

    if cfg.DDP.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.DDP.NUM_GPUS > 1:
        # Make model replica operate on the current device
        # model = torch.nn.parallel.DistributedDataParallel(
        #     module=model,
        #     device_ids=[cur_device],
        #     output_device=cur_device,
        # )
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg.DDP.NUM_GPUS)))
    return model

