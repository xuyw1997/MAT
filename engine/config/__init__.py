"""The global configs registry"""
from .default  import _C


__all__ = ['get_cfg']



def get_cfg():
    """Get a yacs CfgNode object with default values for by name.

    Parameters
    ----------
    name : str
        The name of the root config, e.g. action_recognition, coot, directpose...

    Returns
    -------
    yacs.CfgNode object

    """
    return _C.clone()
