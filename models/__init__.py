# models/__init__.py
# Architecture registry. Files are named after what the architecture is, not after
# a version number; each module exposes a class named PointEdgeSegNet.

from .edgeconv import PointEdgeSegNet as EdgeConvSegNet
from .stencil import PointEdgeSegNet as StencilSegNet

ARCHS = {
    'edgeconv': EdgeConvSegNet,   # kNN-graph EdgeConv (legacy, v1)
    'stencil': StencilSegNet,     # voxel-stencil lookup (current, v2)
}
_ALIASES = {'v1': 'edgeconv', 'v2': 'stencil'}


def resolve_arch(name):
    """Normalize an architecture name ('v1'/'v2' aliases included)."""
    key = _ALIASES.get(name, name)
    if key not in ARCHS:
        raise KeyError(f"unknown architecture '{name}' (available: {sorted(ARCHS)})")
    return key


def get_arch(name):
    return ARCHS[resolve_arch(name)]
