# Backward-compatibility shim; the implementation lives in models/stencil.py.
from models.stencil import *  # noqa: F401,F403
from models.stencil import PointEdgeSegNet as PointEdgeSegNetV2  # noqa: F401
from models.stencil import (serialize, stencil_neighbors, stencil_groups,  # noqa: F401
                            window_neighbors, grid_pool, MetaBlock)
