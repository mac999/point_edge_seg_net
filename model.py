# Backward-compatibility shim; the implementation lives in models/edgeconv.py.
from models.common import FeatureGate, AttentionModule, LightweightTransformer
from models.edgeconv import *  # noqa: F401,F403
from models.edgeconv import PointEdgeSegNet, EdgeConv, grid_subsample  # noqa: F401
