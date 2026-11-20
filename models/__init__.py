"""Neural network modules for Squared Sense."""

from models.baselines import (
    Generative3DMAE,
    SingleLODJEPA,
)
from models.cnn3d import (
    ConvNeXt3DSEBlock,
    CoordConv3D,
    SiameseVoxelCNN,
    count_params,
)
from models.gating import (
    DENSE_MFLOPS_PER_STEP,
    MFLOPS_PER_SHELL,
    MFLOPS_TRANSFORMER,
    EventDrivenShellGate,
)
from models.temporal_lstm import (
    DualLSTMTemporalIntegrator,
    LSTMCellNNX,
    QueryCrossAttentionBlock,
    ego_motion_shift_3d,
)
from models.transformer import (
    JEPAPredictor,
    MultiLODBackbone,
    PreLNTransformerBlock,
    SpatialJEPA,
    sample_3d_jepa_masks,
)

__all__ = [
    "CoordConv3D",
    "ConvNeXt3DSEBlock",
    "SiameseVoxelCNN",
    "count_params",
    "PreLNTransformerBlock",
    "MultiLODBackbone",
    "JEPAPredictor",
    "SpatialJEPA",
    "sample_3d_jepa_masks",
    "EventDrivenShellGate",
    "MFLOPS_PER_SHELL",
    "MFLOPS_TRANSFORMER",
    "DENSE_MFLOPS_PER_STEP",
    "ego_motion_shift_3d",
    "LSTMCellNNX",
    "QueryCrossAttentionBlock",
    "DualLSTMTemporalIntegrator",
    "SingleLODJEPA",
    "Generative3DMAE",
]
