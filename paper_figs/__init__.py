"""Publication Vector Figures Package for Squared Sense World Model.

Compliant with Springer Nature LNEE standards:
  - Vector PDF figures + 600 DPI PNG raster previews
  - Grayscale-safe colorblind palettes with distinct markers/linestyles
  - Zero raw Minecraft screenshots (rendered directly from NumPy voxel arrays).
"""

from paper_figs.make_figures import (
    plot_3d_patch_similarity,
    plot_gating_efficiency,
    plot_navigation_trajectories,
    plot_training_and_rankme,
)

__all__ = [
    "plot_training_and_rankme",
    "plot_3d_patch_similarity",
    "plot_gating_efficiency",
    "plot_navigation_trajectories",
]
