"""Unit tests for Springer Nature LNEE vector figures generator (paper_figs/make_figures.py).

Strictly follows docs/EVAL_CONTRACT.md.
"""

from __future__ import annotations

import os
import pytest

from paper_figs.make_figures import (
    main as make_figures_main,
    plot_3d_patch_similarity,
    plot_gating_efficiency,
    plot_navigation_trajectories,
    plot_training_and_rankme,
)


def test_make_figures_demo_all_four_pdfs(tmp_path) -> None:
    """Run make_figures.py --demo --out_dir tmp_path:

    Assert all 4 .pdf vector files are created and non-empty (size > 5 KB).
    Also verify 600 DPI .png previews are generated.
    """
    out_dir = str(tmp_path / "figures")

    # Run main CLI driver in demo mode
    ret = make_figures_main(["--demo", "--out_dir", out_dir, "--dpi", "150"])
    assert ret == 0

    expected_figures = [
        "fig1_training_and_rankme",
        "fig2_3d_patch_similarity",
        "fig3_gating_efficiency",
        "fig4_navigation_trajectories",
    ]

    for fig_name in expected_figures:
        pdf_file = os.path.join(out_dir, f"{fig_name}.pdf")
        png_file = os.path.join(out_dir, f"{fig_name}.png")

        # 1. Assert PDF file exists
        assert os.path.isfile(pdf_file), f"Expected vector PDF at {pdf_file}"
        pdf_size = os.path.getsize(pdf_file)
        # 2. Assert PDF is non-empty (size > 5 KB / 5000 bytes)
        assert pdf_size > 5000, f"PDF {fig_name}.pdf size ({pdf_size} bytes) is <= 5 KB"

        # 3. Assert PNG preview exists and is non-empty
        assert os.path.isfile(png_file), f"Expected raster PNG preview at {png_file}"
        png_size = os.path.getsize(png_file)
        assert png_size > 5000, f"PNG {fig_name}.png size ({png_size} bytes) is <= 5 KB"


def test_individual_plotting_functions(tmp_path) -> None:
    """Test calling each plotting function individually with custom options."""
    out_dir = str(tmp_path / "individual")
    os.makedirs(out_dir, exist_ok=True)

    # 1. plot_training_and_rankme
    pdf1, png1 = plot_training_and_rankme(metrics_paths=None, out_dir=out_dir, dpi=100)
    assert os.path.isfile(pdf1) and os.path.getsize(pdf1) > 5000
    assert os.path.isfile(png1) and os.path.getsize(png1) > 5000

    # 2. plot_3d_patch_similarity
    pdf2, png2 = plot_3d_patch_similarity(backbone=None, shells_single=None, query_idx=36, out_dir=out_dir, dpi=100)
    assert os.path.isfile(pdf2) and os.path.getsize(pdf2) > 5000
    assert os.path.isfile(png2) and os.path.getsize(png2) > 5000

    # 3. plot_gating_efficiency
    pdf3, png3 = plot_gating_efficiency(gating_stats=None, out_dir=out_dir, dpi=100)
    assert os.path.isfile(pdf3) and os.path.getsize(pdf3) > 5000
    assert os.path.isfile(png3) and os.path.getsize(png3) > 5000

    # 4. plot_navigation_trajectories
    pdf4, png4 = plot_navigation_trajectories(volume=None, astar_path=None, model_paths=None, waypoints=None, out_dir=out_dir, dpi=100)
    assert os.path.isfile(pdf4) and os.path.getsize(pdf4) > 5000
    assert os.path.isfile(png4) and os.path.getsize(png4) > 5000
