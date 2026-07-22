import os
import shutil
import tempfile

import numpy as np
import plotly.graph_objects as go

from aidevelopementtoolkit.logging_utils.pyplot_utils import plot_heatmap, plot_scatter
from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

if __name__ == "__main__":

    logger = get_formatted_logger()

    logger.info("Starting plot_utils integration test.")

    test_dir = tempfile.mkdtemp(prefix="test_plots_")
    logger.info("Temporary plot directory: %s", test_dir)

    try:
        # Test plot_heatmap()
        logger.info("Testing plot_heatmap()")

        matrix_path = os.path.join(
            test_dir,
            "matrix.png",
        )

        matrix = np.random.rand(10, 15).astype(np.float32)

        fig_matrix = plot_heatmap(
            data=matrix,
            title="Random Matrix",
            xaxis_title="Columns",
            yaxis_title="Rows",
            path=matrix_path,
        )

        assert isinstance(fig_matrix, go.Figure)
        assert os.path.exists(matrix_path)

        logger.info("✓ plot_heatmap() passed")


        # Test plot_scatter() without labels
        logger.info("Testing plot_scatter() without labels")

        scatter_path = os.path.join(
            test_dir,
            "scatter.png",
        )

        x = np.random.randn(100)
        y = np.random.randn(100)

        fig_scatter = plot_scatter(
            x=x,
            y=y,
            title="Random Scatter",
            xaxis_title="X",
            yaxis_title="Y",
            path=scatter_path,
        )

        assert isinstance(fig_scatter, go.Figure)
        assert os.path.exists(scatter_path)

        logger.info("✓ plot_scatter() without labels passed")


        # Test plot_scatter() with labels
        logger.info("Testing plot_scatter() with labels")

        labeled_scatter_path = os.path.join(
            test_dir,
            "scatter_labels.png",
        )

        labels = np.random.choice(
            ["class_a", "class_b", "class_c"],
            size=100,
        )

        fig_scatter_labels = plot_scatter(
            x=x,
            y=y,
            title="Labeled Scatter",
            xaxis_title="X",
            yaxis_title="Y",
            path=labeled_scatter_path,
            labels=labels,
            marker_size=10,
            marker_opacity=0.8,
            palette="Set2",
        )

        assert isinstance(fig_scatter_labels, go.Figure)
        assert os.path.exists(labeled_scatter_path)

        # One trace per label
        assert len(fig_scatter_labels.data) == len(
            np.unique(labels)
        )

        logger.info("✓ plot_scatter() with labels passed")


        # Test invalid input handling
        logger.info("Testing invalid inputs")

        try:
            plot_heatmap(
                data=np.random.rand(10),
                title="Invalid",
                xaxis_title="X",
                yaxis_title="Y",
                path=os.path.join(test_dir, "invalid.png"),
            )
            raise AssertionError(
                "plot_heatmap() should have failed for 1D input"
            )

        except ValueError:
            logger.info("✓ plot_heatmap() invalid input passed")


        try:
            plot_scatter(
                x=np.random.rand(10),
                y=np.random.rand(20),
                title="Invalid",
                xaxis_title="X",
                yaxis_title="Y",
                path=os.path.join(test_dir, "invalid.png"),
            )
            raise AssertionError(
                "plot_scatter() should have failed for different shapes"
            )

        except ValueError:
            logger.info("✓ plot_scatter() invalid input passed")


        logger.info(
            "All plot_utils integration tests passed successfully."
        )

    finally:
        shutil.rmtree(test_dir)
        logger.info("Temporary plot directory removed.")