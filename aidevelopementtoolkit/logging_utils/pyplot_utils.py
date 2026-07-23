from typing import List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger
import mlflow


import numpy as np
import plotly.graph_objects as go

def plot_heatmap(
        data: np.ndarray,
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        path: str,
        xticklabels: Optional[List[str]] = None,
        yticklabels: Optional[List[str]] = None,
    ) -> go.Figure:
    """Plot a matrix as an annotated heatmap.

    Parameters
    ----------
    data : np.ndarray
        Matrix to plot with shape `(N, M)`.

    title : str
        Plot title.

    xaxis_title : str
        Title for the X-axis.

    yaxis_title : str
        Title for the Y-axis.

    path : str
        Path where the figure will be saved. If an MLFlow
        run is started, the figure is logged in the given `path`.

    xticklabels : Optional[List[str]], default=None
        Labels for the X-axis ticks. Must have length equal to the number of
        columns in `data`. If `None`, column indices are used.

    yticklabels : Optional[List[str]], default=None
        Labels for the Y-axis ticks. Must have length equal to the number of
        rows in `data`. If `None`, row indices are used.

    Returns
    -------
    go.Figure
        Matrix figure.

    Examples
    --------
    Create and save a heatmap from a NumPy matrix:

    >>> import numpy as np
    >>> matrix = np.array([
    ...     [1.0, 2.0, 3.0],
    ...     [4.0, 5.0, 6.0],
    ... ])
    >>> fig = plot_heatmap(
    ...     data=matrix,
    ...     title="Example Heatmap",
    ...     xaxis_title="Columns",
    ...     yaxis_title="Rows",
    ...     path="heatmap.png",
    ... )

    The function logs the figure to MLflow automatically if an MLflow
    run is active:

    >>> import mlflow
    >>> with mlflow.start_run():
    ...     plot_heatmap(
    ...         data=matrix,
    ...         title="MLflow Heatmap",
    ...         xaxis_title="Columns",
    ...         yaxis_title="Rows",
    ...         path="figures/heatmap.png",
    ...     )
    """

    logger = get_formatted_logger(name=__name__, level="ERROR")

    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        logger.error(f"Expected a 2D matrix, got {data.shape}")
        raise ValueError()

    data_min = np.min(data)
    data_max = np.max(data)

    if data_min == data_max:
        zmin = data_min - 1
        zmax = data_max + 1
    else:
        zmin = data_min
        zmax = data_max

    n_rows, n_cols = data.shape

    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            colorscale="Blues",
            zmin=zmin,
            zmax=zmax,
            text=np.round(data, 2),
            texttemplate="%{z:.2f}",
            textfont={"size": 12},
            colorbar=dict(title="Value"),
            hovertemplate=(
                "Row: %{y}<br>"
                "Col: %{x}<br>"
                "Value: %{z:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_cols)),
        ticktext=xticklabels if xticklabels is not None else [str(i) for i in range(n_cols)],
        side="bottom",
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_rows)),
        ticktext=yticklabels if yticklabels is not None else [str(i) for i in range(n_rows)],
        autorange="reversed",
    )

    if mlflow.active_run() is not None:
        mlflow.log_figure(fig, path)
    else:
        fig.write_image(path)

    return fig


def plot_scatter(
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        path: str,
        labels: Optional[np.ndarray] = None,
        marker_size: int = 8,
        marker_opacity: float = 1,
        palette: str = "Plotly",
        xticklabels: Optional[List[str]] = None,
        yticklabels: Optional[List[str]] = None,
    ) -> go.Figure:
    """Plot a scatter plot.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of the points. Shape `(N,)`.

    y : np.ndarray
        Y coordinates of the points. Shape `(N,)`.

    title : str
        Plot title.

    xaxis_title : str
        Title for the X-axis.

    yaxis_title : str
        Title for the Y-axis.

    path : str
        Path where the figure will be saved. If an MLFlow
        run is started, the figure is logged in the given `path`.

    labels : Optional[np.ndarray], default=None
        Class labels associated with each point. Shape `(N,)`.
        Each unique label is assigned a different color.

    marker_size : int, default=8
        Marker size.

    marker_opacity : float, default=1
        Marker opacity.

    palette : str, default="Plotly"
        Qualitative Plotly color palette to use when labels are provided.
        Examples: "Plotly", "D3", "Set1", "Set2", "Dark24".

    xticklabels : Optional[List[str]], default=None
        Custom labels for the X-axis ticks. If `None`, Plotly defaults are used.

    yticklabels : Optional[List[str]], default=None
        Custom labels for the Y-axis ticks. If `None`, Plotly defaults are used.

    Returns
    -------
    go.Figure
        Scatter plot figure.

    Examples
    --------
    Create a simple scatter plot:

    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([2, 4, 1, 5])
    >>> fig = plot_scatter(
    ...     x=x,
    ...     y=y,
    ...     title="Example Scatter Plot",
    ...     xaxis_title="X",
    ...     yaxis_title="Y",
    ...     path="scatter.png",
    ... )

    Create a scatter plot with class labels:

    >>> labels = np.array(["cat", "dog", "cat", "dog"])
    >>> fig = plot_scatter(
    ...     x=x,
    ...     y=y,
    ...     labels=labels,
    ...     title="Scatter Plot by Class",
    ...     xaxis_title="X",
    ...     yaxis_title="Y",
    ...     path="scatter_labels.png",
    ...     palette="Set2",
    ... )

    The function logs the figure to MLflow automatically if an MLflow
    run is active:

    >>> import mlflow
    >>> with mlflow.start_run():
    ...     plot_scatter(
    ...         x=x,
    ...         y=y,
    ...         title="MLflow Scatter Plot",
    ...         xaxis_title="X",
    ...         yaxis_title="Y",
    ...         path="figures/scatter.png",
    ...     )
    """

    logger = get_formatted_logger(name=__name__, level="ERROR")

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        logger.error(
            "The given `x` and `y` must have shape `(N,)`. "
            f"Received {x.shape} and {y.shape}."
        )
        raise ValueError()

    if x.shape != y.shape:
        logger.error(
            "The given `x` and `y` have different shapes: "
            f"{x.shape} vs {y.shape}."
        )
        raise ValueError()

    if labels is not None:
        labels = np.asarray(labels)

        if labels.shape != x.shape:
            logger.error(
                "The given `labels` must have shape `(N,)`. "
                f"Received {labels.shape}."
            )
            raise ValueError()

    fig = go.Figure()

    if labels is None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    opacity=marker_opacity,
                ),
            )
        )

    else:
        # Preserve the original label order
        unique_labels = list(dict.fromkeys(labels))

        # Check palette existence
        if not hasattr(px.colors.qualitative, palette):
            available_palettes = [
                p
                for p in dir(px.colors.qualitative)
                if not p.startswith("_")
            ]

            logger.error(
                f"Unknown palette '{palette}'. "
                f"Available palettes: {available_palettes}"
            )
            raise ValueError(f"Unknown palette: {palette}")

        colors = getattr(px.colors.qualitative, palette)

        # Repeat colors if there are more classes than available colors
        colors = [
            colors[i % len(colors)]
            for i in range(len(unique_labels))
        ]

        label_to_color = dict(zip(unique_labels, colors))

        # Create one scatter trace per class
        for label in unique_labels:
            mask = labels == label

            fig.add_trace(
                go.Scatter(
                    x=x[mask],
                    y=y[mask],
                    mode="markers",
                    name=str(label),
                    marker=dict(
                        size=marker_size,
                        opacity=marker_opacity,
                        color=label_to_color[label],
                    ),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    if xticklabels is not None:
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(xticklabels))), ticktext=xticklabels)
    if yticklabels is not None:
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(yticklabels))), ticktext=yticklabels)

    if mlflow.active_run() is not None:
        mlflow.log_figure(fig, path)
    else:
        fig.write_image(path)

    return fig


def plot_histogram(
        x: np.ndarray,
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        path: str,
        nbins: Optional[int] = None,
        color: Optional[str] = None,
        opacity: float = 0.75,
        xticklabels: Optional[List[str]] = None,
        yticklabels: Optional[List[str]] = None,
    ) -> go.Figure:
    """Plot a histogram.

    Parameters
    ----------
    x : np.ndarray
        Data to plot. Shape `(N,)`.

    title : str
        Plot title.

    xaxis_title : str
        Title for the X-axis.

    yaxis_title : str
        Title for the Y-axis.

    path : str
        Path where the figure will be saved. If an MLFlow
        run is started, the figure is logged in the given `path`.

    nbins : Optional[int], default=None
        Number of bins. If `None`, Plotly selects the number
        of bins automatically.

    color : Optional[str], default=None
        Bar color as a CSS color string (e.g. `"steelblue"`).
        If `None`, the default Plotly color is used.

    opacity : float, default=0.75
        Bar opacity between 0 and 1.

    xticklabels : Optional[List[str]], default=None
        Custom labels for the X-axis ticks. If `None`, Plotly defaults are used.

    yticklabels : Optional[List[str]], default=None
        Custom labels for the Y-axis ticks. If `None`, Plotly defaults are used.

    Returns
    -------
    go.Figure
        Histogram figure.

    Examples
    --------
    Create a simple histogram:

    >>> import numpy as np
    >>> x = np.random.randn(500)
    >>> fig = plot_histogram(
    ...     x=x,
    ...     title="Example Histogram",
    ...     xaxis_title="Value",
    ...     yaxis_title="Count",
    ...     path="histogram.png",
    ... )

    Create a histogram with a fixed number of bins and a custom color:

    >>> fig = plot_histogram(
    ...     x=x,
    ...     title="Custom Histogram",
    ...     xaxis_title="Value",
    ...     yaxis_title="Count",
    ...     path="histogram_custom.png",
    ...     nbins=30,
    ...     color="steelblue",
    ...     opacity=0.8,
    ... )

    The function logs the figure to MLflow automatically if an MLflow
    run is active:

    >>> import mlflow
    >>> with mlflow.start_run():
    ...     plot_histogram(
    ...         x=x,
    ...         title="MLflow Histogram",
    ...         xaxis_title="Value",
    ...         yaxis_title="Count",
    ...         path="figures/histogram.png",
    ...     )
    """

    logger = get_formatted_logger(name=__name__, level="ERROR")

    x = np.asarray(x)

    if x.ndim != 1:
        logger.error(
            f"The given `x` must have shape `(N,)`. Received {x.shape}."
        )
        raise ValueError()

    marker_kwargs = dict(opacity=opacity)
    if color is not None:
        marker_kwargs["color"] = color

    fig = go.Figure(
        data=go.Histogram(
            x=x,
            nbinsx=nbins,
            marker=marker_kwargs,
            hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        bargap=0.05,
    )

    if xticklabels is not None:
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(xticklabels))), ticktext=xticklabels)
    if yticklabels is not None:
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(yticklabels))), ticktext=yticklabels)

    if mlflow.active_run() is not None:
        mlflow.log_figure(fig, path)
    else:
        fig.write_image(path)

    return fig
