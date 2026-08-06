"""Local (per-sample) plot definitions rendered to individual image files.

Provides a small composable framework (`get_plots_local` -> `plot_model` ->
`graph_model` -> `method_*`) for building multi-panel figures out of a
single sample's images and local metrics.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize


class get_plots_local:
    """Renders a set of local plots to disk for a single sample.

    Args:
        save_dir: Directory where the rendered plot JPEGs are saved.
        plot_set: Mapping from plot name to a `plot_model` (or compatible)
            instance exposing `size_plot_width`, `size_plot_height` and
            `create_plot`.
        nb_plots: Number of samples to plot (used upstream to select samples;
            kept here for reference).
        model_name: Model name used to build the output file names.
    """

    def __init__(
        self, save_dir: str, plot_set: Dict[str, Any], nb_plots: int, model_name: str
    ) -> None:
        self.save_dir = Path(save_dir).resolve()
        self.plot_set = plot_set
        self.nb_plots = nb_plots
        self.model_name = model_name

    def __call__(
        self,
        images: Dict[str, np.ndarray],
        metrics: Dict[str, Any],
        plot_name_sample: str,
        row: pd.Series,
    ) -> None:
        """Renders every configured plot to a JPEG file for this sample.

        Args:
            images: Dictionary of loaded/computed images for the sample.
            metrics: Dictionary of local metrics computed for the sample.
            plot_name_sample: Sample-specific name fragment used in the output filename.
            row: GeoDataFrame row corresponding to the sample.

        Returns:
            None.
        """
        for plot_name_base, plot in self.plot_set.items():
            fig = plt.figure(figsize=(plot.size_plot_width, plot.size_plot_height))
            plot.create_plot(images, metrics, row)
            plot_path = (
                self.save_dir / f"{plot_name_base}_{plot_name_sample}_{self.model_name}_plot.jpg"
            )
            fig.savefig(plot_path, bbox_inches="tight", pad_inches=0)
            plt.close()


class plot_model:
    """Arranges a list of graphs on a grid of subplots and lays out one figure.

    Args:
        graph_list: List of `graph_model` instances placed on the grid.
        nb_row: Number of rows in the subplot grid.
        nb_col: Number of columns in the subplot grid.
        size_plot_width: Figure width, in inches.
        size_plot_height: Figure height, in inches.
        hspace: Horizontal space between subplots.
        wspace: Width space between subplots.
    """

    def __init__(
        self,
        graph_list: list,
        nb_row: int,
        nb_col: int,
        size_plot_width: float,
        size_plot_height: float,
        hspace: float = 0.3,
        wspace: float = 0.3,
    ) -> None:
        self.graph_list = graph_list
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.size_plot_width = size_plot_width
        self.size_plot_height = size_plot_height
        self.hspace = hspace  # Horizontal space between subplots
        self.wspace = wspace  # Width space between subplots

    def create_plot(
        self, images: Dict[str, np.ndarray], metrics: Dict[str, Any], row: pd.Series
    ) -> None:
        """Creates every configured graph on the current figure's subplot grid.

        Args:
            images: Dictionary of loaded/computed images for the sample.
            metrics: Dictionary of local metrics computed for the sample.
            row: GeoDataFrame row corresponding to the sample.

        Returns:
            None.
        """
        for graph in self.graph_list:
            ax = plt.subplot2grid(
                (self.nb_row, self.nb_col),
                (graph.idx_row, graph.idx_col),
                rowspan=graph.rowspan,
                colspan=graph.colspan,
            )
            graph.create_graph(images, metrics, ax, row)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=self.hspace, wspace=self.wspace)


class graph_model:
    """Wraps a plotting method (`method_*`) and places it in a subplot with a title.

    Args:
        idx_row: Row index of the subplot in the grid.
        idx_col: Column index of the subplot in the grid.
        graph_title: Title displayed above the subplot; may contain `<date>`
            and/or `<year>` placeholders resolved from `grouping_dates`.
        method_graph: Callable `(images, metrics, ax, row)` that draws the graph.
        grouping_dates: Name of the row column holding the date used to
            resolve `<date>`/`<year>` placeholders in `graph_title`, or falsy
            to disable date-based title formatting.
        rowspan: Number of grid rows the subplot spans.
        colspan: Number of grid columns the subplot spans.
    """

    def __init__(
        self,
        idx_row: int,
        idx_col: int,
        graph_title: Optional[str],
        method_graph: Any,
        grouping_dates: Optional[str],
        rowspan: int = 1,
        colspan: int = 1,
    ) -> None:
        self.idx_row = idx_row
        self.idx_col = idx_col
        self.graph_title = graph_title
        self.method_graph = method_graph
        self.rowspan = rowspan
        self.colspan = colspan
        self.grouping_dates = grouping_dates

    def create_graph(
        self, images: Dict[str, np.ndarray], metrics: Dict[str, Any], ax: Axes, row: pd.Series
    ) -> None:
        """Draws the wrapped graph and sets its (optionally date-formatted) title.

        Args:
            images: Dictionary of loaded/computed images for the sample.
            metrics: Dictionary of local metrics computed for the sample.
            ax: Matplotlib axes to draw on.
            row: GeoDataFrame row corresponding to the sample; used to resolve
                `grouping_dates` for the title.

        Returns:
            None.
        """
        self.method_graph(images, metrics, ax, row)

        date = row[self.grouping_dates] if self.grouping_dates else None

        if self.graph_title:
            graph_title = self.graph_title
            if "<date>" in graph_title:
                graph_title = graph_title.replace("<date>", date)
            if "<year>" in graph_title:
                graph_title = graph_title.replace("<year>", date[:4])

            ax.set_title(graph_title, fontsize=16)


class method_imshow:
    """Displays an image, optionally center-cropped to a real-world patch size.

    Args:
        image_name: Name of the image to display in `images`.
        real_patch_size: If set (with `resolution`), the image is center-cropped
            to this real-world size before display.
        resolution: Spatial resolution of a pixel, used to convert
            `real_patch_size` to a pixel patch size.
        cmap: Matplotlib colormap used for display.
        norm: Matplotlib normalization used for display.
    """

    def __init__(
        self,
        image_name: str,
        real_patch_size: Optional[float],
        resolution: Optional[float],
        cmap: Optional[str] = None,
        norm: Optional[Normalize] = None,
    ) -> None:
        self.image_name = image_name
        self.real_patch_size = real_patch_size
        self.resolution = resolution
        if self.real_patch_size is not None and self.resolution is not None:
            self.patch_size = self.real_patch_size / self.resolution
        else:
            self.patch_size = None
        self.cmap = cmap
        self.norm = norm

    def __call__(
        self, images: Dict[str, np.ndarray], metrics: Dict[str, Any], ax: Axes, row: pd.Series
    ) -> None:
        """Displays the (optionally cropped) image on the given axes.

        Args:
            images: Dictionary of loaded/computed images, must contain `image_name`.
            metrics: Dictionary of local metrics computed for the sample (unused).
            ax: Matplotlib axes to draw on.
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            None.
        """
        if self.image_name not in images:
            Exception(f"You must first load {self.image_name}")
        image = images[self.image_name]

        if self.real_patch_size is not None:
            # We crop the image to the patch size percentage
            height, width = image.shape[-2], image.shape[-1]
            if self.patch_size is not None:
                x_start = int(height / 2 - self.patch_size / 2)  # we center the crop
                x_stop = int(x_start + self.patch_size)
                y_start = int(width / 2 - self.patch_size / 2)  # we center the crop
                y_stop = int(y_start + self.patch_size)

                image = image[..., x_start:x_stop, y_start:y_stop]

            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
                gain = 0.4 / image.mean()
                image = np.clip(image * gain, 0, 1)

            # Check if the image has only one unique value and modify if needed
            if "mask" in self.image_name:
                first_value = image[..., 0, 0]
                if not np.any(image != first_value):
                    image[..., 0, 0] = 0
        ax.imshow(image, cmap=self.cmap, norm=self.norm)
        ax.axis("off")


class method_bar:
    """Draws a simple bar chart of a fixed list of local metrics.

    Args:
        metrics_list: List of metric names to plot as bars, in order.
        y_min: Lower y-axis limit.
        y_max: Upper y-axis limit.
    """

    def __init__(self, metrics_list: list, y_min: Optional[float], y_max: Optional[float]) -> None:
        self.metrics_list = metrics_list
        self.y_min = y_min
        self.y_max = y_max

    def __call__(
        self, images: Dict[str, np.ndarray], metrics: Dict[str, Any], ax: Axes, row: pd.Series
    ) -> None:
        """Draws the bar chart on the given axes.

        Args:
            images: Dictionary of loaded/computed images for the sample (unused).
            metrics: Dictionary of local metrics, must contain every name in `metrics_list`.
            ax: Matplotlib axes to draw on.
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            None.
        """
        for key in self.metrics_list:
            if key not in metrics:
                Exception(f"You must first compute metric {key}")
        ax.bar(self.metrics_list, [metrics[key] for key in self.metrics_list])
        ax.set_ylim(self.y_min, self.y_max)
        ax.grid(color="gray", linestyle="dashed", axis="y")
        plt.xticks(rotation=40)


class method_table:
    """Draws a two-column table of metric names and values.

    Args:
        metrics_list: List of metric names to display, in order.
        font_size: Font size used for the table text.
        table_width_scale: Accepted for interface consistency; not used as the
            table width scale, which is hardcoded to 0.7.
        table_height_scale: Accepted for interface consistency; not used as the
            table height scale, which is hardcoded to 1.8.
    """

    def __init__(
        self,
        metrics_list: list,
        font_size: float,
        table_width_scale: float,
        table_height_scale: float,
    ) -> None:
        self.metrics_list = metrics_list
        self.font_size = font_size
        self.table_width_scale = 0.7
        self.table_height_scale = 1.8

    def __call__(
        self, images: Dict[str, np.ndarray], metrics: Dict[str, Any], ax: Axes, row: pd.Series
    ) -> None:
        """Draws the metrics table on the given axes.

        Args:
            images: Dictionary of loaded/computed images for the sample (unused).
            metrics: Dictionary of local metrics, must contain every name in `metrics_list`.
            ax: Matplotlib axes to draw on.
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            None.
        """
        for key in self.metrics_list:
            if key not in metrics:
                Exception(f"You must first compute metric {key}")

        # Create table data with metric names and values
        table_data = []
        for metric_name in self.metrics_list:
            value = metrics[metric_name]
            # Format the value to 3 decimal places if it's a float
            if isinstance(value, float):
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
            table_data.append([metric_name, formatted_value])

        # Create the table
        table = ax.table(
            cellText=table_data, colLabels=["Metric", "Value"], cellLoc="center", loc="center"
        )
        table.set_fontsize(self.font_size)
        table.scale(self.table_width_scale, self.table_height_scale)

        ax.axis("off")
