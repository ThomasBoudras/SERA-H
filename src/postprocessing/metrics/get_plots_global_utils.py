"""Global plot definitions built from aggregated metrics.

Provides a small composable framework (`get_plots_global` -> `plot_model` ->
`graph_model` -> `method_*`) for building multi-panel figures out of the
global metrics dictionary, and returns any per-graph metrics extracted while
plotting (e.g. R2, MAE) for optional export.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from scipy.interpolate import interpn
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

log = logging.getLogger(__name__)


class get_plots_global:
    """Renders a set of global plots to disk and collects any metrics they extract.

    Args:
        save_dir: Directory where the rendered plot PDFs are saved.
        plot_set: Mapping from plot name to a `plot_model` (or compatible)
            instance exposing `size_plot_width`, `size_plot_height` and
            `create_plot`.
        model_name: Model name used to build the output file names.
        metrics_to_save: Either `"all"`, a list of metric names to keep, or
            falsy to save nothing.
        output_path_xlsx: Path to the Excel file where selected metrics are saved.
    """

    def __init__(
        self,
        save_dir: str,
        plot_set: Dict[str, Any],
        model_name: str,
        metrics_to_save: Union[str, list, bool] = False,
        output_path_xlsx: Optional[str] = None,
    ) -> None:
        self.save_dir = Path(save_dir).resolve()
        self.plot_set = plot_set
        self.model_name = model_name
        self.metrics_to_save = metrics_to_save
        self.output_path_xlsx = output_path_xlsx

    def __call__(self, metrics_global: Dict[str, Any]) -> Dict[str, Any]:
        """Renders every configured plot to a PDF file and collects their extracted metrics.

        Args:
            metrics_global: Dictionary of global metrics available to the plots.

        Returns:
            Dictionary mapping `{plot_name}_{key}` to the metrics extracted by
            each plot.
        """
        global_plot_metrics = {}
        for plot_name_base, plot in self.plot_set.items():
            fig = plt.figure(figsize=(plot.size_plot_width, plot.size_plot_height))
            plot_metrics = plot.create_plot(metrics_global)
            plot_path = self.save_dir / f"{plot_name_base}_{self.model_name}_plot.pdf"
            fig.savefig(plot_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            for key, value in plot_metrics.items():
                global_plot_metrics[f"{plot_name_base}_{key}"] = value

        return global_plot_metrics


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
        graph_list: List[Any],
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

    def create_plot(self, metrics_global: Dict[str, Any]) -> Dict[str, Any]:
        """Creates every configured graph on the current figure's subplot grid.

        Args:
            metrics_global: Dictionary of global metrics available to the graphs.

        Returns:
            Dictionary mapping `{graph_title}_{key}` to the metrics extracted by
            each graph.
        """
        global_graph_metrics = {}
        for graph in self.graph_list:
            ax = plt.subplot2grid(
                (self.nb_row, self.nb_col),
                (graph.idx_row, graph.idx_col),
                rowspan=graph.rowspan,
                colspan=graph.colspan,
            )
            graph_metrics = graph.create_graph(metrics_global, ax)

            for key, value in graph_metrics.items():
                global_graph_metrics[f"{graph.graph_title}_{key}"] = value

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=self.hspace, wspace=self.wspace)

        return global_graph_metrics


class graph_model:
    """Wraps a plotting method (`method_*`) and places it in a subplot with a title.

    Args:
        idx_row: Row index of the subplot in the grid.
        idx_col: Column index of the subplot in the grid.
        graph_title: Title displayed above the subplot, or falsy to skip it.
        method_graph: Callable `(metrics_global, ax) -> Optional[dict]` that
            draws the graph and optionally returns extracted metrics.
        rowspan: Number of grid rows the subplot spans.
        colspan: Number of grid columns the subplot spans.
        bold_title: Option to set the graph title in bold, defaults to False.
    """

    def __init__(
        self,
        idx_row: int,
        idx_col: int,
        graph_title: Optional[str],
        method_graph: Any,
        rowspan: int = 1,
        colspan: int = 1,
        bold_title: bool = False,
    ) -> None:
        self.idx_row = idx_row
        self.idx_col = idx_col
        self.graph_title = graph_title
        self.method_graph = method_graph
        self.rowspan = rowspan
        self.colspan = colspan
        self.bold_title = bold_title  # Option to set the graph title in bold, defaults to False

    def create_graph(self, metrics_global: Dict[str, Any], ax: Axes) -> Dict[str, Any]:
        """Draws the wrapped graph and sets its title.

        Args:
            metrics_global: Dictionary of global metrics available to the graph.
            ax: Matplotlib axes to draw on.

        Returns:
            Dictionary of metrics extracted by the graph, or an empty dict if
            `method_graph` returns None.
        """
        graph_metrics = self.method_graph(metrics_global, ax)
        if self.graph_title:
            graph_title = self.graph_title
            if self.bold_title:
                ax.set_title(graph_title, fontweight="bold")
            else:
                ax.set_title(graph_title)

        if graph_metrics is not None:
            return graph_metrics
        return {}


class method_bar:
    """Draws a simple bar chart of a fixed list of global metrics.

    Args:
        metrics_list: List of metric names to plot as bars, in order.
        y_min: Lower y-axis limit.
        y_max: Upper y-axis limit.
    """

    def __init__(
        self, metrics_list: List[str], y_min: Optional[float], y_max: Optional[float]
    ) -> None:
        self.metrics_list = metrics_list
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, metrics_global: Dict[str, Any], ax: Axes) -> None:
        """Draws the bar chart on the given axes.

        Args:
            metrics_global: Dictionary of global metrics, must contain every
                name in `metrics_list`.
            ax: Matplotlib axes to draw on.

        Returns:
            None.

        Raises:
            Exception: If a metric in `metrics_list` is missing from `metrics_global`.
        """
        for key in self.metrics_list:
            if key not in metrics_global:
                raise Exception(f"You must first compute metric {key}")
        ax.bar(self.metrics_list, [metrics_global[key] for key in self.metrics_list])
        ax.set_ylim(self.y_min, self.y_max)
        ax.grid(color="gray", linestyle="dashed", axis="y")
        plt.xticks(rotation=40)

        return None


class method_curve_by_bins_intervals:
    """Draws one or more curves of a metric evaluated over a series of bins.

    Args:
        metric_names: Metric name or list of metric names to plot; each yields
            one curve, read as `{metric_name}_{bin}` for every bin.
        bins: Sorted list of bin edges.
        x_label: X-axis label, or None to skip it.
        y_label: Y-axis label, or None to skip it.
        colors: List of colors, cycled across curves.
        legend_bins: Tick labels displayed for each bin on the x-axis.
        y_min: Lower y-axis limit.
        y_max: Upper y-axis limit.
        log_scale: If True, uses a logarithmic y-axis.
        labels: Legend labels for each curve (used only if plotting more than one curve).
        legend_loc: Matplotlib legend location.
        include_outer_bins: If True, adds an unbounded bin below the first edge
            and above the last edge.
    """

    def __init__(
        self,
        metric_names: Union[str, List[str]],
        bins: list,
        x_label: Optional[str],
        y_label: Optional[str],
        colors: list,
        legend_bins: list,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        log_scale: bool = False,
        labels: Optional[List[str]] = None,
        legend_loc: str = "upper right",
        include_outer_bins: bool = True,
    ) -> None:
        self.metric_names = metric_names
        self.bins = [None] + bins + [None] if include_outer_bins else bins
        self.x_label = x_label
        self.y_label = y_label
        self.legend_bins = legend_bins
        self.y_min = y_min
        self.y_max = y_max
        self.log_scale = log_scale
        self.labels = labels
        self.colors = colors
        self.legend_loc = legend_loc

    def __call__(self, metrics_global: Dict[str, Any], ax: Axes) -> Dict[str, Any]:
        """Draws one curve per metric name, with one point per bin.

        Args:
            metrics_global: Dictionary of global metrics, read as
                `{metric_name}_{bin}` for each metric name and bin.
            ax: Matplotlib axes to draw on.

        Returns:
            Dictionary mapping `{metric_name}_{bin}` to the plotted value.
        """
        # X positions are defined once, based on the number of bins
        x_positions = range(len(self.legend_bins))

        graph_metrics = {}
        for i, metric_name in enumerate(self.metric_names):
            metric_values = []

            for bin_idx in range(len(self.bins) - 1):
                min_bin_current = self.bins[bin_idx]
                max_bin_current = self.bins[bin_idx + 1]
                max_bin_current_str = "" if max_bin_current is None else str(int(max_bin_current))
                min_bin_current_str = "" if min_bin_current is None else str(int(min_bin_current))
                name_bin = f"{min_bin_current_str}-{max_bin_current_str}"

                metric_key = f"{metric_name}_{name_bin}"
                metric_value = metrics_global.get(metric_key, 0)
                metric_values.append(metric_value)
                graph_metrics[f"{metric_key}"] = metric_value

            # Select the color and label for this curve
            current_color = self.colors[i % len(self.colors)]
            current_label = (
                self.labels[i] if self.labels is not None and len(self.labels) > 1 else None
            )

            ax.plot(
                x_positions,
                metric_values,
                marker="o",
                linestyle="-",
                color=current_color,
                label=current_label,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(self.legend_bins, rotation=45, ha="right")

        if self.x_label is not None:
            ax.set_xlabel(self.x_label, fontsize=13)

        if self.y_label is not None:
            ax.set_ylabel(self.y_label, fontsize=13)

        if self.y_min is not None and self.y_max is not None:
            ax.set_ylim(self.y_min, self.y_max)

        if self.log_scale:
            ax.set_yscale("log")

        ax.grid(color="#C4C4C4", linestyle="--", axis="y", linewidth=0.7, zorder=1)

        # Show the legend only when there is more than one curve
        if len(self.metric_names) > 1:
            ax.legend(loc=self.legend_loc, frameon=False)

        return graph_metrics


class method_curve_by_values:
    """Draws one or more curves of a metric evaluated at a fixed list of values.

    Args:
        metric_names: List of metric names to plot; each yields one curve,
            read as `{metric_name}_{value}` for every value.
        values: List of numeric values used to build the x-axis and the metric keys.
        legend_values: Tick labels displayed for each value on the x-axis.
        x_label: X-axis label, or None to skip it.
        y_label: Y-axis label, or None to skip it.
        colors: List of colors, cycled across curves, or None for matplotlib defaults.
        y_min: Lower y-axis limit.
        y_max: Upper y-axis limit.
        labels: Legend labels for each curve (used only if plotting more than one curve).
        legend_loc: Matplotlib legend location.
    """

    def __init__(
        self,
        metric_names: List[str],
        values: list,
        legend_values: list,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        colors: Optional[list] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        labels: Optional[List[str]] = None,
        legend_loc: str = "upper right",
    ) -> None:
        self.metric_names = metric_names
        self.values = values
        self.legend_values = legend_values
        self.x_label = x_label
        self.y_label = y_label
        self.colors = colors
        self.y_min = y_min
        self.y_max = y_max
        self.labels = labels
        self.legend_loc = legend_loc

    def __call__(self, metrics_global: Dict[str, Any], ax: Axes) -> Dict[str, Any]:
        """Draws one curve per metric name, with one point per value.

        Args:
            metrics_global: Dictionary of global metrics, read as
                `{metric_name}_{value}` for each metric name and value.
            ax: Matplotlib axes to draw on.

        Returns:
            Dictionary mapping `{metric_name}_{value}` to the plotted value.
        """
        x_positions = range(len(self.values))

        graph_metrics = {}
        for i, metric_name in enumerate(self.metric_names):
            metric_values = []

            for value_current in self.values:
                name_value_current = str(int(value_current))

                metric_key = f"{metric_name}_{name_value_current}"
                metric_value = metrics_global.get(metric_key, 0)
                graph_metrics[f"{metric_key}"] = metric_value
                metric_values.append(metric_value)

            # Cycle through the color list if there are fewer colors than curves
            color = self.colors[i % len(self.colors)] if self.colors is not None else None
            label = self.labels[i] if self.labels is not None and len(self.labels) > 1 else None

            ax.plot(
                x_positions, metric_values, marker="o", linestyle="-", color=color, label=label
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(self.legend_values, rotation=45, ha="right")

        if self.x_label:
            ax.set_xlabel(self.x_label, fontsize=13)

        if self.y_label:
            ax.set_ylabel(self.y_label, fontsize=13)

        if self.y_min is not None and self.y_max is not None:
            ax.set_ylim(self.y_min, self.y_max)

        ax.grid(color="#C4C4C4", linestyle="--", axis="y", linewidth=0.7, zorder=1)

        # Show the legend only when there is more than one curve
        if len(self.metric_names) > 1:
            ax.legend(loc=self.legend_loc, frameon=False)

        return graph_metrics


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
        metrics_list: List[str],
        font_size: float,
        table_width_scale: float,
        table_height_scale: float,
    ) -> None:
        self.metrics_list = metrics_list
        self.font_size = font_size
        self.table_width_scale = 0.7
        self.table_height_scale = 1.8

    def __call__(self, metrics_global: Dict[str, Any], ax: Axes) -> None:
        """Draws the metrics table on the given axes.

        Args:
            metrics_global: Dictionary of global metrics, must contain every
                name in `metrics_list`.
            ax: Matplotlib axes to draw on.

        Returns:
            None.

        Raises:
            Exception: If a metric in `metrics_list` is missing from `metrics_global`.
        """
        for key in self.metrics_list:
            if key not in metrics_global:
                raise Exception(f"You must first compute metric {key}")

        # Create table data with metric names and values
        table_data = []
        for metric_name in self.metrics_list:
            value = metrics_global[metric_name]
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

        return None


class method_scatter_density:
    """Draws a density-colored scatter plot of two arrays with optional fit/1:1 lines.

    Args:
        metric_name: Name of the global metric holding an `(x, y)` pair of arrays.
        x_label: X-axis label, or falsy to skip it.
        y_label: Y-axis label, or falsy to skip it.
        min_range: Lower bound of the shared x/y plot range.
        max_range: Upper bound of the shared x/y plot range.
        bins: Number of bins (per axis) used for the 2D density histogram.
        metrics_plot: List of metric names to display as text on the plot;
            supported values are `"r2"`, `"mae"`, `"rmse"` and `"me"`.
        max_points_on_scatter: Maximum number of points drawn on the scatter
            plot; points are randomly subsampled above this count.
        point_size: Marker size for the scatter points.
        x_y_line: If True, draws a 1:1 reference line.
        fit_line: If True, draws a linear regression fit line.
        show_legend: Unused parameter kept for interface consistency.
        add_noise: If True, adds uniform noise in `[-0.5, 0.5]` to the y values
            for visualization purposes only (not used in the computed metrics).
        min_normalization: Lower bound for the density color normalization, or
            None to derive it from the data (1st percentile).
        max_normalization: Upper bound for the density color normalization, or
            None to derive it from the data (75th percentile).
    """

    def __init__(
        self,
        metric_name: str,
        x_label: Optional[str],
        y_label: Optional[str],
        min_range: float,
        max_range: float,
        bins: int,
        metrics_plot: List[str],
        max_points_on_scatter: int,
        point_size: float,
        x_y_line: bool,
        fit_line: bool,
        show_legend: bool,
        add_noise: bool,
        min_normalization: Optional[float],
        max_normalization: Optional[float],
    ) -> None:
        self.metric_name = metric_name
        self.x_label = x_label
        self.y_label = y_label
        self.range_fig = (min_range, max_range)
        self.bins = bins
        self.metrics_plot = metrics_plot
        self.max_points_on_scatter = max_points_on_scatter
        self.point_size = point_size
        self.x_y_line = x_y_line
        self.fit_line = fit_line
        self.show_legend = show_legend
        self.add_noise = add_noise
        self.min_normalization = min_normalization
        self.max_normalization = max_normalization

    def __call__(self, metrics_global: Dict[str, Any], ax: Axes) -> Dict[str, float]:
        """Draws the density scatter plot and computes the requested text metrics.

        Args:
            metrics_global: Dictionary of global metrics, must contain
                `metric_name` as an `(x, y)` pair of arrays.
            ax: Matplotlib axes to draw on.

        Returns:
            Dictionary mapping each requested metric in `metrics_plot` (e.g.
            `"r2"`, `"mae"`, `"rmse"`, `"me"`) to its computed value.
        """

        # Extract x and y values from the specified metric in metrics_global.
        x_init = metrics_global[self.metric_name][0]
        y_init = metrics_global[self.metric_name][1]

        # Remove NaNs for metric calculation
        valid_indices = ~np.isnan(x_init) & ~np.isnan(y_init)
        x_init = x_init[valid_indices]
        y_init = y_init[valid_indices]

        if self.add_noise:
            # Add uniform noise in the range [-0.5, 0.5] to x and y for visualization
            x = x_init
            y = y_init + np.random.uniform(-0.5, 0.5, size=y_init.shape)
        else:
            x, y = x_init, y_init

        # Calculate density using a 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            x, y, bins=self.bins, range=[self.range_fig, self.range_fig], density=True
        )
        z = interpn(
            (x_edges[:-1], y_edges[:-1]),
            hist,
            np.vstack([x, y]).T,
            method="splinef2d",
            bounds_error=False,
            fill_value=0,
        )

        # Sort points by density for better plotting
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        # Check if the number of points exceeds the maximum allowed for scatter plotting.
        if len(x) > self.max_points_on_scatter:
            # If so, generate a random sample of indices.
            idx = np.random.choice(np.arange(len(x)), self.max_points_on_scatter, replace=False)
            x = x[idx]
            y = y[idx]
            z = z[idx]

        # Set plot limits and aspect ratio.
        ax.set_xlim(self.range_fig)
        ax.set_ylim(self.range_fig)
        ax.set_aspect("equal", adjustable="box")

        # Create the scatter plot with density coloring
        if self.min_normalization is None:
            min_normalization = np.quantile(z, 0.01) + 1e-5

            log.info(f"Min normalization for scatter plot: {min_normalization}")
        else:
            min_normalization = self.min_normalization
        if self.max_normalization is None:
            max_normalization = np.quantile(z, 0.75)
            log.info(f"Max normalization for scatter plot: {max_normalization}")
        else:
            max_normalization = self.max_normalization

        norm = colors.LogNorm(vmin=min_normalization, vmax=max_normalization, clip=True)

        ax.scatter(
            x,
            y,
            c=z,
            s=self.point_size,
            cmap="inferno",
            norm=norm,
            edgecolor="none",
            rasterized=True,
        )

        # Add a 1:1 line for reference if specified.
        if self.x_y_line:
            ax.plot(self.range_fig, self.range_fig, "k--", label="1:1")

        # Add a linear fit line if specified.
        if self.fit_line:
            # Perform linear regression
            slope, intercept, _, _, _ = linregress(x_init, y_init)

            # Generate the fit line
            fit_line = slope * np.array(self.range_fig) + intercept
            ax.plot(
                self.range_fig,
                fit_line,
                color="blue",
                linestyle="dotted",
                linewidth=1.5,
                label="Linear fit",
            )

        # Add axis labels
        if self.x_label:
            ax.set_xlabel(self.x_label)
        if self.y_label:
            ax.set_ylabel(self.y_label)

        # Build the text for metrics to be displayed on the plot.
        metrics_text = []
        graph_metrics = {}
        if "r2" in self.metrics_plot:
            r2 = r2_score(x_init, y_init)
            metrics_text.append(f"$R^2$ = {r2:.3f}")
            graph_metrics["r2"] = r2
        if "mae" in self.metrics_plot:
            mae = mean_absolute_error(x_init, y_init)
            metrics_text.append(f"MAE = {mae:.3f}")
            graph_metrics["mae"] = mae
        if "rmse" in self.metrics_plot:
            rmse = np.sqrt(mean_squared_error(x_init, y_init))
            metrics_text.append(f"RMSE = {rmse:.3f}")
            graph_metrics["rmse"] = rmse
        if "me" in self.metrics_plot:
            me = np.mean(y_init - x_init)  # Prediction - Target
            metrics_text.append(f"ME = {me:.3f}")
            graph_metrics["me"] = me

        # Display the metrics in a text box.
        if metrics_text:
            ax.text(
                self.range_fig[1] - 0.04 * (self.range_fig[1] - self.range_fig[0]),
                self.range_fig[0] + 0.1 * (self.range_fig[1] - self.range_fig[0]),
                "\n".join(metrics_text),
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    facecolor="white", alpha=0.9, edgecolor="none", boxstyle="square,pad=0.5"
                ),
            )

        return graph_metrics


class method_boxplot_by_bins:
    """Draws grouped boxplots of one or more metrics across a series of bins.

    Optionally overlays a secondary bar plot showing the proportion of values
    in each bin for a set of proportion metrics.

    Args:
        metrics_list: List of metric names to plot; each holds a list of
            per-bin arrays of values.
        proportion_metric: Iterable of metric names (each a list of per-bin
            arrays) used to compute the proportion overlay; pass an empty
            iterable to disable the overlay.
        legend_labels: Legend label for each metric in `metrics_list`.
        bins: List of bin labels (used only for tick count and labeling).
        x_label: X-axis label, or None to skip it.
        y_label: Y-axis label, or None to skip it.
        colors: Color palette name, list of colors, or None for the default "Set2" palette.
        legends_and_labels_shown: If False, hides ticks, labels, legend and spines.
        y_min: Lower y-axis limit.
        y_max: Upper y-axis limit.
        max_points_per_bin: Maximum number of points randomly kept per bin
            before drawing the boxplot, or None to keep all.
        legend_loc: Matplotlib legend location.
    """

    def __init__(
        self,
        metrics_list: List[str],
        proportion_metric: list,
        legend_labels: Optional[List[str]],
        bins: list,
        x_label: Optional[str],
        y_label: Optional[str],
        colors: Union[str, list, None],
        legends_and_labels_shown: bool,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        max_points_per_bin: Optional[int] = None,
        legend_loc: str = "upper right",
    ) -> None:
        self.metrics_list = metrics_list
        self.proportion_metric = list(proportion_metric)
        self.legend_labels = legend_labels
        self.bins = bins
        self.x_label = x_label
        self.y_label = y_label
        self.colors = colors
        self.y_min = y_min
        self.y_max = y_max
        self.legends_and_labels_shown = legends_and_labels_shown
        self.max_points_per_bin = max_points_per_bin
        self.legend_loc = legend_loc

    def __call__(self, metrics_global: Dict[str, Any], ax: Axes) -> Dict[str, Any]:
        """Draws the grouped boxplots (and optional proportion overlay) on the given axes.

        Args:
            metrics_global: Dictionary of global metrics, must contain every
                name in `metrics_list` (and `proportion_metric` if set).
            ax: Matplotlib axes to draw on.

        Returns:
            Dictionary of per-bin summary statistics (q1, q3, median, whiskers,
            count, mean, min, max) for each metric and bin.

        Raises:
            Exception: If a metric in `metrics_list` or `proportion_metric` is
                missing from `metrics_global`.
        """
        # Ensure all provided metric names are present in the global metrics dictionary
        for metric_name in self.metrics_list:
            if metric_name not in metrics_global:
                raise Exception(f"Metric '{metric_name}' not found.")
        # Check that the proportion metric exists if it is specified
        if self.proportion_metric is not None:
            for pm in self.proportion_metric:
                if pm not in metrics_global:
                    raise Exception(f"Proportion metric '{pm}' not found.")

        num_metrics = len(self.metrics_list)
        positions = np.arange(len(self.bins))  # x positions for boxplots
        box_width = 0.8 / max(
            num_metrics, 1
        )  # width of each boxplot depending on number of metrics

        # Select default color palette if colors is None, otherwise use provided palette
        if self.colors is None:
            palette = sns.color_palette("Set2", num_metrics)
        elif isinstance(self.colors, str):
            palette = sns.color_palette(self.colors, num_metrics)
        else:
            palette = self.colors

        legend_handles = []
        graph_metrics = {}
        for i, metric_name in enumerate(self.metrics_list):
            data_to_plot_full = metrics_global[metric_name]

            if self.max_points_per_bin is not None:
                data_to_plot = []
                for bin_data in data_to_plot_full:
                    if len(bin_data) > self.max_points_per_bin:
                        # Using np.random.choice for efficient sampling
                        sampled_data = np.random.choice(
                            np.array(bin_data), self.max_points_per_bin, replace=False
                        )
                        data_to_plot.append(sampled_data)
                    else:
                        data_to_plot.append(bin_data)
            else:
                data_to_plot = data_to_plot_full

            # Calculate boxplot position offset so boxes don't overlap for multiple metrics
            offset = (i - (num_metrics - 1) / 2) * box_width
            current_positions = positions + offset
            color = palette[i % len(palette)]

            # Create the boxplot for this metric
            bp = ax.boxplot(
                data_to_plot,
                positions=current_positions,
                widths=box_width * 0.9,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor=color, linewidth=1.5, alpha=0.7),
                medianprops=dict(color="black", linewidth=1.0),
                whiskerprops=dict(color=color, linewidth=1.3),
                capprops=dict(color=color, linewidth=1.3),
                zorder=3,
            )

            # Style the whiskers and caps for clarity
            for whisker in bp["whiskers"]:
                whisker.set_linestyle("-")
            for cap in bp["caps"]:
                cap.set_alpha(0.9)

            # Add corresponding legend entry if legend labels are provided
            if (
                self.legend_labels is not None
                and self.legends_and_labels_shown
                and i < len(self.legend_labels)
            ):
                legend_handles.append(
                    mpatches.Patch(
                        facecolor=color, edgecolor=color, alpha=0.7, label=self.legend_labels[i]
                    )
                )

            name_graph_metrics = (
                f"{metric_name}_{self.legend_labels[i]}"
                if self.legend_labels and i < len(self.legend_labels)
                else metric_name
            )

            for j, bin_name in enumerate(self.bins):
                if len(data_to_plot[j]) == 0:
                    continue
                # When patch_artist=True, boxes are PathPatch objects, not Line2D.
                # We can get Q1 and Q3 from the path vertices.
                path = bp["boxes"][j].get_path()
                vertices = path.vertices

                bin_str = str(bin_name)

                graph_metrics[f"{name_graph_metrics}_{bin_str}_q1"] = np.min(vertices[:, 1])
                graph_metrics[f"{name_graph_metrics}_{bin_str}_q3"] = np.max(vertices[:, 1])

                graph_metrics[f"{name_graph_metrics}_{bin_str}_median"] = bp["medians"][
                    j
                ].get_ydata()[0]
                graph_metrics[f"{name_graph_metrics}_{bin_str}_lower_whisker"] = bp["whiskers"][
                    j * 2
                ].get_ydata()[1]
                graph_metrics[f"{name_graph_metrics}_{bin_str}_upper_whisker"] = bp["whiskers"][
                    j * 2 + 1
                ].get_ydata()[1]

                graph_metrics[f"{name_graph_metrics}_{bin_str}_nb_values"] = len(data_to_plot[j])
                graph_metrics[f"{name_graph_metrics}_{bin_str}_mean"] = np.mean(data_to_plot[j])
                graph_metrics[f"{name_graph_metrics}_{bin_str}_min"] = np.min(data_to_plot[j])
                graph_metrics[f"{name_graph_metrics}_{bin_str}_max"] = np.max(data_to_plot[j])

        # Set the x-tick positions at the center of each bin
        ax.set_xticks(positions)

        # Set the label names for bins, rotated for better readability
        if self.legends_and_labels_shown:
            ax.set_xticklabels(self.bins, rotation=40, ha="right")
        else:
            ax.set_xticklabels([])

        # Set the x and y axis labels if provided
        if self.x_label is not None and self.legends_and_labels_shown:
            ax.set_xlabel(self.x_label, fontsize=13)

        if self.y_label is not None and self.legends_and_labels_shown:
            ax.set_ylabel(self.y_label, fontsize=13)

        if not self.legends_and_labels_shown:
            ax.set_yticklabels([])

        # Optionally set y-axis limits if provided
        if self.y_min is not None and self.y_max is not None:
            ax.set_ylim(self.y_min, self.y_max)

        ax.tick_params(axis="y", direction="in", length=6)
        ax.tick_params(axis="x", direction="in", length=6)

        # Draw a horizontal line at y=0 for reference
        ax.axhline(0, color="black", linewidth=1.0, linestyle="-", zorder=0)
        # Add grid lines for better readability
        ax.grid(color="#C4C4C4", linestyle="--", axis="y", linewidth=0.7, zorder=1)
        ax.set_facecolor("white")

        # Hide unnecessary plot spines for a cleaner look
        if not self.legends_and_labels_shown:
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(False)

        # Add the legend to the plot if any legend handles were created
        if legend_handles and self.legends_and_labels_shown:
            legend = ax.legend(handles=legend_handles, loc=self.legend_loc, frameon=False)
            legend.set_zorder(4)

        # If a proportion metric is specified, display a bar overlay for counts/proportions
        if self.proportion_metric is not None:
            ax2 = ax.twinx()

            # Compute the number of values in each bin for all proportion metrics
            all_proportions = []
            for pm in self.proportion_metric:
                proportion_data = metrics_global[pm]
                proportions = [len(data) for data in proportion_data]
                all_proportions.append(proportions)

            # Compute the percentage that each bin represents across ALL proportion metrics
            total_values = sum(sum(props) for props in all_proportions)

            num_prop_metrics = len(self.proportion_metric)
            bar_width = 0.75 / max(num_prop_metrics, 1)

            max_prop_percentage = 0

            for j, pm in enumerate(self.proportion_metric):
                proportions = all_proportions[j]
                proportions_percentage = [
                    prop / total_values * 100 if total_values > 0 else 0 for prop in proportions
                ]

                if proportions_percentage:
                    max_prop_percentage = max(max_prop_percentage, max(proportions_percentage))

                # Offset for multibar
                offset = (j - (num_prop_metrics - 1) / 2) * bar_width
                current_positions = positions + offset

                # Use the same color as the boxplot if the metric is in metrics_list
                if pm in self.metrics_list:
                    color_idx = self.metrics_list.index(pm)
                else:
                    color_idx = j

                color = palette[color_idx % len(palette)]

                # Overlay a bar plot showing the proportions for each bin
                ax2.bar(
                    current_positions,
                    proportions_percentage,
                    alpha=0.3,
                    color=color,
                    width=bar_width * 0.95,
                    edgecolor="none",
                )

            if self.legends_and_labels_shown:
                ax2.set_ylabel("Proportion (%)", fontsize=13)

            ax2.tick_params(axis="y", direction="in", length=6)
            if not self.legends_and_labels_shown:
                ax2.set_yticklabels([])

            # Hide all spines for the secondary axis as well
            if not self.legends_and_labels_shown:
                for spine_pos in ["top", "right", "left", "bottom"]:
                    ax2.spines[spine_pos].set_visible(False)

            # Set y-axis range for the secondary axis based on the data
            if max_prop_percentage > 0:
                ax2.set_ylim(0, max_prop_percentage * 1.25)
            else:
                ax2.set_ylim(0, 1)
            # Avoid grid on the overlay axis for clarity
            ax2.grid(False)

            # Ensure ax is drawn on top of ax2 and has a transparent background
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)

        return graph_metrics


class method_mae_curve_by_bins:
    """Draws MAE curves per bin for one or more metrics, with an optional proportion overlay.

    Args:
        metrics_list: List of metric names to plot; each holds a list of
            per-bin arrays of values from which the MAE is computed.
        proportion_metric: Iterable of metric names (each a list of per-bin
            arrays) used to compute the proportion overlay; pass an empty
            iterable to disable the overlay.
        legend_labels: Legend label for each metric in `metrics_list`.
        bins: List of bin labels (used only for tick count and labeling).
        x_label: X-axis label, or None to skip it.
        y_label: Y-axis label, or None to skip it.
        colors: Color palette name, list of colors, or None for the default "Set2" palette.
        legends_and_labels_shown: If False, hides ticks, labels, legend and spines.
        y_min: Lower y-axis limit.
        y_max: Upper y-axis limit.
        legend_loc: Matplotlib legend location.
    """

    def __init__(
        self,
        metrics_list: List[str],
        proportion_metric: list,
        legend_labels: Optional[List[str]],
        bins: list,
        x_label: Optional[str],
        y_label: Optional[str],
        colors: Union[str, list, None],
        legends_and_labels_shown: bool,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        legend_loc: str = "upper right",
    ) -> None:
        self.proportion_metric = list(proportion_metric)
        self.legend_labels = legend_labels
        self.bins = bins
        self.x_label = x_label
        self.y_label = y_label
        self.colors = colors
        self.legends_and_labels_shown = legends_and_labels_shown
        self.y_min = y_min
        self.y_max = y_max
        self.legend_loc = legend_loc

    def __call__(self, metrics_global: Dict[str, Any], ax: Axes) -> Dict[str, Any]:
        """Draws the per-bin MAE curves (and optional proportion overlay) on the given axes.

        Args:
            metrics_global: Dictionary of global metrics, must contain every
                name in `metrics_list` (and `proportion_metric` if set).
            ax: Matplotlib axes to draw on.

        Returns:
            Dictionary mapping `{metric_name}_{bin}` to the computed MAE value.

        Raises:
            Exception: If a metric in `metrics_list` or `proportion_metric` is
                missing from `metrics_global`.
        """
        for metric_name in self.metrics_list:
            if metric_name not in metrics_global:
                raise Exception(f"Metric '{metric_name}' not found.")
        if self.proportion_metric is not None:
            for pm in self.proportion_metric:
                if pm not in metrics_global:
                    raise Exception(f"Proportion metric '{pm}' not found.")

        positions = np.arange(len(self.bins))
        num_metrics = len(self.metrics_list)

        if self.colors is None:
            palette = sns.color_palette("Set2", num_metrics)
        elif isinstance(self.colors, str):
            palette = sns.color_palette(self.colors, num_metrics)
        else:
            palette = self.colors

        graph_metrics = {}
        for i, metric_name in enumerate(self.metrics_list):
            data_per_bin = metrics_global[metric_name]
            mae_values = []
            for i, data in enumerate(data_per_bin):
                if len(data) > 0:
                    mae_values.append(np.mean(np.abs(data)))
                    graph_metrics[f"{metric_name}_{self.bins[i]}"] = np.mean(np.abs(data))
                else:
                    mae_values.append(np.nan)
                    graph_metrics[f"{metric_name}_{self.bins[i]}"] = np.nan

            color = palette[i % len(palette)]
            label = (
                self.legend_labels[i]
                if self.legend_labels and i < len(self.legend_labels)
                else None
            )

            ax.plot(
                positions,
                mae_values,
                marker="o",
                linestyle="-",
                color=color,
                linewidth=2,
                markersize=6,
                label=label,
            )

        ax.set_xticks(positions)

        if self.legends_and_labels_shown:
            ax.set_xticklabels(self.bins, rotation=40, ha="right")
        else:
            ax.set_xticklabels([])

        if self.x_label is not None and self.legends_and_labels_shown:
            ax.set_xlabel(self.x_label, fontsize=13)

        if self.y_label is not None and self.legends_and_labels_shown:
            ax.set_ylabel(self.y_label, fontsize=13)

        if not self.legends_and_labels_shown:
            ax.set_yticklabels([])

        if self.y_min is not None and self.y_max is not None:
            ax.set_ylim(self.y_min, self.y_max)

        ax.tick_params(axis="y", direction="in", length=6)
        ax.tick_params(axis="x", direction="in", length=6)

        ax.axhline(0, color="black", linewidth=1.0, linestyle="-", zorder=0)
        ax.grid(color="#C4C4C4", linestyle="--", axis="y", linewidth=0.7, zorder=1)
        ax.set_facecolor("white")

        if not self.legends_and_labels_shown:
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(False)

        if self.legend_labels and self.legends_and_labels_shown:
            legend = ax.legend(loc=self.legend_loc, frameon=False)
            legend.set_zorder(4)

        if self.proportion_metric is not None:
            ax2 = ax.twinx()

            all_proportions = []
            for pm in self.proportion_metric:
                proportion_data = metrics_global[pm]
                proportions = [len(data) for data in proportion_data]
                all_proportions.append(proportions)

            total_values = sum(sum(props) for props in all_proportions)

            num_prop_metrics = len(self.proportion_metric)
            bar_width = 0.75 / max(num_prop_metrics, 1)

            max_prop_percentage = 0

            for j, pm in enumerate(self.proportion_metric):
                proportions = all_proportions[j]
                proportions_percentage = [
                    prop / total_values * 100 if total_values > 0 else 0 for prop in proportions
                ]

                if proportions_percentage:
                    max_prop_percentage = max(max_prop_percentage, max(proportions_percentage))

                # Offset for multibar
                offset = (j - (num_prop_metrics - 1) / 2) * bar_width
                current_positions = positions + offset

                # Use the same color as the curve if the metric is in metrics_list
                if pm in self.metrics_list:
                    color_idx = self.metrics_list.index(pm)
                else:
                    color_idx = j

                color = palette[color_idx % len(palette)]

                ax2.bar(
                    current_positions,
                    proportions_percentage,
                    alpha=0.3,
                    color=color,
                    width=bar_width * 0.95,
                    edgecolor="none",
                )

            if self.legends_and_labels_shown:
                ax2.set_ylabel("Proportion (%)", fontsize=13)

            ax2.tick_params(axis="y", direction="in", length=6)
            if not self.legends_and_labels_shown:
                ax2.set_yticklabels([])

            if not self.legends_and_labels_shown:
                for spine_pos in ["top", "right", "left", "bottom"]:
                    ax2.spines[spine_pos].set_visible(False)

            if max_prop_percentage > 0:
                ax2.set_ylim(0, max_prop_percentage * 1.25)
            else:
                ax2.set_ylim(0, 1)
            ax2.grid(False)

            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)

        return graph_metrics
