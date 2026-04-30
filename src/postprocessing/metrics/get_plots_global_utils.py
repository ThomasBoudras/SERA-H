from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
from scipy.interpolate import interpn
from scipy.stats import linregress
import logging

log = logging.getLogger(__name__)

class get_plots_global :
    def __init__(self, save_dir, plot_set, model_name) :
        self.save_dir = Path(save_dir).resolve()        
        self.plot_set = plot_set
        self.model_name = model_name

    def __call__(self, metrics_global):
        for plot_name_base, plot in self.plot_set.items():
            fig = plt.figure(figsize=(plot.size_plot_width, plot.size_plot_height))
            plot.create_plot(metrics_global)
            plot_path = self.save_dir / f"{plot_name_base}_{self.model_name}_plot.pdf"
            fig.savefig(
                plot_path, bbox_inches="tight", pad_inches=0
            )
            plt.close()

class plot_model :
    def __init__(self, graph_list, nb_row, nb_col, size_plot_width, size_plot_height, hspace=0.3, wspace=0.3):
        self.graph_list = graph_list
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.size_plot_width = size_plot_width
        self.size_plot_height = size_plot_height
        self.hspace = hspace  # Horizontal space between subplots
        self.wspace = wspace  # Width space between subplots
    
    def create_plot(self, metrics_global):
        for graph in self.graph_list :
            ax = plt.subplot2grid((self.nb_row, self.nb_col), (graph.idx_row, graph.idx_col), rowspan=graph.rowspan, colspan=graph.colspan)
            graph.create_graph(metrics_global, ax)
        
        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=self.hspace, wspace=self.wspace)


        
class graph_model :
    def __init__(self, idx_row, idx_col, graph_title, method_graph, rowspan=1, colspan=1, bold_title=False):
        self.idx_row = idx_row
        self.idx_col = idx_col
        self.graph_title = graph_title
        self.method_graph = method_graph
        self.rowspan = rowspan
        self.colspan = colspan
        self.bold_title = bold_title  # Option to set the graph title in bold, defaults to False

    def create_graph(self, metrics_global, ax):
        self.method_graph(metrics_global, ax)
        if self.graph_title:
            graph_title = self.graph_title
            if self.bold_title:
                ax.set_title(graph_title, fontweight='bold')
            else:
                ax.set_title(graph_title)


class method_bar :
    def __init__(self, metrics_list, y_min, y_max):
        self.metrics_list = metrics_list
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, metrics_global, ax) :
        for key in self.metrics_list :
            if key not in metrics_global :
                Exception(f"You must first compute metric {key}")
        ax.bar(self.metrics_list, [metrics_global[key] for key in self.metrics_list])
        ax.set_ylim(self.y_min, self.y_max)
        ax.grid(color="gray", linestyle="dashed", axis="y")
        plt.xticks(rotation=40)

class method_curve_by_bins :
    def __init__(
        self, 
        metric_name, 
        bins, 
        x_label, 
        y_label,
        colors, 
        legend_bins, 
        y_min=None, 
        y_max=None,
        log_scale=False
    ):
        self.metric_name = metric_name
        self.bins = [None] + bins + [None]
        self.x_label = x_label
        self.y_label = y_label
        self.colors = colors
        self.legend_bins = legend_bins
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, metrics_global, ax) :
        
        metric_values = []
        
        for bin_idx in range(len(self.bins) - 1):
            min_area_current = self.bins[bin_idx]
            max_area_current = self.bins[bin_idx + 1]
            max_area_current_str = "" if max_area_current is None else str(int(max_area_current))
            min_area_current_str = "" if min_area_current is None else str(int(min_area_current))
            name_bin = f"{min_area_current_str}-{max_area_current_str}"
            
            metric_value = metrics_global[f"{self.metric_name}_{name_bin}"]
            metric_values.append(metric_value)

        # Plot a single curve connecting all the bins
        x_positions = range(len(metric_values))
        ax.plot(x_positions, metric_values, marker='o', linestyle='-', color=self.colors[0] if isinstance(self.colors, list) else self.colors)
        
        # Set the x-axis ticks to show the bin labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(self.legend_bins, rotation=45, ha='right')

        if self.x_label is not None:
            ax.set_xlabel(self.x_label, fontsize=13)
        
        if self.y_label is not None:
            ax.set_ylabel(self.y_label, fontsize=13)

        if self.y_min is not None and self.y_max is not None:
            ax.set_ylim(self.y_min, self.y_max)
            
        ax.grid(color="#C4C4C4", linestyle="--", axis="y", linewidth=0.7, zorder=1)


class method_table :
    def __init__(self, metrics_list, font_size, table_width_scale, table_height_scale):
        self.metrics_list = metrics_list
        self.font_size = font_size
        self.table_width_scale = 0.7
        self.table_height_scale = 1.8

    def __call__(self, metrics_global, ax) :
        for key in self.metrics_list :
            if key not in metrics_global :
                Exception(f"You must first compute metric {key}")
        
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
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='center')
        table.set_fontsize(self.font_size)
        table.scale(self.table_width_scale, self.table_height_scale)
        
        ax.axis('off')


class method_scatter_density:
    def __init__(
        self, 
        metric_name, 
        x_label, 
        y_label, 
        min_range,
        max_range,
        bins, 
        metrics_plot,
        max_points_on_scatter,
        point_size,
        x_y_line,
        fit_line,
        show_legend,
        add_noise,
        min_normalization,
        max_normalization
    ):
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

    def __call__(self, metrics_global, ax):

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
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=self.bins, range=[self.range_fig, self.range_fig], density=True)
        z = interpn((x_edges[:-1], y_edges[:-1]), hist, np.vstack([x, y]).T, method="splinef2d", bounds_error=False, fill_value=0)
        
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
        ax.set_aspect('equal', adjustable='box')

        # Create the scatter plot with density coloring
        if self.min_normalization is None :
            min_normalization = np.quantile(z, 0.01) + 1e-5

            log.info(f"Min normalization for scatter plot: {min_normalization}")
        else:
            min_normalization = self.min_normalization
        if self.max_normalization is None :
            max_normalization = np.quantile(z, 0.75)
            log.info(f"Max normalization for scatter plot: {max_normalization}")
        else:
            max_normalization = self.max_normalization
       
        norm = colors.LogNorm(vmin=min_normalization, vmax=max_normalization, clip=True)
   
        ax.scatter(x, y, c=z, s=self.point_size, cmap="inferno", norm=norm, edgecolor="none", rasterized=True)

        # Add a 1:1 line for reference if specified.
        if self.x_y_line:
            ax.plot(self.range_fig, self.range_fig, 'k--', label='1:1')

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
        if "r2" in self.metrics_plot:
            r2 = r2_score(x_init, y_init)
            metrics_text.append(f"$R^2$ = {r2:.3f}")
        if "mae" in self.metrics_plot:
            mae = mean_absolute_error(x_init, y_init)
            metrics_text.append(f"MAE = {mae:.3f}")
        if "rmse" in self.metrics_plot:
            rmse = np.sqrt(mean_squared_error(x_init, y_init))
            metrics_text.append(f"RMSE = {rmse:.3f}")
        if "me" in self.metrics_plot:
            me = np.mean(y_init - x_init) # Prediction - Target
            metrics_text.append(f"ME = {me:.3f}")
        
        # Display the metrics in a text box.
        if metrics_text:
            ax.text(
                self.range_fig[1] - 0.04 * (self.range_fig[1] - self.range_fig[0]),
                self.range_fig[0] + 0.1 * (self.range_fig[1] - self.range_fig[0]),
                "\n".join(metrics_text), 
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='square,pad=0.5')
            )

class method_boxplot_by_bins:
    def __init__(self, metrics_list, proportion_metric, legend_labels, bins, x_label, y_label, colors, legends_and_labels_shown, y_min=None, y_max=None, max_points_per_bin=None):
        # Store input parameters for later use in plotting
        self.metrics_list = metrics_list
        self.proportion_metric = proportion_metric
        self.legend_labels = legend_labels
        self.bins = bins
        self.x_label = x_label
        self.y_label = y_label
        self.colors = colors   
        self.y_min = y_min
        self.y_max = y_max
        self.legends_and_labels_shown = legends_and_labels_shown
        self.max_points_per_bin = max_points_per_bin


    def __call__(self, metrics_global, ax):
        # Ensure all provided metric names are present in the global metrics dictionary
        for metric_name in self.metrics_list:
            if metric_name not in metrics_global:
                raise Exception(f"Metric '{metric_name}' not found.")
        # Check that the proportion metric exists if it is specified
        if self.proportion_metric is not None and self.proportion_metric not in metrics_global:
            raise Exception(f"Proportion metric '{self.proportion_metric}' not found.")

        num_metrics = len(self.metrics_list)
        positions = np.arange(len(self.bins))  # x positions for boxplots
        box_width = 0.8 / max(num_metrics, 1)  # width of each boxplot depending on number of metrics

        # Select default color palette if colors is None, otherwise use provided palette
        if self.colors is None:
            palette = sns.color_palette("Set2", num_metrics)
        elif isinstance(self.colors, str):
            palette = sns.color_palette(self.colors, num_metrics)
        else:
            palette = self.colors

        legend_handles = []
        for i, metric_name in enumerate(self.metrics_list):
            data_to_plot_full = metrics_global[metric_name]

            if self.max_points_per_bin is not None:
                data_to_plot = []
                for bin_data in data_to_plot_full:
                    if len(bin_data) > self.max_points_per_bin:
                        # Using np.random.choice for efficient sampling
                        sampled_data = np.random.choice(np.array(bin_data), self.max_points_per_bin, replace=False)
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
            if self.legend_labels is not None and self.legends_and_labels_shown and i < len(self.legend_labels):
                legend_handles.append(
                    mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.7, label=self.legend_labels[i])
                )

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

        ax.tick_params(axis='y', direction='in', length=6)
        ax.tick_params(axis='x', direction='in', length=6)

        # Draw a horizontal line at y=0 for reference
        ax.axhline(0, color="black", linewidth=1.0, linestyle="-", zorder=0)
        # Add grid lines for better readability
        ax.grid(color="#C4C4C4", linestyle="--", axis="y", linewidth=0.7, zorder=1)
        ax.set_facecolor("white")

        # Hide unnecessary plot spines for a cleaner look
        if not self.legends_and_labels_shown :
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(False)

        # Add the legend to the plot if any legend handles were created
        if legend_handles and self.legends_and_labels_shown:
            legend = ax.legend(handles=legend_handles, loc="lower left", frameon=False)
            legend.set_zorder(4)

        # If a proportion metric is specified, display a bar overlay for counts/proportions
        if self.proportion_metric is not None:
            ax2 = ax.twinx()
 
             # Compute the number of values in each bin
            proportion_data = metrics_global[self.proportion_metric]
            proportions = [len(data) for data in proportion_data]

            # Compute the percentage that each bin represents
            total_values = sum(proportions)
            proportions_percentage = [prop / total_values * 100 if total_values > 0 else 0 for prop in proportions]

            # Overlay a bar plot showing the proportions for each bin
            ax2.bar(
                positions,
                proportions_percentage,
                alpha=0.3,
                color="#BFBFBF",
                width=0.75,
                edgecolor="none",
            )

            if self.legends_and_labels_shown:
                ax2.set_ylabel('Proportion (%)', fontsize=13)

            ax2.tick_params(axis='y', direction='in', length=6)
            if not self.legends_and_labels_shown:
                ax2.set_yticklabels([])
            
            # Hide all spines for the secondary axis as well
            if not self.legends_and_labels_shown :
                for spine_pos in ['top', 'right', 'left', 'bottom']:
                    ax2.spines[spine_pos].set_visible(False)

            # Set y-axis range for the secondary axis based on the data
            if proportions_percentage and max(proportions_percentage) > 0:
                ax2.set_ylim(0, max(proportions_percentage) * 1.25)
            else:
                ax2.set_ylim(0, 1)
            # Avoid grid on the overlay axis for clarity
            ax2.grid(False)

             # Ensure ax is drawn on top of ax2 and has a transparent background
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)


class method_mae_curve_by_bins:
    def __init__(self, metrics_list, proportion_metric, legend_labels, bins, x_label, y_label, colors, legends_and_labels_shown, y_min=None, y_max=None):
        self.metrics_list = metrics_list
        self.proportion_metric = proportion_metric
        self.legend_labels = legend_labels
        self.bins = bins
        self.x_label = x_label
        self.y_label = y_label
        self.colors = colors
        self.legends_and_labels_shown = legends_and_labels_shown
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, metrics_global, ax):
        for metric_name in self.metrics_list:
            if metric_name not in metrics_global:
                raise Exception(f"Metric '{metric_name}' not found.")
        if self.proportion_metric is not None and self.proportion_metric not in metrics_global:
            raise Exception(f"Proportion metric '{self.proportion_metric}' not found.")

        positions = np.arange(len(self.bins))
        num_metrics = len(self.metrics_list)
        
        if self.colors is None:
            palette = sns.color_palette("Set2", num_metrics)
        elif isinstance(self.colors, str):
            palette = sns.color_palette(self.colors, num_metrics)
        else:
            palette = self.colors

        for i, metric_name in enumerate(self.metrics_list):
            data_per_bin = metrics_global[metric_name]
            mae_values = []
            for data in data_per_bin:
                if len(data) > 0:
                    mae_values.append(np.mean(np.abs(data)))
                else:
                    mae_values.append(np.nan)
            
            color = palette[i % len(palette)]
            label = self.legend_labels[i] if self.legend_labels and i < len(self.legend_labels) else None
            
            ax.plot(positions, mae_values, marker='o', linestyle='-', color=color, linewidth=2, markersize=6, label=label)

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

        ax.tick_params(axis='y', direction='in', length=6)
        ax.tick_params(axis='x', direction='in', length=6)

        ax.axhline(0, color="black", linewidth=1.0, linestyle="-", zorder=0)
        ax.grid(color="#C4C4C4", linestyle="--", axis="y", linewidth=0.7, zorder=1)
        ax.set_facecolor("white")

        if not self.legends_and_labels_shown:
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(False)

        if self.legend_labels and self.legends_and_labels_shown:
            legend = ax.legend(loc="upper left", frameon=False)
            legend.set_zorder(4)

        if self.proportion_metric is not None:
            ax2 = ax.twinx()
            proportion_data = metrics_global[self.proportion_metric]
            proportions = [len(data) for data in proportion_data]
            total_values = sum(proportions)
            proportions_percentage = [prop / total_values * 100 if total_values > 0 else 0 for prop in proportions]

            ax2.bar(
                positions,
                proportions_percentage,
                alpha=0.3,
                color="#BFBFBF",
                width=0.75,
                edgecolor="none",
            )

            if self.legends_and_labels_shown:
                ax2.set_ylabel('Proportion (%)', fontsize=13)

            ax2.tick_params(axis='y', direction='in', length=6)
            if not self.legends_and_labels_shown:
                ax2.set_yticklabels([])
            
            if not self.legends_and_labels_shown:
                for spine_pos in ['top', 'right', 'left', 'bottom']:
                    ax2.spines[spine_pos].set_visible(False)

            if proportions_percentage and max(proportions_percentage) > 0:
                ax2.set_ylim(0, max(proportions_percentage) * 1.25)
            else:
                ax2.set_ylim(0, 1)
            ax2.grid(False)

            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
