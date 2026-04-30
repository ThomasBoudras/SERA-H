from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class get_plots_local :
    def __init__(self, save_dir, plot_set, nb_plots, model_name) :
        self.save_dir = Path(save_dir).resolve()
        self.plot_set = plot_set
        self.nb_plots = nb_plots
        self.model_name = model_name

    def __call__(self, images, metrics, plot_name_sample, row):
        for plot_name_base, plot in self.plot_set.items():
            fig = plt.figure(figsize=(plot.size_plot_width, plot.size_plot_height))
            plot.create_plot(images, metrics, row)
            plot_path = self.save_dir / f"{plot_name_base}_{plot_name_sample}_{self.model_name}_plot.jpg"
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
    
    def create_plot(self, images, metrics, row):
        for graph in self.graph_list :
            ax = plt.subplot2grid((self.nb_row, self.nb_col), (graph.idx_row, graph.idx_col), rowspan=graph.rowspan, colspan=graph.colspan)
            graph.create_graph(images, metrics, ax, row)
        
        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=self.hspace, wspace=self.wspace)

        
class graph_model :
    def __init__(self, idx_row, idx_col, graph_title, method_graph, grouping_dates, rowspan= 1, colspan = 1):
        self.idx_row = idx_row
        self.idx_col = idx_col
        self.graph_title = graph_title
        self.method_graph = method_graph
        self.rowspan = rowspan
        self.colspan= colspan
        self.grouping_dates = grouping_dates

    def create_graph(self, images, metrics, ax, row):
        self.method_graph(images, metrics, ax, row)
        
        date = row[self.grouping_dates] if self.grouping_dates else None
        
        if self.graph_title :
            graph_title = self.graph_title
            if "<date>" in graph_title :
                graph_title = graph_title.replace("<date>", date)
            if "<year>" in graph_title :
                graph_title = graph_title.replace("<year>", date[:4])
            
            ax.set_title(graph_title, fontsize=16)


class method_imshow :
    def __init__(self, image_name, real_patch_size, resolution, cmap=None, norm=None):
        self.image_name = image_name
        self.real_patch_size = real_patch_size
        self.resolution = resolution
        if self.real_patch_size is not None and self.resolution is not None :
            self.patch_size = self.real_patch_size / self.resolution 
        else :
            self.patch_size = None
        self.cmap = cmap
        self.norm = norm

    def __call__(self, images, metrics, ax, row):
        if self.image_name not in images :
            Exception(f"You must first load {self.image_name}")
        image = images[self.image_name]
        
        if self.real_patch_size is not None :
            # We crop the image to the patch size percentage
            height, width = image.shape[-2], image.shape[-1]
            if self.patch_size is not None :
                x_start = int(height/2 - self.patch_size/2) # we center the crop
                x_stop = int(x_start + self.patch_size)
                y_start = int(width/2 - self.patch_size/2) # we center the crop
                y_stop = int(y_start + self.patch_size)
                
                image = image[..., x_start:x_stop, y_start:y_stop ]
            
            if len(image.shape) == 3 :
                image = np.transpose(image, (1, 2, 0))
                gain = 0.4/image.mean()
                image = np.clip(image*gain, 0, 1)

            # Check if the image has only one unique value and modify if needed
            if "mask" in self.image_name :
                first_value = image[..., 0, 0]
                if not np.any(image != first_value):
                    image[..., 0, 0] = 0
        ax.imshow(image, cmap=self.cmap, norm=self.norm)
        ax.axis("off")
        


class method_bar :
    def __init__(self, metrics_list, y_min, y_max):
        self.metrics_list = metrics_list
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, images, metrics, ax, row) :
        for key in self.metrics_list :
            if key not in metrics :
                Exception(f"You must first compute metric {key}")
        ax.bar(self.metrics_list, [metrics[key] for key in self.metrics_list])
        ax.set_ylim(self.y_min, self.y_max)
        ax.grid(color="gray", linestyle="dashed", axis="y")
        plt.xticks(rotation=40)


class method_table :
    def __init__(self, metrics_list, font_size, table_width_scale, table_height_scale):
        self.metrics_list = metrics_list
        self.font_size = font_size
        self.table_width_scale = 0.7
        self.table_height_scale = 1.8

    def __call__(self, images, metrics, ax, row) :
        for key in self.metrics_list :
            if key not in metrics :
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
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='center')
        table.set_fontsize(self.font_size)
        table.scale(self.table_width_scale, self.table_height_scale)
        
        ax.axis('off')

