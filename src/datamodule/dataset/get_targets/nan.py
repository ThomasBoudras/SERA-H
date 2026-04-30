import torch

class getNan :
    def __init__(self, date_column):
        self.date_column = date_column

    def __call__(self, bounds, row, transform) :
        if self.date_column is not None:
            lidar_date = int(row[self.date_column])
            return torch.tensor(torch.nan), {"lidar_acquisition_date": torch.tensor(lidar_date)}
        return torch.tensor(torch.nan), {}