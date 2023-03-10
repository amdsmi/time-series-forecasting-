import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class WaterDataSet(Dataset):
    def __init__(self, root_dir, input_len, output_len, data_col, target_col, duration=None):
        super(WaterDataSet, self).__init__()
        '''
        root_dir = dataset directory 
        input_len = input sequence length
        output_len = output sequence length
        data_col = used column for training 
        target_col = column that should be predict  
        duration = hourly daily monthly ['h', 'D', 'M']
        '''
        self.root_dir = root_dir
        self.input_len = input_len
        self.output_len = output_len
        self.data_col = data_col
        self.target_col = target_col
        self.duration = duration
        self.data = self._data_reader

    @property
    def _data_reader(self):
        data = pd.read_csv(self.root_dir)

        if self.duration is not None:
            data['Date'] = pd.to_datetime(data['Date'])
            new_data = data.groupby(pd.PeriodIndex(data['Date'], freq=self.duration)).sum(numeric_only=True).reset_index()
        else:
            data['Date'] = pd.to_datetime(data['Date'])
            new_data = data.reset_index()

        return new_data

    def __len__(self):
        return len(self.data) - self.input_len - self.output_len

    def __getitem__(self, item):
        in_out = self.data[item: item + self.input_len + self.output_len]
        data = in_out[:self.input_len][self.data_col]
        label = in_out[self.input_len:][self.target_col]
        data = torch.from_numpy(data.to_numpy(dtype='float32'))
        label = torch.from_numpy(label.to_numpy(dtype='float32'))

        return data, label


if __name__ == '__main__':

    dataset = WaterDataSet(root_dir='test.csv', input_len=10, output_len=10,
                           data_col=['CHLOROPHYL', 'EC', 'DO', 'ORP', 'PH'],
                           target_col=['Water_Temp'])
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    for data, label in data_loader:
        print(data.shape, label.shape)