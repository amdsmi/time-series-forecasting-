import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class WaterDataSet(Dataset):
    def __init__(self, root_dir, source_len, target_len, output_len, step_ahead, data_col, target_col, duration=None):
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
        self.source_len = source_len
        self.target_len = target_len
        self.output_len = output_len
        self.step_ahead = step_ahead
        self.data_col = data_col
        self.target_col = target_col
        self.duration = duration
        self.data = self._data_reader

    @property
    def _data_reader(self):
        data = pd.read_csv(self.root_dir)

        if self.duration is not None:
            data['Date'] = pd.to_datetime(data['Date'])
            new_data = data.groupby(pd.PeriodIndex(data['Date'], freq=self.duration)).sum(
                numeric_only=True).reset_index()
        else:
            data['Date'] = pd.to_datetime(data['Date'])
            new_data = data.reset_index()

        return new_data

    def __len__(self):
        return len(self.data) - self.source_len - self.step_ahead - self.step_ahead

    def __getitem__(self, item):
        source_target_out = self.data[item: item + self.source_len + self.step_ahead + self.step_ahead]
        source = source_target_out[:self.source_len][self.data_col]
        target = source_target_out[-(self.step_ahead + self.target_len): -self.step_ahead][self.data_col]
        output = source_target_out[-self.output_len:][self.data_col]

        source = torch.from_numpy(source.to_numpy(dtype='float32'))
        target = torch.from_numpy(target.to_numpy(dtype='float32'))
        output = torch.from_numpy(output.to_numpy(dtype='float32'))

        return source, target, output


if __name__ == '__main__':

    dataset = WaterDataSet(root_dir='test.csv',
                           source_len=10,
                           target_len=10,
                           output_len=10,
                           step_ahead=8,
                           data_col=['Water_Temp', 'EC', 'DO', 'ORP', 'PH', 'CHLOROPHYL'],
                           target_col=['Water_Temp', 'EC', 'DO', 'ORP', 'PH', 'CHLOROPHYL'])
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    for source, target, output in data_loader:
        print(source.shape, target.shape, output.shape)
