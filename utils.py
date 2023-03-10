import pandas as pd


def xls_2_csv(source_path, destination_path):
    data = pd.read_excel(source_path, sheet_name='raw_data')
    data = data[['Date', 'CHLOROPHYL (µg/l)', 'EC (µS/cm)', 'DO (mg/l)', 'ORP (mV)', 'pH ', 'Water Temp (°C)']]
    data.rename(columns={'Date': 'Date', 'CHLOROPHYL (µg/l)': 'CHLOROPHYL',
                         'EC (µS/cm)': 'EC', 'DO (mg/l)': 'DO',
                         'ORP (mV)': 'ORP', 'pH ': 'PH',
                         'Water Temp (°C)': 'Water_Temp'}, inplace=True)
    for i in range(len(data)):
        for j in range(1, len(data.iloc[i, :])):
            if str(data.iloc[i, j]).isdigit():
                pass
            else:
                data.iloc[i, j] = data.iloc[i - 1, j]
    data.to_csv(destination_path, index=False)


def train_test_val(root_dir, percent):
    data = pd.read_csv(root_dir)
    length = len(data)
    train = data[:int(length * percent[0])]
    val = data[int(length * percent[0]): int(length * percent[0] + length * percent[1])]
    test = data[int((length * percent[0]) + (length * percent[1])): int((length * percent[0]) + (length * percent[1]) + (length * percent[2]))]
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    val.to_csv('validation.csv', index=False)


if __name__ == '__main__':
    xls_2_csv('Greek_lake_data.xlsx', 'water_quality.csv')
    train_test_val('water_quality.csv', [0.8, 0.1, 0.1])