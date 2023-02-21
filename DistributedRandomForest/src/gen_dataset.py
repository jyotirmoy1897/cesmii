import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class GenData(Dataset):
    def __init__(self, data_x, data_y, transform=None, target_transform=None):
        #self.input_data = pd.read_excel(data_path)
        self.labels = data_x
        self.features = data_y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return feature, label

'''
def test():
    data_path = "A:\Research Projects\Distributed_RF\dataset\grinding data.xlsx"
    CESMII_dataset = CESMII_Machine_Dataset(data_path)
    train_size = int(len(CESMII_dataset) * 0.75)
    test_size = int(len(CESMII_dataset) * 0.25)
    train_CESMII_dataset, test_CESMII_dataset = random_split(CESMII_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_CESMII_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_CESMII_dataset, batch_size=4, shuffle=True)


    for i in train_dataloader:
        print(i[0].numpy())
        break
'''

if __name__ == '__main__':
    test()