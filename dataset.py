import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MNISTPlusDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, :-1].values.astype('float32').reshape((28, 28))
        label = self.data.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
