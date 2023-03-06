import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DataSet import NiiDataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def showNii(img):
    img = img.cpu()
    for i in range(img.shape[1]):
        plt.imshow(img[:, i, :], cmap='gray')
        plt.show()
        pass


def train(train_Loader):
    t = enumerate(iter(train_Loader))
    for batch_idx, batch in t:
        img_data = batch[0].type(torch.FloatTensor).to(device).squeeze()
        label_data = batch[1].type(torch.FloatTensor).to(device).squeeze()
        showNii(img_data)


if __name__ == '__main__':
    train_dataset = NiiDataSet()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    train(train_loader)
