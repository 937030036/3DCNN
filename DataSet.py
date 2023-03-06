import os
import random

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class NiiDataSet(Dataset):
    def __init__(self, datapath="./data/", transform=None):
        self.transform = transform
        filelist = os.listdir(datapath)
        self.imglist = []
        self.labellist = []
        for tmp in filelist:
            if 'image' in tmp:
                self.imglist.append(datapath + tmp)
            elif 'label' in tmp:
                self.labellist.append(datapath + tmp)
                pass
            pass
        self._shuffle()
        pass

    def _shuffle(self):
        ans = list(zip(self.imglist, self.labellist))
        random.shuffle(ans)
        self.imglist, self.labellist = zip(*ans)
        pass

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        assert len(self.imglist) == len(self.labellist)

        img = sitk.ReadImage(self.imglist[idx])
        img = sitk.GetArrayFromImage(img)
        label = sitk.ReadImage(self.labellist[idx])
        label = sitk.GetArrayFromImage(label)

        if self.transform:
            # TODO
            pass

        return torch.Tensor(img.astype(float)), torch.tensor(label.astype(float))
