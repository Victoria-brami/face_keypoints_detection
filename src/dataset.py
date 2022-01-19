import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class FaceDataset(Dataset):

    def __init__(self, transform=None, mode='train', datapath='../data'):
        self.data_df = pd.read_csv(os.path.join(datapath, 'training.csv'))
        self.mode = mode
        self.transform = transform
        self.w = 96
        self.h = 96
        self.c = 1
        self.split_dataset()


    def split_dataset(self):
        nb_items = self.data_df.shape[0]
        if self.mode == 'train' or self.mode =='eval':
            nb_items = int(0.92 * self.data_df.shape[0])
            self.data = self.data_df.head(nb_items)
            if self.mode =='train':
                nb_items = int(0.9 * nb_items)
                self.data = self.data.head(nb_items)
            else:
                nb_items -= int(0.9 * nb_items)
                self.data = self.data.tail(nb_items)
        elif self.mode == 'test':
            nb_items -= int(0.9 * self.data_df.shape[0])
            self.data = self.data_df.tail(nb_items)
        del self.data_df

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        target_cols = list(self.data.drop('Image', axis=1).columns)
        raw_image = np.array(self.data['Image'][item].str.split().tolist(), dtype='float')
        image = raw_image.reshape(-1, self.h, self.w, self.c)
        label = self.data[target_cols].values[item]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def show_keypoints(self, sample):
        image, landmark = sample['image'], sample['label']
        plt.imshow(image)
        for i in range(len(landmark) // 2):
            plt.scatter(landmark[2*i], landmark[2*i+1], s=12, c='red')
        plt.show()


