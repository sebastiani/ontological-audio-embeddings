import os
import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class AudioSetDataset(Dataset):
    """Google AudioSet  in Raw in Form"""

    def __init__(self, csvFile,  labelDict):
        """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
        """
        self.dataFrame = pd.read_csv(csvFile)
        with open(labelDict, 'rb') as f:
            self.label_dict, self.num_classes = pickle.load(f)

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        audioName = self.dataFrame[idx, 0]
        label = self.dataFrame[idx, 1]

        label = self.label_dict[label]
        y = np.zeros((1, self.num_classes)).astype('float')
        y[label] = 1.0
        audio = librosa.load(audioName)

        sample = {'audio': audio, 'label': y}
        return sample

