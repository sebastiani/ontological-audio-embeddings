import os
import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset
from scipy.stats import beta as Beta

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class AudioSetDataset(Dataset):
    """Google AudioSet  in Raw in Form"""

    def __init__(self, csvFile,  labelDict, alpha=2, beta=5):
        """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
        """
        self.dataFrame = pd.read_csv(csvFile)
        with open(labelDict, 'rb') as f:
            self.label_dict, self.num_classes = pickle.load(f)

        self.alpha = alpha
        self.beta = beta

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        audioName = self.dataFrame.iloc[[idx]]['Path'].values[0]
        #label = self.dataFrame.iloc[[idx]]

        #label = self.label_dict[label]
        #y = np.zeros((1, self.num_classes)).astype('float')
        #y[label] = 1.0
        audio = np.expand_dims(np.load(audioName), axis=0)

        snr = Beta.rvs(self.alpha, self.beta, size=1)
        var = np.var(audio, axis=1)[0]
        snr = 0.5 if snr >= 0.5 else snr
        noise_var = snr * var
        noisy_audio = audio + np.random.normal(0.0, noise_var, audio.shape)
        sample = {'clean_audio': audio, 'noisy_audio': noisy_audio}
        return sample

