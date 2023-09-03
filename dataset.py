import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2

import featuring
import librosa


class CoughingPrecomputedMelspecs(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        samples = pd.read_csv(annotations_file, sep=' ', header=None)
        self.specgram_files = list(samples[0])
        self.labels=list(samples[1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        melspec = torch.load(self.specgram_files[idx])
        img = melspec.numpy()
        img = np.moveaxis(img, 0, -1)
        img=cv2.resize(img, (32,32), interpolation=cv2.INTER_CUBIC)
        img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.moveaxis(img, -1,0)
        melspec=torch.tensor(img)
        label = self.labels[idx]
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=2).type(torch.float)
        if self.transform:
            melspec = self.transform(melspec)
        if self.target_transform:
            label = self.target_transform(label)
        return melspec, label

def collate_fn(batch):

    in_data = []
    targets = []

    # Gather in lists, and encode labels as indices
    for data_sample, target in batch:
        in_data += [data_sample]
        targets += [torch.tensor(target)]

    # Group the list of tensors into a batched tensor
    tensors = torch.stack(in_data)
    targets = torch.stack(targets)

    return tensors, targets

class Coughing(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.samples = pd.read_csv(annotations_file, sep='\t', header=None)


    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        waveform, sr = librosa.load(self.samples.iloc[idx][0])
        patch = waveform[self.samples.iloc[idx][1] : self.samples.iloc[idx][2]]
        melspec = featuring.mel_spectrogram(patch)
        img = melspec.numpy()
        img = np.moveaxis(img, 0, -1)
        img=cv2.resize(img, (32,32), interpolation=cv2.INTER_CUBIC)
        img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.moveaxis(img, -1,0)
        melspec=torch.tensor(img)
        label = self.samples.iloc[idx][3]
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=2).type(torch.float)
        if self.transform:
            melspec = self.transform(melspec)
        if self.target_transform:
            label = self.target_transform(label)
        return melspec, label

def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    in_data = []
    targets = []

    # Gather in lists, and encode labels as indices
    for data_sample, target in batch:
        in_data += [data_sample]
        targets += [torch.tensor(target)]

    # Group the list of tensors into a batched tensor
    tensors = torch.stack(in_data)
    targets = torch.stack(targets)

    return tensors, targets

