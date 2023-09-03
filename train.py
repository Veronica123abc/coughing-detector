import numpy as np
import torch
import pandas as pd
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
#import librosa
import seaborn as sns
#import librosa.display
import skimage.io
import os
from scipy import signal
from scipy.io import wavfile
from playsound import playsound
from torchaudio.models import wavernn
from torchaudio.models import WaveRNN
from torchaudio.transforms import MelSpectrogram
import torchaudio
import torchaudio
from IPython.display import Audio, display
from torchaudio.datasets import LJSPEECH
import sounddevice as sd
import featuring
import synthesizer
import librosa
import utils
import dataset
import torchvision
import torchvision.transforms as transforms
device = "cuda"
batch_size = 32
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models.simple_cnn

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_set = dataset.CoughingPrecomputedMelspecs("train.txt")
test_set = dataset.CoughingPrecomputedMelspecs("test.txt")

#train_set = dataset.Coughing("kallekula2.txt")
#test_set = dataset.Coughing("kallekula2.txt")
# for i in train_set:
#     print(i[1])

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    #collate_fn=dataset.collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    #collate_fn=dataset.collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

def train(model=None, epoch=10):
    device = torch.device("cuda")
    net=models.simple_cnn.Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()



            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:  # print every 2000 mini-batches
                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        with torch.no_grad():
            correct_pred = 0
            total_pred = 0
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                _, labels = torch.max(labels, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred += 1
                    total_pred += 1
            print(f"Epoch {epoch} {correct_pred} of {total_pred} ({int(100 * float(correct_pred)/float(total_pred))})")
    torch.save(net.state_dict(), 'weights.pt')
