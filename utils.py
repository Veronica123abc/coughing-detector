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
import random
import math
import models


def create_sample_display(specgrams, title=None, ylabel="freq_bin", ax=None, columns=5):
    fig, ax = plt.subplots(math.ceil(len(specgrams) / columns), 5)
    fig.suptitle(title)
    ctr=0
    for ctr in range(len(specgrams)):
        ax[ctr // columns][ctr % columns].imshow(librosa.power_to_db(specgrams[ctr][0]),
                                     origin="lower", aspect="auto", interpolation="nearest")
    for ax in fig.get_axes():
        ax.label_outer()


def plot_samples(annotations='training.txt', num_samples=27, columns = 5, shuffle=True):
    trues = []
    falses = []
    samples = pd.read_csv(annotations, sep=' ', header=None)
    true_files = list(samples.query("@samples[1]==1")[0])
    false_files = list(samples.query("@samples[1]==0")[0])
    if shuffle:
        random.shuffle(true_files)
        random.shuffle(false_files)
    for f in range(num_samples):
        trues += [torch.load(true_files[f])]
        falses += [torch.load(false_files[f])]
    create_sample_display(trues, title="Coughs", ylabel="mel freq")
    create_sample_display(falses, title="Background", ylabel="mel freq")
    plt.show()


def inference(file):
    device = torch.device("cuda")
    label = open('label.txt', 'w')
    net = models.simple_cnn.Net()
    #net = net.to(device)
    net.load_state_dict(torch.load('weights.pt'))

    net.eval()
    waveform, sr = librosa.load(file)
    ctr = 0
    interval = 8000
    while ctr < len(waveform)-interval:
        patch = waveform[ctr:ctr+interval]
        melspect = featuring.mel_spectrogram(patch, sr)
        img = melspect.numpy()
        img = np.moveaxis(img, 0, -1)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.moveaxis(img, -1, 0)
        melspec = torch.tensor(img)
        melspec = torch.unsqueeze(melspec, 0)
        #melspec.to(device)
        prediction = net(melspec)
        m = torch.nn.Softmax(dim=1)
        prediction = m(prediction).detach().numpy()
        cls = np.argmax(prediction)
        print(np.argmax(prediction))
        if cls:
            label.write(f"{float(ctr)/16000}\t{float(ctr + interval)/16000}\tc\n")
        print(ctr)
        ctr += 1000

