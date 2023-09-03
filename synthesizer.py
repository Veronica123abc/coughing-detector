import numpy as np
import torch
import pandas as pd
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import librosa
import seaborn as sns
import librosa.display
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


# Merges a list of files according to requested weights
def merge(audiofiles, weights, align='start'):

    waveforms = []
    weights = [float(w) / sum(weights) for w in weights] # Normalize the weights
    for file, weight in zip(audiofiles, weights):
        w,s = librosa.load(file)
        waveforms.append({'waveform': w, 'sample_rate': s, 'weight': weight})
    min_sample_rate = min([waveform['sample_rate' ] for waveform in waveforms])
    max_length = max([len(waveform['waveform']) for waveform in waveforms])
    merged = np.zeros(max_length, dtype=np.float32)
    for waveform in waveforms:
        wf = librosa.resample(waveform['waveform'], orig_sr=waveform['sample_rate'], target_sr=min_sample_rate)
        merged += waveform['weight'] * np.pad(wf, (0,max_length - len(wf)))
    return merged, min_sample_rate

def add_background_to_event(event, background_file, weight = 0.5, target_sr=16000):
    background, s = librosa.load(background_file)
    background = librosa.resample(background, orig_sr=s, target_sr=target_sr) if s != target_sr else background # Resample background if different
    if len(background) < len(event):
        background = np.pad(background, (0, len(event)), mode='reflect')
    start_background = np.random.randint(len(background) - len(event))
    merged = weight * background[start_background: start_background + len(event)] + (1 - weight) * event
    return merged

def extract_labeled_events(labeled_dir, waveform_file='data.wav', label_file='label.label'): # waveform_file, label_file):
    waveform, sr = librosa.load(os.path.join(labeled_dir, waveform_file))
    df = pd.read_csv(os.path.join(labeled_dir, label_file))
    events = []
    for index, row in df.iterrows():
        event = waveform[int(sr * row['Time(Seconds)']): int(sr * row['Time(Seconds)'] + sr * row['Length(Seconds)'])]#
        events.append(event)
    return events




