import numpy as np
import torch
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
from torchaudio.transforms import MelSpectrogram
import parser
import synthesizer
import uuid
import math

labelmap={'cough':0,'background':1}
def mel_spectrogram(event, sample_rate = 16000):
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128

    mel_spectrogram = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )

    event_tensor = torch.tensor(event)
    event_tensor = torch.unsqueeze(event_tensor, dim=0)
    melspec = mel_spectrogram(event_tensor)
    return melspec


def generate_train_test_validate(data_dir, annotation_file, sample_rate = 16000):
    all_annotations = open(annotation_file, 'w')
    labeled_coughings = parser.extract_labeled_files('data/coughing')
    labeled_coughings += parser.extract_labeled_files('data/coughing_batch_2')
    for file in labeled_coughings:
        print('Storing events for ', file)
        df = pd.read_csv(os.path.join(file, 'audacity_label.txt'), sep='\t')
        for index, row in df.iterrows():
            all_annotations.write(f"{os.path.join(file, 'data.wav')}\t {int(row[0] * sample_rate)}\t"
                                  f"{int(row[1] * sample_rate)}\t{labelmap[row[2]]}\n")




def create_mel_specs_for_training(data_dir, train_file = 'train.txt' , test_file = 'test.txt', background_weight=0):
    training = open(train_file, 'w')
    testing = open(test_file, 'w')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    labeled_coughings = parser.extract_labeled_files('data/coughing')
    labeled_coughings += parser.extract_labeled_files('data/coughing_batch_2')
    split = math.floor(len(labeled_coughings) * 0.7)
    train_coughings = labeled_coughings[0:split]
    test_coughings = labeled_coughings[split:-1]

    true_events = []
    backgrounds = parser.extract_background_files(['data/laugh', 'data/people_talking' 'data/mic_tapping'])
    for file in train_coughings:
        events = synthesizer.extract_labeled_events(file)
        for event in events:
            idx = np.random.randint(len(backgrounds)) # randomly select a background file
            true_event_with_background = synthesizer.add_background_to_event(
                event, backgrounds[idx], weight=background_weight
            )
            pure_background_event = synthesizer.add_background_to_event(event, backgrounds[idx], weight=1.0)
            true_events.append(true_event_with_background)
            true_mel_spec = mel_spectrogram(true_event_with_background)
            background_mel_spec = mel_spectrogram(pure_background_event)
            filename_true = str(uuid.uuid4()) + '.pt'
            filename_background = str(uuid.uuid4()) + '.pt'
            torch.save(true_mel_spec, os.path.join(data_dir, filename_true))
            torch.save(background_mel_spec, os.path.join(data_dir, filename_background))
            training.write(os.path.join(data_dir, filename_true) + ' 1\n')
            training.write(os.path.join(data_dir, filename_background) + ' 0\n')

    for file in test_coughings:
        events = synthesizer.extract_labeled_events(file)
        for event in events:
            idx = np.random.randint(len(backgrounds)) # randomly select a background file
            true_event_with_background = synthesizer.add_background_to_event(
                event, backgrounds[idx], weight=background_weight
            )
            pure_background_event = synthesizer.add_background_to_event(event, backgrounds[idx], weight=1.0)
            true_events.append(true_event_with_background)
            true_mel_spec = mel_spectrogram(true_event_with_background)
            background_mel_spec = mel_spectrogram(pure_background_event)
            filename_true = str(uuid.uuid4()) + '.pt'
            filename_background = str(uuid.uuid4()) + '.pt'
            torch.save(true_mel_spec, os.path.join(data_dir, filename_true))
            torch.save(background_mel_spec, os.path.join(data_dir, filename_background))
            testing.write(os.path.join(data_dir, filename_true) + ' 1\n')
            testing.write(os.path.join(data_dir, filename_background) + ' 0\n')
    print(len(true_events))



def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def spectrogram(samples, sample_rate):
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

