import IPython.display as ipd

import os
import pandas as pd
import librosa
import glob
import librosa.display
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from keras.callbacks import EarlyStopping

from keras import regularizers

from tensorflow import keras

from sklearn.preprocessing import LabelEncoder

from datetime import datetime

import os
from moviepy.editor import *


def openfileDialog():
    import easygui
    path = easygui.fileopenbox()
    return path


def createaClip(path):
    clip = VideoFileClip(path)
    clip.audio.write_audiofile('testfile' + '.wav')


def createaSubClip(path, start, end):
    clip = VideoFileClip(path)
    clip = clip.subclip(start, end)
    clip.audio.write_audiofile("testfile" + '.wav')


# Although this function was modified and many parameters were explored with, most of it
# came from Source 8 (sources in the READ.ME)

def extract_features_files(filename):
    # Sets the name to be the path to where the file is in my computer
    file_name = filename

    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)

    return mfccs, chroma, mel, contrast, tonnetz


def predict(filepath, model):
    features, labels = np.empty((0, 193)), np.empty(0)
    mfccs, chroma, mel, contrast, tonnetz = extract_features_files(filepath)
    extracted_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features = np.vstack([features, extracted_features])
    feats = np.array(features)
    # Make predictions
    y_prediction = model.predict_classes(feats)
    return y_prediction


def main():
    model = keras.models.load_model('model_saved.h5')
    Categories = ['Ahsan Iqbal', 'Arif Alvi', 'Asad Umer', 'Asif Ghafoor', 'Asif Zardari', 'Bilawal Bhutto',
                  'Fawad Chaudary', 'Fayyaz Ul Hassan', ' Fazal Ul Rehman', ' Hammad Azhar', 'Imran Khan',
                  'Khawaja Asif']
    print("Welcome to Binary Audio Classification \n 1.Short Clip\n 2.Long Clip\n 3.Exit\n")
    x = int(input('Enter Your Choice: '))
    if x == 1:
        path = openfileDialog()
        createaClip(path=path)
        preds = predict('testfile.wav', model=model)
        print(Categories[preds[0]])
    elif x == 2:
        path = openfileDialog()
        start = int(input('Enter start of subclip: '))
        end = int(input('Enter end of SubClip : '))
        createaSubClip(path=path, start=start, end=end)
        preds = predict('testfile.wav', model=model)
        print(Categories[preds[0]])
        os.remove('testfile.wav')
    else:
        return


main()
