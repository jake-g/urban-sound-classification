import collections
import glob
import pickle

import matplotlib

matplotlib.use('Agg')  # no Xwindows needed
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from matplotlib.pyplot import specgram
from sklearn.preprocessing import normalize, MinMaxScaler

TrainingParams = collections.namedtuple('TrainingParams', ['n_epoch', 'batch_size', 'early_stop_patience'])

FEATURE_SET_MEANS = 'means'
FEATURE_SET_SPECS = 'specs'
FEATURE_SET_MFCCS = 'mfccs'
FEATURE_SET_SPECS_NORM = 'specs_norm'

CLASS_NAMES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
               'gun_shot', 'jackhammer', 'siren', 'street_music']


'''
PROCESSING
'''

'''
The librosa library comes with several methods (http://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf), including:

Mel-frequency cepstral coefficients (MFCC) - https://en.wikipedia.org/wiki/Mel-frequency_cepstrum

Chromagram of a short-time Fourier transform - projects bins representing the 12 distinct semitones (or chroma) of the musical octave http://labrosa.ee.columbia.edu/matlab/chroma-ansyn/

Mel-scaled power spectrogram - uses https://en.wikipedia.org/wiki/Mel_scale to provide greater resolution for the more informative (lower) frequencies

Octave-based spectral contrast (http://ieeexplore.ieee.org/document/1035731/)

Tonnetz - estimates tonal centroids as coordinates in a six-dimensional interval space (https://sites.google.com/site/tonalintervalspace/)
'''

N_FFT = 1024
HOP_SIZE = 512


def extract_features_spectrograms(parent_dir, sub_dirs, file_ext="*.wav", bands=128,
                                  frames=128, normalize_data=False):  # sliding window spectrals
    log_specgrams = []
    labels = []
    if normalize_data:
        rescale = MinMaxScaler(feature_range=(0, 1), copy=True)  # rescale between 0 and 1

    for l, sub_dir in enumerate(sub_dirs):
        print('parsing %s...' % sub_dir)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, s = librosa.load(fn)
            label = fn.split('/')[fn.count('/')].split('-')[1]  # TODO change to fn.split('/').pop().split('-')[1]
            melspec = librosa.feature.melspectrogram(sound_clip, sr=s, n_mels=bands, n_fft=N_FFT, hop_length=HOP_SIZE)
            if melspec.shape[1] < frames:
                continue
            logspec = librosa.logamplitude(melspec)
            if normalize_data:
                logspec = rescale.fit_transform(get_random_patch(logspec, frames).T)
            log_specgrams.append(logspec)
            labels.append(label)
    log_specgrams = np.array(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    labels = one_hot_encode(np.array(labels, dtype=np.int))
    # # Add channel of deltas
    # features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    # for i in range(len(features)):
    #     features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    return log_specgrams, labels


def extract_features_spectrograms_norm(parent_dir, sub_dirs, file_ext="*.wav", bands=128,
                                       frames=128, normalize_data=True):  # sliding window spectrals
    # helper for dump_features
    return extract_features_spectrograms(**locals())  # pass all input vars


def extract_features_mfccs(parent_dir, sub_dirs, file_ext="*.wav", bands=20, frames=128):
    mfccs = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        print('parsing %s...' % sub_dir)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, s = librosa.load(fn)
            label = fn.split('/')[fn.count('/')].split('-')[1]
            mfcc = librosa.feature.mfcc(y=sound_clip, sr=s, n_mfcc=bands, n_fft=N_FFT, hop_length=HOP_SIZE)
            if mfcc.shape[1] < frames:
                continue
            mfccs.append(get_random_patch(mfcc, frames))
            labels.append(label)

    mfccs = np.expand_dims(np.asarray(mfccs), 3)  # .reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((mfccs, np.zeros(np.shape(mfccs))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    features = np.transpose(features, axes=[0, 2, 1, 3])

    return np.array(features), one_hot_encode(np.array(labels, dtype=np.int))


def extract_features_means(parent_dir, sub_dirs, file_ext='*.wav'):
    features, labels = np.empty((0, 193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        print('parsing %s...' % sub_dir)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:  # sometimes failss??
                # mean value of spectral content for Feed Forward Net
                X, sample_rate = librosa.load(fn)
                stft = np.abs(librosa.stft(X))
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
                contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
                tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            except:
                print('error, skipping...', fn)
                pass
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, fn.split('/')[fn.count('/')].split('-')[1])
    return np.array(features), one_hot_encode(np.array(labels, dtype=np.int))


def load_sound_files(path, file_paths):
    raw_sounds = []
    for fp in file_paths:
        X, sr = librosa.load(path + fp)
        raw_sounds.append(X)
    return raw_sounds


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def inv_one_hot_encode(labels):  # inverse (index of 1 for each row)
    return np.argmax(labels, axis=1)


def split_data_random(features, labels, split_percent=0.70):
    train_indices = np.random.rand(len(features)) < split_percent
    train_x = features[train_indices]
    train_y = labels[train_indices]
    test_x = features[~train_indices]
    test_y = labels[~train_indices]
    return train_x, train_y, test_x, test_y


def get_random_patch(frames, num_frames):
    # TODO: Randomize
    start_frame = 0
    return frames[:, start_frame:start_frame + num_frames]


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


'''
IO
'''


def load_data(f):
    print('loading %s...' % f)
    return pickle.load(open(f, "rb"), encoding="latin1")  # specifying encoding for python3 compat


def dump_data(path, objects, names, ext='.p'):
    for i, o in enumerate(objects):
        print('dumping %s...' % names[i])
        pickle.dump(o, open(path + names[i] + ext, "wb"))


def load_paths(f):
    with open(f, 'r') as ymlfile:
        paths = yaml.load(ymlfile)
        for key, path in paths.items():
            if not os.path.exists(path):
                print("Creating %s directory at %s" % (key, path))
                os.makedirs(path)
        return paths


'''
PLOTS
'''

plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13


def plot_waves(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=900)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot', x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=900)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_log_power_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=900)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        D = librosa.logamplitude(np.abs(librosa.stft(f)) ** 2, ref_power=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_all(sound_names, raw_sounds):
    plot_waves(sound_names, raw_sounds)
    plot_specgram(sound_names, raw_sounds)
    plot_log_power_specgram(sound_names, raw_sounds)


def plot_keras_loss(path, history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower left')
    plt.savefig(path + '.png')


def plot_confusion_matrix(cm, path, classes=CLASS_NAMES,  title='Confusion matrix', cmap=plt.cm.Greys):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path + '.png')



'''
TEST
'''


def plot_classes(data_path):
    sound_file_paths = [
        "57320-0-0-7.wav", "24074-1-0-3.wav", "15564-2-0-1.wav",
        "31323-3-0-1.wav", "46669-4-0-35.wav", "89948-5-0-0.wav",
        "40722-8-0-4.wav", "103074-7-3-2.wav", "106905-8-0-0.wav",
        "108041-9-0-4.wav"
    ]
    sound_names = [
        "air conditioner", "car horn", "children playing", "dog bark",
        "drilling", "engine idling", "gun shot", "jackhammer", "siren",
        "street music"
    ]
    sound_file_paths = ['fold1/' + s for s in sound_file_paths]

    raw_sounds = load_sound_files(data_path, sound_file_paths)
    plot_all(sound_names, raw_sounds)
