import utils as u

paths = u.load_paths('PATHS.yaml')  # get paths from file
AUDIO_DIR = paths['audio_data']
SAVE_DIR = paths['extracted_data']
TRAIN = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8']
VAL = ['fold9']
TEST = ['fold10']
SAMPLE = ['fold_sample']


def dump_features(name, extractor):
    print('Extracting %s' % name)
    fnames = ['features_' + name, 'labels_' + name]
    train_features, train_labels = extractor(AUDIO_DIR, TRAIN)
    val_features, val_labels = extractor(AUDIO_DIR, VAL)
    test_features, test_labels = extractor(AUDIO_DIR, TEST)
    data = [(train_features, val_features, test_features), (train_labels, val_labels, test_labels)]
    u.dump_data(SAVE_DIR, data, fnames)


def dump_sample(name, extractor):
    print('Extracting %s' % name)
    fnames = ['features_' + name, 'labels_' + name]
    samp_features, samp_labels = extractor(AUDIO_DIR, SAMPLE)
    data = [(samp_features), (samp_labels)]
    u.dump_data(SAVE_DIR, data, fnames)

if __name__ == "__main__":
    # test
    # dump_sample('sample', u.extract_features_spectrograms_norm)

    dump_features('mfccs', u.extract_features_mfccs)
    dump_features('means', u.extract_features_means)
    dump_features('specs', u.extract_features_spectrograms)
    dump_features('specs_norm', u.extract_features_spectrograms_norm)


