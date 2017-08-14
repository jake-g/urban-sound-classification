# Sound Classification

### TODO
* improve RNN
* tune params
* improve accuracy

### Setup
On area51 the features and labels have already been serialized in `/data/urban_sound/`

On another machine....
* Edit `PATHS.yaml`
* run `dump_features`

### Run
* run `fnn.py` or `cnn.py`
* results are printed and saved in `./results` or whatever you put in `PATHS.yaml`

### Models
#### Feed Forward Net
* `fnn.py`, `fnn2.py`
* Accuracy: ~0.58
* Runtime:  ~40 sec
* Features: average value of several librosa features

#### Convolutional Net
* `cnn.py`, `cnn2.py`
* Accuracy: ~0.77
* Runtime:  ~3 min
* Features: windowed spectrograms (log scale, mel-frequency)

#### Recurrent Net
* `rnn.py`
* untested
* Features: mfcc coefficients

### Dataset
 [UrbanSound8k](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html)

### Features
`dump_features.py` extracts different features from the dataset and saves each feature set with an accompanying labels file (see `utils.py` for feature notes). Different models use different features. 

**WARNING** takes nearly 2 hrs and 8 gbs of space

### Requirements
Python 2.7
* Keras
* Tensorflow 11
* Numpy
* Matplotlib
* Librosa

--jake