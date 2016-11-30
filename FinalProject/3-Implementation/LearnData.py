import matplotlib.pyplot as plt
import numpy as np
import librosa

import ExtractFeatures as ef

[trainX, trainY, testX, testY] = ef.LoadData()
trainFolder  = "./Data/diyDataset/train/"

print trainX[0:3]
print trainY[0:3]
cmnFile = trainFolder + trainX[0] + ".wav"
engFile = trainFolder + trainX[1] + ".wav"

print cmnFile
print engFile

y, sr = librosa.load(cmnFile)
print "end", y

D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D)

 # Passing through arguments to the Mel filters
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

import matplotlib.pyplot as plt
librosa.display.specshow(librosa.logamplitude(S,
                                               ref_power=np.max),
                          y_axis='mel', fmax=8000,
                          x_axis='time')
plt.colorbar()
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

y, sr = librosa.load(engFile)
print "cmm", y
D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D)

 # Passing through arguments to the Mel filters
#S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

import matplotlib.pyplot as plt
librosa.display.specshow(librosa.logamplitude(S,
                                               ref_power=np.max),
                          y_axis='mel', fmax=8000,
                          x_axis='time')

plt.colorbar()
plt.title('Mel spectrogram')
plt.tight_layout()

plt.show()