import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc

wave_path = "test.wav"

sample_rate, X = scipy.io.wavfile.read(wave_path)
ceps, mspec, spec = mfcc(X)
print ceps.shape
print mspec.shape
print spec.shape

num_ceps = len(ceps)
print np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
