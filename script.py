import soundfile

# import librosa
# import numpy as np
import matplotlib.pyplot as plt

# from scipy import signal
# from scipy.fft import fftshift
# import matplotlib.pyplot as plt

clean_audio, framerate = soundfile.read("audio_samples/" + "20-205-0000.flac")
noisy_audio, framerate = soundfile.read("audio_samples/" + "20-205-0000_noisy.wav")

print(clean_audio)
