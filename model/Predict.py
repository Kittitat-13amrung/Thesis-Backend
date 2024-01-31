from scipy.io import wavfile
import numpy as np
import librosa

path_audio = "audio/audio_mic/"
filename = ""
file_audio = path_audio + filename + "_mic.wav"
data = wavfile.read(file_audio)

def preprocess_audio(self, data):
    data = data.astype(float)
    if self.normalize:
        data = librosa.util.normalize(data)
    if self.downsample:
        data = librosa.resample(data, orig_sr = self.sr_original, target_sr = self.sr_downs)
        self.sr_curr = self.sr_downs
    if self.preproc_mode == "c":
        data = np.abs(librosa.cqt(data,
            hop_length=self.hop_length, 
            sr=self.sr_curr, 
            n_bins=self.cqt_n_bins, 
            bins_per_octave=self.cqt_bins_per_octave))
    elif self.preproc_mode == "m":
        data = librosa.feature.melspectrogram(y=data, sr=self.sr_curr, n_fft=self.n_fft, hop_length=self.hop_length)
    elif self.preproc_mode == "cm":
        cqt = np.abs(librosa.cqt(data, 
            hop_length=self.hop_length, 
            sr=self.sr_curr, 
            n_bins=self.cqt_n_bins, 
            bins_per_octave=self.cqt_bins_per_octave))
        mel = librosa.feature.melspectrogram(y=data, sr=self.sr_curr, n_fft=self.n_fft, hop_length=self.hop_length)
        data = np.concatenate((cqt,mel),axis = 0)
    elif self.preproc_mode == "s":
        data = np.abs(librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length))
    else:
        print("invalid representation mode.")

    return data

