import os
import matplotlib.pyplot as plt
import numpy as np
import librosa


AUDIO_EXTENSIONS = [".wav", ".mp3"]


def get_path_list(path_dir, max_size=None, extentions=AUDIO_EXTENSIONS):
    paths = list()

    for dirpath, _, files in os.walk(path_dir):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            if fname.endswith(tuple(extentions)):
                paths.append(fname)

    return sorted(paths)[:max_size]


def read_audio_file(file_path, sr=None):
    try:
        audio_data, sampling_rate = librosa.load(file_path, sr=sr)
        return audio_data, sampling_rate
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None, None


def downsample_audio(audio, original_sr, target_sr):
    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)


def audio_to_melspectrogram(audio, sampling_rate, hop_length=256, n_fft=2048):
    spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sampling_rate, hop_length=hop_length, n_fft=n_fft
    )
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def cut_random_segment(audio, segment_size):
    max_start_time = max(0, len(audio) - segment_size)
    start_time = np.random.randint(0, max_start_time)
    end_time = start_time + segment_size

    if len(audio) < end_time:
        audio = np.pad(audio[start_time:], (0, end_time - len(audio)), "constant")
    else:
        audio = audio[start_time:end_time]

    return audio


def show_melspectrogram(
    spectrogram, sampling_rate, title="log-frequency power spectrogram"
):
    librosa.display.specshow(spectrogram, x_axis="time", y_axis="mel", sr=sampling_rate)
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.show()


def show_signal(audio):
    plt.plot(audio)
    plt.title("Signal")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()


def read_as_melspectrogram(file_path):
    audio, sampling_rate = read_audio_file(file_path)
    spectrogram = audio_to_melspectrogram(audio, sampling_rate)
    return spectrogram
