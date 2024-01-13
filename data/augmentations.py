import numpy as np
import librosa


def add_noise(data):
    noise = np.random.normal(0, 0.1, len(data))
    audio_noisy = data + noise
    return audio_noisy


def pitch_shifting(data, sr=8000, bins_per_octave=12, pitch_pm=2):
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    data = librosa.effects.pitch_shift(
        data.astype("float64"),
        sr,
        n_steps=pitch_change,
        bins_per_octave=bins_per_octave,
    )
    return pitch_shifting


def random_shift(data):
    timeshift_fac = 0.2 * 2 * (np.random.uniform() - 0.5)  # up to 20% of length
    start = int(data.shape[0] * timeshift_fac)
    if start > 0:
        data = np.pad(data, (start, 0), mode="constant")[0 : data.shape[0]]
    else:
        data = np.pad(data, (0, -start), mode="constant")[0 : data.shape[0]]
    return data


def volume_scaling(data, sr=8000):
    dyn_change = np.random.uniform(low=1.5, high=2.5)
    data = data * dyn_change
    return data


def time_stretching(data, rate=1.5):
    input_length = len(data)
    streching = data.copy()
    streching = librosa.effects.time_stretch(streching, rate)

    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(data, (0, max(0, input_length - len(streching))), "constant")
    return streching
