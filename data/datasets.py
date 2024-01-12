import torch
from torchvision import transforms

import data.misc


class AudioLibary(torch.utils.data.Dataset):
    def __init__(self, root="", sampling_rate=8000, segment_length=10, max_size=None):
        self.root = root
        self.sampling_rate = sampling_rate
        self.segment_size = segment_length * sampling_rate
        self.max_size = max_size

        self._init()

    def _init(self):
        self.paths = data.misc.get_path_list(self.root, self.max_size)
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def preprocess_audio(self, path):
        audio, sampling_rate = data.misc.read_audio_file(path)

        if sampling_rate > self.sampling_rate:
            audio = data.misc.downsample_audio(audio, sampling_rate, self.sampling_rate)
            sampling_rate = self.sampling_rate

        audio = data.misc.cut_random_segment(audio, self.segment_size)
        spectrogram = data.misc.audio_to_melspectrogram(audio, sampling_rate)

        spectrogram = self.preprocess(spectrogram)
        return spectrogram

    def __getitem__(self, index):
        example = dict()

        path = self.paths[index]
        example["image"] = self.preprocess_image(path)
        example["path"] = path

        return example

    def __len__(self):
        return len(self.paths)
