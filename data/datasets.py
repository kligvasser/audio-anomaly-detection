import torch
from torchvision import transforms

import data.misc


SEGMENT_SIZE = 1024 * 8
HOP_LENGHT = SEGMENT_SIZE // 256


class AudioLibary(torch.utils.data.Dataset):
    def __init__(
        self,
        root="",
        sampling_rate=8000,
        segment_size=SEGMENT_SIZE,
        hop_length=HOP_LENGHT,
        max_size=None,
    ):
        self.root = root
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.max_size = max_size

        self._init()

    def _init(self):
        self.paths = data.misc.get_path_list(self.root, self.max_size)
        self.preprocess = transforms.Compose(  # TODO: check presentation
            [
                # transforms.Resize(),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ]
        )

    def preprocess_audio(self, path):
        audio, sampling_rate = data.misc.read_audio_file(path)

        if sampling_rate > self.sampling_rate:
            audio = data.misc.downsample_audio(audio, sampling_rate, self.sampling_rate)
            sampling_rate = self.sampling_rate

        audio = data.misc.cut_random_segment(audio, self.segment_size - 1)
        spectrogram = data.misc.audio_to_melspectrogram(
            audio, sampling_rate, hop_length=self.hop_length
        )

        spectrogram = self.preprocess(spectrogram)
        return spectrogram

    def __getitem__(self, index):
        example = dict()

        path = self.paths[index]
        example["input"] = self.preprocess_audio(path)
        example["path"] = path

        return example

    def __len__(self):
        return len(self.paths)
