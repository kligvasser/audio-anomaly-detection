import torch.nn as nn


CODE_NUM_CHANNELS = 2


class EncBlock(nn.Module):
    def __init__(
        self, in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2
    ):
        super(EncBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class DecBlock(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        padding=1,
        stride=2,
        output_padding=1,
    ):
        super(DecBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            output_padding=output_padding,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels=[64, 128, 256, 512, 512]):
        super(Encoder, self).__init__()

        self.first_conv = nn.Conv2d(
            in_channels, num_channels[0], kernel_size=7, stride=2, padding=3
        )
        self.last_channel = num_channels[0]

        blocks = list()
        for num_channel in num_channels:
            blocks.append(
                EncBlock(in_channels=self.last_channel, out_channels=num_channel)
            )
            self.last_channel = num_channel
        self.features = nn.Sequential(*blocks)

        self.code_conv = nn.Conv2d(
            self.last_channel, CODE_NUM_CHANNELS, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        c = self.code_conv(x)
        return c


class Decoder(nn.Module):
    def __init__(self, out_channels, num_channels=[512, 512, 256, 128, 64]):
        super(Decoder, self).__init__()
        self.code_conv = nn.ConvTranspose2d(
            CODE_NUM_CHANNELS, num_channels[0], kernel_size=8, stride=2, padding=3
        )
        self.last_channel = num_channels[0]

        blocks = list()
        for last_channel in num_channels:
            blocks.append(
                DecBlock(in_channels=self.last_channel, out_channels=last_channel)
            )
            self.last_channel = last_channel
        self.features = nn.Sequential(*blocks)

        self.last_conv = nn.Conv2d(
            self.last_channel, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.code_conv(x)
        x = self.features(x)
        c = self.last_conv(x)
        return c


class VGGAutoencoder(nn.Module):
    def __init__(self, in_channels, num_channels=[64, 128, 256, 512, 512]):
        super().__init__()
        self.encoder = Encoder(in_channels, num_channels)
        self.decoder = Decoder(in_channels, num_channels[::-1])

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    import torch

    enc = Encoder(3, [64, 128, 256, 512, 512])
    dec = Decoder(3, [512, 512, 256, 128, 64])

    x = torch.rand(2, 3, 256, 256)
    y = enc(x)
    xx = dec(y)

    au = VGGAutoencoder(3)
    y = au(x)

    print(xx.size(), y.size())
