import torch
import torch.nn as nn
import torch.nn.functional as F

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         return x

# #
# # Autoencoder
# #

# class EncoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv_block = ConvBlock(in_channels, out_channels)
#         self.pool = nn.MaxPool2d(2)

#     def forward(self, x):
#         x = self.conv_block(x)
#         x = self.pool(x)
#         return x

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv_block = ConvBlock(out_channels, out_channels)

#     def forward(self, x):
#         x = self.upconv(x)
#         x = self.conv_block(x)
#         return x

# class Autoencoder(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1):
#         super().__init__()
#         self.enc1 = EncoderBlock(in_channels, 64)
#         self.enc2 = EncoderBlock(64, 128)
#         self.enc3 = EncoderBlock(128, 256)
#         self.enc4 = EncoderBlock(256, 512)

#         self.bottleneck = ConvBlock(512, 1024)

#         self.dec1 = DecoderBlock(1024, 512)
#         self.dec2 = DecoderBlock(512, 256)
#         self.dec3 = DecoderBlock(256, 128)
#         self.dec4 = DecoderBlock(128, 64)

#         self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

#     def forward(self, x):
#         x = self.enc1(x)
#         x = self.enc2(x)
#         x = self.enc3(x)
#         x = self.enc4(x)

#         x = self.bottleneck(x)

#         x = self.dec1(x)
#         x = self.dec2(x)
#         x = self.dec3(x)
#         x = self.dec4(x)

#         return self.final_conv(x)
    
# #
# # UNet
# #

# class EncoderSkipBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv_block = ConvBlock(in_channels, out_channels)
#         self.pool = nn.MaxPool2d(2)

#     def forward(self, x):
#         conv = self.conv_block(x)
#         pooled = self.pool(conv)
#         return conv, pooled

# class DecoderSkipBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv_block = ConvBlock(in_channels, out_channels)

#     def forward(self, x, skip):
#         x = self.upconv(x)
#         x = torch.cat((x, skip), dim=1)
#         x = self.conv_block(x)
#         return x

# class UNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1):
#         super().__init__()
#         self.enc1 = EncoderSkipBlock(in_channels, 64)
#         self.enc2 = EncoderSkipBlock(64, 128)
#         self.enc3 = EncoderSkipBlock(128, 256)
#         self.enc4 = EncoderSkipBlock(256, 512)

#         self.bottleneck = ConvBlock(512, 1024)

#         self.dec1 = DecoderSkipBlock(1024, 512)
#         self.dec2 = DecoderSkipBlock(512, 256)
#         self.dec3 = DecoderSkipBlock(256, 128)
#         self.dec4 = DecoderSkipBlock(128, 64)

#         self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

#     def forward(self, x):
#         skip1, x = self.enc1(x)
#         skip2, x = self.enc2(x)
#         skip3, x = self.enc3(x)
#         skip4, x = self.enc4(x)

#         x = self.bottleneck(x)

#         x = self.dec1(x, skip4)
#         x = self.dec2(x, skip3)
#         x = self.dec3(x, skip2)
#         x = self.dec4(x, skip1)

#         return self.final_conv(x)

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         return x

# class EncoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv_block = ConvBlock(in_channels, out_channels)
#         self.pool = nn.MaxPool2d(2)

#     def forward(self, x):
#         conv = self.conv_block(x)
#         pooled = self.pool(conv)
#         return conv, pooled

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv_block = ConvBlock(in_channels, out_channels)

#     def forward(self, x, skip):
#         x = self.upconv(x)
#         x = torch.cat((x, skip), dim=1)
#         x = self.conv_block(x)
#         return x
    
#================== WITHOUT SKIPPED CONNECTIONS ==================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.upconv(x)
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)

        return self.final_conv(x)

class UNetNoSkips(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        return self.final_conv(x)