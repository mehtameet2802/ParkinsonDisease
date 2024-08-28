from io import BytesIO
import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from typing import Type
from torch import Tensor
from PIL import Image


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CNN(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int = 1000
    ) -> None:
        super(CNN, self).__init__()
        if num_layers == 18:
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:

            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def predMRI(imgPath):
    model = CNN(1, 18, BasicBlock, 2)
    model.load_state_dict(torch.load("cnn100epochclamp01Crop.pt"))
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(),
        # transforms.Resize((640, 640)),
        # transforms.CenterCrop((500, 500)),
        transforms.functional.invert,
        # transforms.Resize((224, 224)),
        transforms.CenterCrop((90, 90)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(imgPath)
    img = transform(img)
    maxIntensity = torch.max(img)
    img = torch.clamp(img, 0.4,  maxIntensity - 0.1)
    img = img.unsqueeze(0)
    with torch.no_grad():
        predictions = model(img)
        print(predictions)
        _, yhat = torch.max(predictions.data, 1)
        p = yhat.item()
    return p

def main():
    st.title("Parkinson Disease Detection")
    st.header("Mri Based Parkinson Detection")
    file = st.file_uploader("", type=["png","jpg"])
    show_file = st.empty()

    if not file:
        show_file.info("Please Upload a file : {}".format(' '.join(["png","jpg"])))
        return
    content = file.getvalue()

    if isinstance(file, BytesIO):
        # show_file.image(file)
        image_loc = "D:/College/Be Project/BE/FINAL/testing2/"
        result = predMRI(image_loc+file.name)
        if result==0:
            st.header("Patient is Parkinson Negative")
        else:
            st.header("Patient is Parkinson Positive")

    file.close()
