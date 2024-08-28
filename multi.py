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

class CNN_multi(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 1000
    ) -> None:
        super(CNN_multi, self).__init__()
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
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)


        ## Fdopa
        self.convf1 = nn.Conv2d(1, 64, kernel_size=7, stride = 2, padding=1, bias=False)
        self.bnf1 = nn.BatchNorm2d(64)

        self.convf2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnf2 = nn.BatchNorm2d(64)
        self.convf3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnf3 = nn.BatchNorm2d(64)

        self.convf4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnf4 = nn.BatchNorm2d(64)
        self.convf5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnf5 = nn.BatchNorm2d(64)

        self.d1 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)

        self.convf6 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bnf6 = nn.BatchNorm2d(128)
        self.convf7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnf7 = nn.BatchNorm2d(128)

        self.d2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)

        self.convf8 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bnf8 = nn.BatchNorm2d(256)
        self.convf9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnf9 = nn.BatchNorm2d(256)

        self.d3 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)

        self.convf10 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bnf10 = nn.BatchNorm2d(512)
        self.convf11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnf11 = nn.BatchNorm2d(512)

        self.convf12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnf12 = nn.BatchNorm2d(512)
        self.convf13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnf13 = nn.BatchNorm2d(512)

        self.fcf = nn.Linear(512*self.expansion, num_classes)

        self.fcAll1 = nn.Linear(2, 1)
        self.fcAll2 = nn.Linear(2, 1)
        self.softmax = nn.Softmax(dim=1)

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
        x1, x2 = x

        # Processing for MRI images
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)

        x1 = self.avgpool(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.fc(x1)


        # Processing for FDOPA images
        x2 = self.convf1(x2)
        x2 = self.bnf1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        i = x2
        x2 = self.convf2(x2)
        x2 = self.bnf2(x2)
        x2 = self.relu(x2)
        x2 = self.convf3(x2)
        x2 = self.bnf3(x2)
        x2 += i
        x2 = self.relu(x2)

        i = x2
        x2 = self.convf4(x2)
        x2 = self.bnf4(x2)
        x2 = self.relu(x2)
        x2 = self.convf5(x2)
        x2 = self.bnf5(x2)
        x2 += i
        x2 = self.relu(x2)

        i = x2
        x2 = self.convf6(x2)
        x2 = self.bnf6(x2)
        x2 = self.relu(x2)
        x2 = self.convf7(x2)
        x2 = self.bnf7(x2)
        i = self.d1(i)
        x2 += i
        x2 = self.relu(x2)

        i = x2
        x2 = self.convf8(x2)
        x2 = self.bnf8(x2)
        x2 = self.relu(x2)
        x2 = self.convf9(x2)
        x2 = self.bnf9(x2)
        i = self.d2(i)
        x2 += i
        x2 = self.relu(x2)

        i = x2
        x2 = self.convf10(x2)
        x2 = self.bnf10(x2)
        x2 = self.relu(x2)
        x2 = self.convf11(x2)
        x2 = self.bnf11(x2)
        i = self.d3(i)
        x2 += i
        x2 = self.relu(x2)


        i = x2
        x2 = self.convf12(x2)
        x2 = self.bnf12(x2)
        x2 = self.relu(x2)
        x2 = self.convf13(x2)
        x2 = self.bnf13(x2)
        x2 += i
        x2 = self.relu(x2)

        x2 = self.avgpool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fcf(x2)


        # Concatenate features from both modalities
        x11 = x1.detach()
        x11 = self.softmax(x11)



        x12 = x2.detach()
        x12 = self.softmax(x12)

        xc = torch.cat((x11[:,0].unsqueeze(1), x12[:,0].unsqueeze(1)), dim=1)
        xp = torch.cat((x11[:,1].unsqueeze(1), x12[:,1].unsqueeze(1)), dim=1)


        xc = self.fcAll1(xc)
        xp = self.fcAll2(xp)

        x = torch.cat((xc, xp), dim=1)

        return x1, x2, x
    
def multimodalPred(imgPathMRI, imgPathFdopa):
    model = CNN_multi(1, 18, BasicBlock, 2)
    model.load_state_dict(torch.load("multimodal5.pt"))
    model.eval()
    transformMRI = transforms.Compose([
        transforms.Grayscale(),
        transforms.functional.invert,
        transforms.CenterCrop((90, 90)),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    transformFdopa = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((640, 640)),
            transforms.CenterCrop((250, 250)),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
    ])
    imgMRI = Image.open(imgPathMRI)
    imgMRI = transformMRI(imgMRI)
    maxIntensity = torch.max(imgMRI)
    imgMRI = torch.clamp(imgMRI, 0.4,  maxIntensity - 0.1)
    imgMRI = imgMRI.unsqueeze(0)
    
    imgFdopa = Image.open(imgPathFdopa)
    imgFdopa = transformFdopa(imgFdopa)
    maxIntensity = torch.max(imgFdopa)
    minIntensity = maxIntensity - 0.175
    mask = (imgFdopa >= minIntensity) & (imgFdopa <= maxIntensity)
    imgFdopa[~mask] = minIntensity - 0.02
    imgFdopa = imgFdopa.unsqueeze(0)
    with torch.no_grad():
        _, _, predictions = model((imgMRI, imgFdopa))
        _, yhat = torch.max(predictions.data, 1)
        p = yhat.item()
    return p

def main():
    st.title("Parkinson Disease Detection")
    st.header("Multi Modal Based Parkinson Detection")
    file_mri = st.file_uploader("Please upload MRI scan", type=["png","jpg"])
    file_fdopa = st.file_uploader("Please upload FDOPA scan",type=["png","jpg"])
    show_file = st.empty()

    if not file_mri:
        show_file.info("Please Upload MRI scan : {}".format(' '.join(["png","jpg"])))
        return

    if not file_fdopa:
        show_file.info("Please Upload FDOPA scan : {}".format(' '.join(["png","jpg"])))
        return

    content = file_mri.getvalue()

    if isinstance(file_mri, BytesIO) and isinstance(file_fdopa,BytesIO):
        # show_file.image(file)
        image_loc = "D:/College/Be Project/BE/FINAL/testing2/"
        result = multimodalPred(image_loc+file_mri.name,image_loc+file_fdopa.name)
        if result==0:
            st.header("Patient is Parkinson Negative")
        else:
            st.header("Patient is Parkinson Positive")

    file_mri.close()
    file_fdopa.close()