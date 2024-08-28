from io import BytesIO
import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from typing import Type
from torch import Tensor
from PIL import Image
    
class CNNFdopa(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_classes: int  = 1000
    ) -> None:
        super(CNNFdopa, self).__init__()
        self.expansion = 1

        self.in_channels = 64

        ## Fdopa
        self.convf1 = nn.Conv2d(1, 64, kernel_size=7, stride = 2, padding=1, bias=False)
        self.bnf1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)



    def forward(self, x: Tensor) -> Tensor:
        x = self.convf1(x)
        x = self.bnf1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        i = x
        x = self.convf2(x)
        x = self.bnf2(x)
        x = self.relu(x)
        x = self.convf3(x)
        x = self.bnf3(x)
        x += i
        x = self.relu(x)

        i = x
        x = self.convf4(x)
        x = self.bnf4(x)
        x = self.relu(x)
        x = self.convf5(x)
        x = self.bnf5(x)
        x += i
        x = self.relu(x)

        i = x
        x = self.convf6(x)
        x = self.bnf6(x)
        x = self.relu(x)
        x = self.convf7(x)
        x = self.bnf7(x)
        i = self.d1(i)
        x += i
        x = self.relu(x)

        i = x
        x = self.convf8(x)
        x = self.bnf8(x)
        x = self.relu(x)
        x = self.convf9(x)
        x = self.bnf9(x)
        i = self.d2(i)
        x += i
        x = self.relu(x)

        i = x
        x = self.convf10(x)
        x = self.bnf10(x)
        x = self.relu(x)
        x = self.convf11(x)
        x = self.bnf11(x)
        i = self.d3(i)
        x += i
        x = self.relu(x)


        i = x
        x = self.convf12(x)
        x = self.bnf12(x)
        x = self.relu(x)
        x = self.convf13(x)
        x = self.bnf13(x)
        x += i
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def predFDOPA(imgPath):
    model = CNNFdopa(1, 2)
    model.load_state_dict(torch.load("cnn25epochClamp.pt"))
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((640, 640)),
        transforms.CenterCrop((250, 250)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(imgPath)
    img = transform(img)
    maxIntensity = torch.max(img)
    minIntensity = maxIntensity - 0.175
    mask = (img >= minIntensity) & (img <= maxIntensity)
    img[~mask] = minIntensity - 0.02
    img = img.unsqueeze(0)
    with torch.no_grad():
        predictions = model(img)
        _, yhat = torch.max(predictions.data, 1)
        p = yhat.item()
    return p


def main():
    st.title("Parkinson Disease Detection")
    st.header("FDOPA Based Parkinson Detection")
    file = st.file_uploader("", type=["png","jpg"])
    show_file = st.empty()

    if not file:
        show_file.info("Please Upload a file : {}".format(' '.join(["png","jpg"])))
        return
    content = file.getvalue()

    if isinstance(file, BytesIO):
        # show_file.image(file)
        image_loc = "D:/College/Be Project/BE/FINAL/testing2/"
        result = predFDOPA(image_loc+file.name)
        if result==0:
            st.header("Patient is Parkinson Negative")
        else:
            st.header("Patient is Parkinson Positive")

    file.close()


