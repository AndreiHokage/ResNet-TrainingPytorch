from collections import Counter

import torch
import  torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch.autograd._functions import tensor
from torch.utils.data import DataLoader, Subset


# Create model

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        # print("------------------------------------------------BEGIN-------------------------------------------------------------------------")
        # print("in_channel= ", in_channels, " intermediate_channels= ", intermediate_channels, "identiti_downsmple= ", identity_downsample, " strive= ", stride)
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False);
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()

        self.identity_downsample = identity_downsample
        self.stride = stride

        # print("------------------------------------------------FINISH-------------------------------------------------------------------------")


    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class block1834(nn.Module):

    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block1834, self).__init__()
        self.cv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=(3,3), padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.cv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=(3,3), padding=0, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU()

        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.cv1(x)
        x = self.bn1(x)
        x = self.cv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []
        # print("Num_residual= ", num_residual_blocks, "  in_channels= ", self.in_channels, " intermediate_channels= ", intermediate_channels, " stride= ", stride)

        # stride  != 1 -> resize the image, the identity feature map and the residual one has different sizes
        # self.in_channels != intermediate_Channels * 4 -> different #channel for both feature maps
        # multiplying the identity mapping by the Ws linear projection term to align the dimension of the inputs
        # Overall, the input it s skipped over a block
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            # print(">>> Skip Connection dotted - change the size of input and residual")
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5
indices = torch.arange(100)

#Load data
train_dataset = Subset(datasets.CIFAR10(root="dataset/", train=True, transform=transforms.ToTensor(), download=True), indices)
test_dataset = Subset(datasets.CIFAR10(root="dataset/", train=False, transform=transforms.ToTensor(), download=True), indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#Initialize network

def ResNet18(img_channel=3, num_classes = 10):
    return ResNet([2, 2, 2, 2], img_channel, num_classes).to(device)

def ResNet34(img_channel=3, num_classes = 10):
    return ResNet([3, 4, 6, 3], img_channel, num_classes).to(device)

def ResNet50(img_channel=3, num_classes = 10):
    return ResNet([3, 4, 6, 3], img_channel, num_classes).to(device)

def ResNet101(img_channel=3, num_classes=10):
    return ResNet([3, 4, 23, 3], img_channel, num_classes).to(device)


def ResNet152(img_channel=3, num_classes=10):
    return ResNet([3, 8, 36, 3], img_channel, num_classes).to(device)

# Loss and optimizer
net = ResNet34()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward
        scores = net(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()  # clean the previous gradients
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model is
def check_accuracy(loader, net):
    num_correct = 0
    num_samples = 0
    net.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = net(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    net.train()
    return  num_correct/num_samples

print(f"Accuracy on training set: {check_accuracy(train_loader, net)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, net)*100:.2f}")



























