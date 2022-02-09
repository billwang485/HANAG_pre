import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['vgg16_deform_bottleneck']

def bottleneck(c):
    return nn.Sequential(
        nn.Conv2d(c, c//4, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(c//4),
        nn.ReLU(inplace=True),
        nn.Conv2d(c//4, c, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True)
    )

class VGG_deform_bottleneck(nn.Module):
    def __init__(self):
        super(VGG_deform_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)

        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU(inplace=True)

        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU(inplace=True)

        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 
        self.bn12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.relu13 = nn.ReLU(inplace=True)

        self.max_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

        self.skip_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128)
        self.skip_bn1 = nn.BatchNorm2d(128)

        self.skip_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256)
        self.skip_bn2 = nn.BatchNorm2d(256)

        self.skip_conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512)
        self.skip_bn3 = nn.BatchNorm2d(512)

        self.dw_conv1 = bottleneck(64)
        self.dw_conv2 = bottleneck(128) #nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        self.dw_conv3 = bottleneck(256) #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
        self.dw_conv4 = bottleneck(512) #nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.dw_conv5 = bottleneck(512) #nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)

    def forward(self, x):
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x2 = self.relu2(self.bn2(self.conv2(x1)))
        x2dw = self.dw_conv1(x2)
       
        x2 = x2 + x2dw

        xp1 = self.max_pool1(x2)

        x3 = self.relu3(self.bn3(self.conv3(xp1)))
        x4 = self.relu4(self.bn4(self.conv4(x3)))
        x4dw = self.dw_conv2(x4)

        x4 = x4 + x4dw

        xp2 = self.max_pool2(x4)

        #x_skip = self.skip_bn1(self.skip_conv1(x3))

        #xp2 = x_skip + xp2

        x5 = self.relu5(self.bn5(self.conv5(xp2)))
        x6 = self.relu6(self.bn6(self.conv6(x5)))
        x7 = self.relu7(self.bn7(self.conv7(x6)))
        x7dw = self.dw_conv3(x7)

        x7 = x7 + x7dw

        #x7 = x7 + x5

        xp3 = self.max_pool3(x7)

        #x_skip2 = self.skip_bn2(self.skip_conv2(x5))
        #xp3 = xp3 + x_skip2

        x8 = self.relu8(self.bn8(self.conv8(xp3)))
        x9 = self.relu9(self.bn9(self.conv9(x8)))
        x10 = self.relu10(self.bn10(self.conv10(x9)))
        x10dw = self.dw_conv4(x10)

        x10 = x10 + x10dw

        xp4 = self.max_pool4(x10)

        #x_skip3 = self.skip_bn3(self.skip_conv3(x8))
        #xp4 = xp4 + x_skip3

        x11 = self.relu11(self.bn11(self.conv11(xp4)))
        x12 = self.relu12(self.bn12(self.conv12(x11)))
        x13 = self.relu13(self.bn13(self.conv13(x12)))
        x13dw = self.dw_conv4(x13)

        x13 = x13 + x13dw

        xp5 = self.max_pool5(x13)

        xo = xp5.view(xp5.shape[0], -1)
        out = self.classifier(xo)
        return out

def vgg16_deform_bottleneck():
    return VGG_deform_bottleneck()
