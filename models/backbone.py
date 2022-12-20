import torch
from torch import nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4   
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)   
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)  
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)   
        self.bn3 = nn.BatchNorm2d(num_features=out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x    
        if self.downsample is not None:
            identity = self.downsample(x)   

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity     
        out = self.relu(out)

        return out

class ResNet18_Backbone(nn.Module):
    def __init__(self, BasicBlock=ResBlock) -> None:
        super(ResNet18_Backbone, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(BasicBlock, 64, [[1, 1],[1, 1]])

        # Attention Module
        self.inplanes = 64
        self.can_attention = ChannelAttention(self.inplanes)
        self.spa_attention = SpatialAttention()

        self.conv3 = self._make_layer(BasicBlock, 128, [[2,1],[1,1]])
        self.conv4 = self._make_layer(BasicBlock, 256, [[2,1],[1,1]])
        self.conv5 = self._make_layer(BasicBlock, 512, [[2,1],[1,1]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, 1000)

    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        out = self.can_attention(out) * out
        out = self.spa_attention(out) * out

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        # out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        return out

class ResNet50_Backbone(nn.Module):
    def __init__(self, block=Bottleneck, num_classes=1000):
        super(ResNet50_Backbone, self).__init__()
        self.in_channel = 64    

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)     
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     
        self.layer1 = self._make_layer(block=block, channel=64, block_num=3, stride=1) 

        # Attention Module
        self.inplanes = 64
        self.can_attention = ChannelAttention(self.inplanes)
        self.spa_attention = SpatialAttention()

        self.layer2 = self._make_layer(block=block, channel=128, block_num=4, stride=2)  
        self.layer3 = self._make_layer(block=block, channel=256, block_num=6, stride=2)  
        self.layer4 = self._make_layer(block=block, channel=512, block_num=3, stride=2)  

        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))  
        # self.fc = nn.Linear(in_features=512*block.expansion, out_features=num_classes)

        # # weight init
        # for m in self.modules():    
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None  
        if stride != 1 or self.in_channel != channel*block.expansion:   
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel*block.expansion, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(num_features=channel*block.expansion))

        layers = []  
        layers.append(block(in_channel=self.in_channel, out_channel=channel, downsample=downsample, stride=stride)) 
        self.in_channel = channel*block.expansion   

        for _ in range(1, block_num):  
            layers.append(block(in_channel=self.in_channel, out_channel=channel))

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

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

class ResNet18_Backbone_FPN(nn.Module):
    def __init__(self, BasicBlock=ResBlock) -> None:
        super(ResNet18_Backbone_FPN, self).__init__()
        self.basic_backbone = ResNet18_Backbone(BasicBlock)
        self.average_pooling = nn.AvgPool2d(kernel_size=2)
    
    def forward(self, x):
        feature1 = self.basic_backbone(x)
        # x = self.average_pooling(x)
        feature2 = self.basic_backbone(x)
        return [feature1, feature2]

if __name__ == '__main__':
    data = torch.randn(16, 3, 240, 240)
    net = ResNet18_Backbone_FPN()
    # net = ResNet50_Backbone(Bottleneck)
    out = net(data)
    print(out[0].shape)
    print(out[1].shape)
