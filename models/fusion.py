import torch
from torch import nn

class ConcatFusion(nn.Module):
    def __init__(self):
        super(ConcatFusion, self).__init__()

    def forward(self, feature_map):
        return torch.cat(feature_map, dim=1)

class MAFF(nn.Module):
    def __init__(self, channels=64, ratio1=4, ratio2=8):
        super(MAFF, self).__init__()
        inter_channels1 = int(channels // ratio1)
        inter_channels2 = int(channels // ratio2)
 
        self.global_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels1),
            nn.LeakyReLU(),
            nn.Conv2d(inter_channels1, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels1),
            nn.LeakyReLU(),
            nn.Conv2d(inter_channels1, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # self.global_att2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(channels, inter_channels2, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels2),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(inter_channels2, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels),
        # )
        # self.local_att2 = nn.Sequential(
        #     nn.Conv2d(channels, inter_channels2, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels2),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(inter_channels2, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels),
        # )

        self.sigmoid = nn.Sigmoid()
 
    def forward(self, featuremap):
        xa = featuremap[0] + featuremap[1]
        xl1 = self.local_att1(xa)
        xg1 = self.global_att1(xa)
        # xl2 = self.local_att2(xa)
        # xg2 = self.global_att2(xa)
        xlg = xl1 + xg1 #+ xl2 + xg2
        wei = self.sigmoid(xlg)
        xo = featuremap[0] * wei + featuremap[1] * (1 - wei)
        return xo

if __name__ == '__main__':
    inputs1 = torch.rand(10, 512, 1, 1)
    inputs2 = torch.rand(10, 512, 1, 1)
    layer = MAFF(512, 4)
    out = layer(inputs1, inputs2)
    print(out.shape)