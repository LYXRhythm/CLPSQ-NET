import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import ResNet18_Backbone_FPN, ResNet18_Backbone 
from models.fusion import MAFF, ConcatFusion

# stage one, unsupervised learning
class ModelStage1(nn.Module):
    def __init__(self, args, feature_dim=128):
        super(ModelStage1, self).__init__()
        self.args = args
        self.backbone = ResNet18_Backbone()

        self.fc = nn.Sequential(nn.Linear(512, 256, bias=False),
                               nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True),
                               nn.Linear(256, feature_dim, bias=True))
        
        self.backbone = self.add_pretrained(self.backbone)

    def add_pretrained(self, model):
        if self.args.use_pretrained:
            net_dict = model.state_dict()

            predict_model = torch.load(self.args.pretrained_path)
            state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
            net_dict.update(state_dict)  
            model.load_state_dict(net_dict)
        else:
            pass
        return model

    def forward(self, x):
        x = self.backbone(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
    
# stage two, supervised learning
class ModelStage2(torch.nn.Module):
    def __init__(self, args):
        super(ModelStage2, self).__init__()
        # encoder
        self.backbone = ResNet18_Backbone()

        # classifier
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.fc_stage2 = nn.Sequential(nn.Linear(512, 2000, bias=False),
                               nn.BatchNorm1d(2000),
                               nn.ReLU(inplace=True),
                               nn.Dropout(0.4),
                               nn.Linear(2000, args.num_classes, bias=True))
    
    def forward(self, x):
        x = self.backbone(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc_stage2(feature)
        return out
    
