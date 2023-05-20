import torch 
import torch.nn
import torchvision
import torch.nn.functional as F

class FashionNet(nn.Module):
    def __init__(self, model_name: str, num_classes:int, dropout:float, freeze_backbone:bool = True):
        super(FashionNet, self).__init__()
        self.num_classes =  num_classes
        self.model_name =  model_name

        if self.model_name == "vgg-16":
            self.model = torchvision.models.vgg16(pretrained=True) 
            if freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
            backbone_out = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(backbone_out, 256)

        elif self.model_name == "resnet-50":
            self.model = torchvision.models.resnet50(pretrained=True) 
            if freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
            backbone_out = self.model.fc.in_features
            self.model.fc = nn.Linear(backbone_out, 256)

        elif self.model_name == "efficientnet-b0":
            self.model = torchvision.models.efficientnet_b0(pretrained=True) 
            if freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
            backbone_out = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(backbone_out, 256)

        elif self.model_name == "efficientnet-b7":
            self.model = torchvision.models.efficientnet_b7(pretrained=True) 
            if freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
            backbone_out = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(backbone_out, 256)

        self.dropout = nn.Dropout(dropout) if dropout != None else None
        self.fc2 = nn.Linear(256, self.num_classes)
        
    def forward(self, x):
        x = F.relu(self.model(x))
        if self.dropout != None:
            x = self.dropout(x)
        x = self.fc2(x)
        return x