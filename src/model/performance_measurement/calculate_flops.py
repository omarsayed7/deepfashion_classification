import torch, sys
from pthflops import count_ops

#Local imports
sys.path.append(r"../training")
from models import FashionNet


model = FashionNet(model_name = "vgg-16", num_classes = 28, dropout = 0.5, freeze_backbone = False)

inp = torch.rand(1,3,256,256)

# Count the number of FLOPs
count_ops(model, inp)

#Total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")