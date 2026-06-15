import torch
from torchvision import models
import torch.nn as nn

model = models.squeezenet1_1(weights=None)
print(model.classifier)
