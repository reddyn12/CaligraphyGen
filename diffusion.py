import numpy as np

import torch
import torchvision
from PIL import Image


PATH = "data/Images/Images"
# 14990 FONTS

data = torchvision.datasets.ImageFolder(root=PATH)

print(data)

#TODO: Diffusion is a bit overkill, lets do pixel to pixel mapping