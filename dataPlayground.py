import numpy as np

import torch
import torchvision
from PIL import Image
PATH = "data/"

# 14990 FONTS


data = np.load(PATH + "character_font.npz")
print(type(data))
print(data.files)
# print(data["images"].shape)
# print(data["labels"].shape)
print(data["images"][0])
print(data["labels"][1000])




PATH = "data/Images/Images"


# data = torchvision.datasets.ImageFolder(root=PATH, transform=torchvision.transforms.ToTensor())

data = torchvision.datasets.ImageFolder(root=PATH)



print(data)
print(data[100])
# Image.open(data[100][0])
data[100][0].show()
t = data[100][0].convert("L")

t.show()