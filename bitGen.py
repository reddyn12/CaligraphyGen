import numpy as np

import torch
import torchvision
from PIL import Image
from torchvision.transforms import Compose


PATH = "data/Images/Images"
ALPHABET = []

def showIMG(i):
    i[0].show()
def getLabel(i):
    return i[1]

# 14990 FONTS

def greyScaleHelper(x):
    return x.convert("L")
def dimReduceHelper(x):
    return x[0]



data = torchvision.datasets.ImageFolder(root=PATH, 
                                        transform=Compose([greyScaleHelper, 
                                                           torchvision.transforms.ToTensor(), 
                                                        #    dimReduceHelper,

                                                           
                                                           ]))

#TODO: This is the simplest form of image generation. Will only output same size image as trained images.
#Can probab;ly do funky interpolation between pixels to get higher quily outputed image like TSODING does in his cloding stream


# showIMG(data[100])
# print(getLabel(data[100]))

print(data[1][0])

