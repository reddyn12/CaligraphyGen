
import numpy as np
import torchvision

from PIL import Image
PATH = "data/Images/Images"


# data = torchvision.datasets.ImageFolder(root=PATH, transform=torchvision.transforms.ToTensor())

data = torchvision.datasets.ImageFolder(root=PATH)

SAMPLE = data[93]
IMG = SAMPLE[0]
IMG = IMG.convert("L")
print("SAMPLE_Str", SAMPLE)

# Image.open(SAMPLE[0])
# SAMPLE[0].show()
# imageArr = np.array(list(IMG.getdata()), dtype=np.uint8)
imageArr = np.array(IMG)
print(imageArr)

newImg = Image.fromarray(imageArr)
newImg.show()


def upscaler(image, width=400, height=400):
    image = normalizeHelper(image)
    
    return image
def normalizeHelper(image):
    ans = []
    
    return ans

