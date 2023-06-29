
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
print(imageArr[20:25, 20:25])


# newImg = Image.fromarray(imageArr)
# newImg.show()



def upscalerSimpleRepeat(image):
    width = 63
    height = 63
    ans = np.zeros(shape=(width, height))

    for h in height:
        for w in width:
            nH = None
            nW = None
            if h%2==0:
                nH = h
            if w%2==0:
                nW = w
            if(nW is None and nH is None):
                pass
            elif(nW is None):
                pass
            elif(nH is None):
                pass
            else:
                ans[nH]

    print(type(ans), ans.shape)
    print(type(image), image.shape)




    # image = normalizeHelper(image)
    
    # return image
def normalizeHelper(image):
    ans = []
    
    return ans

upscaler(imageArr)