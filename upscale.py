
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
    iHeight = 32
    iWidth = 32
    width = 63
    height = 63
    ans = np.zeros(shape=(height, iWidth), dtype=np.uint8)

    # currInd = 1
    # imgInd = 0
    # print(type(ans), ans.shape)
    # print(type(image), image.shape)
    # ans[0] = image[0]
    # imgInd = 1
    # while currInd < height-1:
        
    #     ans[currInd] = (image[imgInd-1]+image[imgInd])/2
    #     print("VALS: ", image[imgInd-1], image[imgInd], ans[currInd])
    #     currInd= currInd+2
    #     ans[currInd-1] = image[imgInd]
    #     imgInd = imgInd+1
    
    for i in range(iHeight-1):
        ans[i*2] = image[i]
        ans[i*2+1] = (image[i]+image[i+1])/2
        print(ans[i*2+1, 10:20])
        
    # ans[-1] = image[-1]
    
    return ans



    # image = normalizeHelper(image)
    
    # return image
def normalizeHelper(image):
    ans = []
    
    return ans

arr = upscalerSimpleRepeat(imageArr)
newImg = Image.fromarray(arr)
newImg.show()
IMG.show()