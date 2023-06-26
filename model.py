import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F

#loss = ops.ssim(img1, img2)


class CharPredLinear(nn.Module):
    def __init__(self):
        super(CharPredLinear, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 64, 26,26 for training

            

            nn.Flatten(),
            # This number matters as it is affected by the trainins img size.... I THINK

            nn.Linear(64*16*16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 26)
            
        )


    def forward(self, x):
        x = self.net(x)
        return F.softmax(x, dim=1)
