import torch
import torchvision
from torchvision.transforms import Compose
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from model import CharPredLinear
import torch.nn.functional as F
import sys
from tqdm import tqdm

PATH = "data/Images/Images"
batch_size_train = 2048  #128
batch_size_test = 1000 * 1
batch_size_val = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 10
num_epochs = 4

def greyScaleHelper(x):
    return x.convert("L")
def dimReduceHelper(x):
    return x[0]

# data = torchvision.datasets.ImageFolder(root=PATH, transform=torchvision.transforms.ToTensor())


data = torchvision.datasets.ImageFolder(root=PATH, 
                                        transform=Compose([greyScaleHelper, 
                                                           torchvision.transforms.ToTensor(), 
                                                        #    dimReduceHelper,

                                                           
                                                           ]))


# data=data[0:101]
# print(data.samples)
# data
trainData, testData, valData = random_split(data, [.7, .25, .05])
print(trainData)
print(testData)
print(valData)
print(data[0][0].shape)

trainLoader = DataLoader(trainData, batch_size=batch_size_train, shuffle=True)
testLoader = DataLoader(testData, batch_size=batch_size_test, shuffle=True)
valLoader = DataLoader(valData, batch_size=batch_size_val, shuffle=True)

model = CharPredLinear().to("cuda")

criterion = torch.nn.CrossEntropyLoss()
criterion=F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(trainLoader):
        data = data.to("cuda")
        targets = targets.to("cuda")
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(data)
        # outputs = torch.argmax(outputs, dim=1)
        # outputs = F.soft
        # print(outputs)
        # print(targets)


        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:  # Print the running loss every 100 batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(trainLoader)}], Loss: {running_loss/100:.7f}')
            running_loss = 0.0
        if batch_idx%50 == 0:
            "Validating"
            model.eval()  # Set the model to evaluation mode
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for data, targets in valLoader:
                    data=data.to("cuda")
                    targets = targets.to("cuda")
                    outputs = model(data)
                    _, predicted = torch.max(outputs, 1)
                    total_samples += targets.size(0)
                    total_correct += (predicted == targets).sum().item()

            validation_accuracy = 100 * total_correct / total_samples
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {validation_accuracy:.2f}%')

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, targets in valLoader:
            data=data.to("cuda")
            targets = targets.to("cuda")
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()

    validation_accuracy = 100 * total_correct / total_samples
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {validation_accuracy:.2f}%')

print("Training finished!")

name = "SimpleCharPred"
torch.save(model.state_dict(), "models/"+name+".pth")