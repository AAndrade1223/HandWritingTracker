from PIL import Image
import torch
import torchvision
from torchvision.datasets.dtd import PIL
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF

if __name__ == '__main__':
    
    #PREPROCESS DATA    
    batch_size = 1
    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # DEFINE MODEL
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
    # CREATE MODEL
    net = Net()
    
    # TRAIN MODEL
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    
    #SAVE MODEL 
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    
    #PREPROCESS QUERY IMAGE
    test_image_path='data/testImage/e/e0.png'
    pil_image = Image.open(test_image_path)
    grey_transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize((28,28)), transforms.ToTensor()])
    torch_tensor = grey_transform(pil_image)

    #PREDICT WITH MODEL
    net.eval()
    net(torch_tensor)
    #RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x16 and 256x120)
    # on line 40: x = F.relu(self.fc1(x))
    net.train()
    