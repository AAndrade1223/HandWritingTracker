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

if __name__ == '__main__':
    
    batch_size = 4

    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    #dataiter = iter(trainloader)
    # images, labels = next(dataiter)

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # Cretaing CNN 

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

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

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
    
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    
    test_image_path='data/testImage/sample.jpg'
 
    # Reads a file using pillow
    PIL_image = PIL.Image.open(test_image_path)

    # The image can be converted to tensor using
    tensor_image = transforms.ToTensor(PIL_image)

    # The tensor can be converted back to PIL using
    # new_PIL_image = transform.to_pil_image(tensor_image)

    net.eval()
    net.predict(tensor_image)
    # TypeError: __init__() takes 1 positional argument but 2 were given
    net.train()

    