import matplotlib.pyplot as plt 
import os
import processTestImages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    #Preprocess images
    pti = processTestImages.processTestImages()
    outDirectory=pti.processImageDirectory()    

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(outDirectory, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2) 

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

    for epoch in range(10):  # loop over the dataset multiple times

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
            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')
    
    def plotImages(images,results,batch,saveDir):
        fig = plt.figure(figsize=(36,36)) 
        font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
        plt.rcParams.update({'font.size': 22})
        for i,image in enumerate(images):
            fig.add_subplot(8, 4, i+1)
            result=results[i]
            label="prediction: {r1} confidence: {r2}.".format(r1=result[0], r2=round(result[1].item(),3))           
            plt.imshow(image.numpy()[0])
            plt.axis('off') 
            plt.title(label)
            plt.subplots_adjust(wspace=2,hspace=2)
            plt.tight_layout()
        saveFile=os.path.join(saveDir,"save{b1}.png".format(b1=batch))
        fig.savefig(saveFile)
            
    
    def makePredictions(test, model):
        with torch.no_grad():
            # create figure 
            saveDir=os.path.join(outDirectory,"results")
            os.mkdir(saveDir)
            predictions = []
            i=0
            for data in test:
                i=i+1
                images, _ = data
                outputs = model(images)
                outputs=F.softmax(outputs,dim=1)
                confidence, predicted = torch.max(outputs.data,1)   
                results=torch.stack((predicted,confidence),dim=1)
                predictions.append(results)
                plotImages(images,results,i,saveDir)
            #plt.waitforbuttonpress()
            return predictions
        
    predictions = makePredictions(testloader, net)
   # print(predictions)
    