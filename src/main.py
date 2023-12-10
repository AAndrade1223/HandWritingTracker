import argparse
import csv
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
    parser = argparse.ArgumentParser(description = "Argument Parser")
    parser.add_argument("-tr", "--trainpath", help = "path to directory of train images", required = False, default = "")
    parser.add_argument("-te", "--testpath", help = "path to directory of test images", required = False, default = "")
    parser.add_argument("-r", "--resultpath", help = "path to directory where results are stored", required = False, default = "")  
    parser.add_argument("-c", "--buildCSV", help = "flag to build csv file of results", required = False, action="store_false")  #default true
    parser.add_argument("-p", "--plotResults", help = "flag to plot images with results", required = False, action="store_false")  #default true
    args = parser.parse_args()

    #Get Args
    trainPath = args.trainpath
    testPath = args.testpath
    resultPath = args.resultpath
    buildCSV = args.buildCSV
    plotResults = args.plotResults
    
    #Validate Args
    if os.path.isdir(trainPath):
        print("Train Image Path: " + trainPath)
    else:
        trainPath=""
        print("Using Default Training Dataset.")
        
    if os.path.isdir(testPath):
        print("Test Image Path: " + testPath)
    else:
        testPath=""
        print("Using Default Testing Dataset.")
        
    if os.path.isdir(resultPath):
        print("Final Results Path: " + resultPath)
    else:
        resultPath=""
        print("Using Default Results Directroy. See \".\\HandWritingTracker\\runs\"")
        
    if not (buildCSV or plotResults):
        print("Warning: Results will not be reported or saved. Run again with either -c buildCSV or -p plotResults left as defaulted true.")
        
    #Preprocess Test Images
    pti = processTestImages.processTestImages()
    processedTestImagesPath,resultPath=pti.processTestImageDirectory(testPath,resultPath)    

    #Prepare Training Dataset
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 32
    if trainPath:
        trainset = torchvision.datasets.ImageFolder(trainPath, transform=transform)
    else:
        trainset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        )   
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = trainset.classes
    
    #Prepare Validation DataSet
    validateset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    )
    validationloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    #Prepare Testset
    testset = torchvision.datasets.ImageFolder(processedTestImagesPath, transform=transform)   
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2) 

    #Define Model
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

    #Create Model
    net = Net()

    #Train Model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Training Model...")
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
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'Batch: {epoch + 1}, Size: {i + 1}, Loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    print('Finished Training.')
    

    def validateModel():
        print("Validating Model...")
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            for data in validationloader:
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        
    validateModel()    
    
    def writeToCSV(resultpath,results):
        resultsCSV=os.path.join(resultPath,"results.csv") 
        with open(resultsCSV,'w+') as fd:
            for res in results:
                results_str = "{r0},{r1},{r2},{r3}\n".format(r0=res[0].item(),
                                                                r1=res[1].item(),
                                                                r2=res[2].item(),
                                                                r3=res[3].item())                  
                fd.write(results_str)
        print("Batch Result Data Saved to CSV.")

    #Create Results Plot by Batch
    def plotImages(images,results,batch,filepath):
        fig = plt.figure(figsize=(36,36)) 
        font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
        plt.rcParams.update({'font.size': 22})
        for i,image in enumerate(images):
            fig.add_subplot(8, 4, i+1)
            classif,conf,var,varflag = results[i]
            if (varflag==2):
                var="{:.2f}".format(var)
                label="var: {r0}".format(r0=var)
            else:
                conf="{:.2f}".format(conf)
                label="class: {r1}, conf: {r2}".format(r1=classif,r2=conf)  
            plt.imshow(image.numpy()[0])
            plt.axis('off') 
            plt.title(label)
            plt.subplots_adjust(wspace=2,hspace=2)
            plt.tight_layout()
        filepath=os.path.join(filepath,"batch_{b1}.png".format(b1=batch))
        fig.savefig(filepath)
        print("Plot of Batch Images Saved.")
    
    #Make Predictions of Test Images using Model
    def makePredictions(test, model):
        print("Making Classifications on Test Data...")
        with torch.no_grad(): 
            predictions = []
            i=0
            for data in test:
                print("Testing Batch: {b0} of {b1}".format(b0=i+1,b1=len(test)))
                i=i+1
                images, _ = data
                logits = model(images)
                results = analyzeLogits(logits)
                predictions.append(results)
                if buildCSV:
                    writeToCSV(resultPath,results)
                if plotResults:
                    plotImages(images,results,i,resultPath)
                #plt.show() 
        print("Testing Complete.")
        print("See {p0} for result data.".format(p0=resultPath))
        return predictions
    
    # Analyze logits using softmax and varmax
    def analyzeLogits(logits:torch.Tensor):
        variance = torch.var(torch.abs(logits), dim = 1) 
        varmax_mask = variance < 0.75
        shape = logits.shape
        unknown = torch.zeros(shape[0], device=logits.device)
        unknown[varmax_mask] = 2
        confidence, classif = torch.max(torch.softmax(logits, dim=-1),1)
        output = torch.stack([classif, confidence, variance, unknown], dim = -1)
        return output
        
    #run
    predictions = makePredictions(testloader, net)
    