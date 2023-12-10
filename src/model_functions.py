import helper_functions as hf
import torch
import torch.nn as nn
import torch.optim as optim
    
def trainModel(mod,trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mod.parameters(), lr=0.001, momentum=0.9)

    print("Training Model...")
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mod(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'Batch: {epoch + 1}, Size: {i + 1}, Loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    print('Finished Training.')
    

def validateModel(mod,validationloader,classes):
    print("Validating Model...")
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in validationloader:
            images, labels = data
            outputs = mod(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


#Make Predictions of Test Images using Model
def predictWithModel(testloader, model, resultPath, buildCSV, plotResults, confThreshold, varThreshold):
    print("Making Classifications on Test Data...")
    with torch.no_grad(): 
        lowVarCount = 0
        goodPredCount = 0
        predictions = []
        i=0
        for data in testloader:
            print("Testing Batch: {b0} of {b1}".format(b0=i+1,b1=len(testloader)))
            i=i+1
            images, _ = data
            logits = model(images)
            results = hf.analyzeLogits(logits,varThreshold)
            predictions.append(results)
            for res in results:
                if res[3] == 2:
                    lowVarCount = lowVarCount + 1
                elif res[1] >= confThreshold:
                    goodPredCount = goodPredCount+1
            if buildCSV:
                hf.writeToCSV(resultPath,results)
            if plotResults:
                hf.plotImages(images,results,i,resultPath)
                #plt.show() 
    print("Testing Complete.")
    print("Invalid samples with confidence varainces lower than {s0}: {s1}".format(s0=varThreshold,s1=lowVarCount))
    print("High confidence (>{s0}) Samples: {s1}".format(s0=confThreshold,s1=goodPredCount))
    print("See {p0} for explicit result data.".format(p0=resultPath))
    return predictions
    
