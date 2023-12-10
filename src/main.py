import argparse

from numpy import var
import helper_functions as hf
import model as m
import model_functions as mf
import os
import process_test_images as pti
import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    #Set up argument parser
    parser = argparse.ArgumentParser(description = "Argument Parser")
    parser.add_argument("-ct", "--confidencethreshold", help = "good prediction confidence threshold between 0 and 1. Default 0.8", required = False, default = 0.8)
    parser.add_argument("-vt", "--variancethreshold", help = "accaptable confidence variance threshold. Default 0.75", required = False, default = 0.75)
    parser.add_argument("-tr", "--trainpath", help = "path to directory of trainging dataset of images", required = False, default = "")
    parser.add_argument("-v", "--validpath", help = "path to directory of validation dataset of images", required = False, default = "")
    parser.add_argument("-te", "--testpath", help = "path to directory of test images", required = False, default = "")
    parser.add_argument("-r", "--resultpath", help = "path to directory where results are stored", required = False, default = "")  
    parser.add_argument("-c", "--buildCSV", help = "flag to build csv file of results", required = False, action="store_false")  #default true
    parser.add_argument("-p", "--plotResults", help = "flag to plot images with results", required = False, action="store_false")  #default true
    parser.add_argument("-o", "--overrideExit", help = "flag to override any early exits", required = False, action="store_true")  #default false
    args = parser.parse_args()

    #Get / Validate Args
    confThreshold = args.confidencethreshold
    varThreshold = args.variancethreshold
    trainPath = hf.validatePathArgs(args.trainpath, "Training Dataset")
    validPath = hf.validatePathArgs(args.validpath, "Validation Dataset")
    testPath = hf.validatePathArgs(args.testpath, "Testing Images")
    resultPath = hf.validatePathArgs(args.resultpath, "Results")
    buildCSV = args.buildCSV
    plotResults = args.plotResults
    overrideExit = args.overrideExit
    
    if confThreshold > 1 or confThreshold < 0:
        print("Threshold should be between 0 and 1")
        if not overrideExit:
            exit()           
       
    if not (buildCSV or plotResults):
        print("Warning: Results will not be reported or saved. Run again with either -c buildCSV or -p plotResults left as defaulted true.") 
        if not overrideExit:
            exit()
            
    #creates needed directories and returns absolute paths
    [cwdPath,trainPath,validPath,copyFromTestPath,copyToTestPath,pccdTestImgPath,pccdTestImgPath,pccdTestImgClssPath,resultPath] = hf.setMakePaths(trainPath,validPath,testPath,resultPath)

    #Prepare Training Dataset
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 32
    #If custom training set
    if trainPath:
        trainset = torchvision.datasets.ImageFolder(trainPath, transform=transform)
    #else default training set
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
    
    #Preprocess Test Images
    pti.processTestImageDirectory(cwdPath,copyFromTestPath,copyToTestPath,pccdTestImgClssPath,overrideExit)   

    #Prepare Testset
    # If there are no test images, exit early
    if not os.listdir(pccdTestImgClssPath):
        print("No test images available.")
        print("Exiting early to avoid crash...")
        exit()
    testset = torchvision.datasets.ImageFolder(pccdTestImgPath, transform=transform)   
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2) 
     
    #Create Model
    mod = m.model()

    #Train Model
    mf.trainModel(mod,trainloader)
        
    #Validate Model
    mf.validateModel(mod,validationloader,classes)    
      
    #Made Predictions
    mf.predictWithModel(testloader,mod,resultPath,buildCSV,plotResults,confThreshold,varThreshold)
    