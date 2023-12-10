from datetime import datetime
import matplotlib.pyplot as plt
import os
import torch

# Helper Functions


def validatePathArgs(argPath, argDescStr):
    if not argPath:
        return ""
    elif os.path.isfile(argPath):
        print(
            "{s0} path is a file. Must be directory. {s1}".format(
                s0=argDescStr, s1=argPath
            )
        )
        print("Using default path.")
        return ""
    elif not os.path.isdir(argPath):
        print("{s0} path is not valid: {s1}".format(s0=argDescStr, s1=argPath))
        print("Using default path.")
        return ""
    else:
        print("{s0} path accepted: {s1}".format(s0=argDescStr, s1=argPath))
        return argPath


def setMakePaths(trainPath, validPath, testPath, resultPathParent):
    # cwdPath              = absolute path of this file
    # copyTestPath         = absolute path of where orginal test images are copied to in results folder
    # pccdTestImgPath      = absolute path of processed test images
    # pccdTestImgClssPath  = absolute path of class file for processed test images
    # resultPath           = absolute path of directory where the result plots and csv will be stored
    # |-cwdPath (..\HandWritingTracker)
    # |----runs
    # |-------datetime
    # |-----------originalTestImages
    # |-----------processedTestImages
    # |---------------images
    # |-----------results
    # |----src
    # |------data
    # |-----------sampleNumbersOneImage

    # Current Working Directory
    cwdPath = os.path.dirname(os.path.abspath(__file__))

    # resultPathParent = parent folder for this run
    todaynowstr = getTodayNowStr()
    if not resultPathParent:
        resultPathParent = os.path.dirname(cwdPath)
        resultPathParent = os.path.join(resultPathParent, "runs\\")
    resultPathParent = os.path.join(resultPathParent, todaynowstr)

    # subfolders for this run:

    # original test images are copied FROM this folder
    if not testPath:
        copyFromTestPath = os.path.join(cwdPath, "data\\testData\\testImages\\")
    else:
        copyFromTestPath = testPath

    # original test images are copied TO this folder
    copyToTestPath = os.path.join(resultPathParent, "originalTestImages\\")

    # processed test images are stored in class folder "images" in this folder
    pccdTestImgPath = os.path.join(resultPathParent, "processedTestImages\\")
    pccdTestImgClssPath = os.path.join(pccdTestImgPath, "images\\")

    # result data is stored in this folder
    resultPath = os.path.join(resultPathParent, "results\\")

    makeFileStructure(copyToTestPath, pccdTestImgClssPath, resultPath)

    # return absolute paths
    paths = [
        cwdPath,
        trainPath,
        validPath,
        copyFromTestPath,
        copyToTestPath,
        pccdTestImgPath,
        pccdTestImgPath,
        pccdTestImgClssPath,
        resultPath,
    ]
    absPaths = absPath(paths)
    return absPaths


# Recursively makes directories
def makeFileStructure(copyTestPath, pccdTestImgClssPath, resultPath):
    os.makedirs(copyTestPath)
    os.makedirs(pccdTestImgClssPath)
    os.makedirs(resultPath)


# Return absolte paths for list of paths
def absPath(paths):
    absPaths = []
    for path in paths:
        if path:
            path = os.path.abspath(path)
        absPaths.append(path)
    return absPaths


# Analyze logits using softmax and varmax
def analyzeLogits(logits: torch.Tensor, varThreshold):
    variance = torch.var(torch.abs(logits), dim=1)
    varmax_mask = variance < varThreshold
    shape = logits.shape
    unknown = torch.zeros(shape[0], device=logits.device)
    unknown[varmax_mask] = 2
    confidence, classif = torch.max(torch.softmax(logits, dim=-1), 1)
    output = torch.stack([classif, confidence, variance, unknown], dim=-1)
    return output


# write results to CSV
def writeToCSV(resultPath, results):
    resultsCSV = os.path.join(resultPath, "results.csv")
    with open(resultsCSV, "w+") as fd:
        for res in results:
            results_str = "{r0},{r1},{r2},{r3}\n".format(
                r0=res[0].item(), r1=res[1].item(), r2=res[2].item(), r3=res[3].item()
            )
            fd.write(results_str)
    print("Batch Result Data Saved to CSV.")


# Create Results Plot by Batch
def plotImages(images, results, batch, filepath):
    fig = plt.figure(figsize=(36, 36))
    font = {"family": "normal", "weight": "bold", "size": 22}
    plt.rcParams.update({"font.size": 22})
    for i, image in enumerate(images):
        fig.add_subplot(8, 4, i + 1)
        classif, conf, var, varflag = results[i]
        if varflag == 2:
            var = "{:.2f}".format(var)
            label = "var: {r0}".format(r0=var)
        else:
            conf = "{:.2f}".format(conf)
            label = "class: {r1}, conf: {r2}".format(r1=classif, r2=conf)
        plt.imshow(image.numpy()[0])
        plt.axis("off")
        plt.title(label)
        plt.subplots_adjust(wspace=2, hspace=2)
        plt.tight_layout()
    filepath = os.path.join(filepath, "batch_{b1}.png".format(b1=batch))
    fig.savefig(filepath)
    print("Plot of Batch Images Saved.")


def getTodayNowStr():
    return (
        datetime.utcnow()
        .strftime("%Y-%m-%d %H:%M:%S.%f")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
    )