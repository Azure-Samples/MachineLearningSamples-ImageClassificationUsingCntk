# -*- coding: utf-8 -*-
import sys
from sklearn import svm, metrics
from shutil import copyfile
from utilities_general_v2 import *
from utilities_CVbasic_v2 import *
random.seed(0)




################################
# Shared helper functions
################################
def getAmlLogger():
    try:
        from azureml.logging import get_azureml_logger
        run_logger = get_azureml_logger()
    except:
        print("Azure ML logger not found.")
        run_logger = []
    return run_logger

def getModelNode(classifier):
    if classifier.startswith("svm"):
        node = "poolingLayer"
    else:
        node = []
    return(node)

def getImgLabelList(imgDict, imgDir, lut = None):
    imgLabelList = []
    for label in imgDict.keys():
        for imgFilename in imgDict[label]:
            imgPath = pathJoin(imgDir, label, imgFilename) #imgDir + "/" + str(label) + "/" + imgFilename
            if lut is None:
                imgLabelList.append((imgPath, label))
            else:
                imgLabelList.append((imgPath, lut[label]))
    return imgLabelList

def getSvmInput(imgDict, features, boL2Normalize, lutLabel2Id = []):
    feats = []
    labels = []
    imgFilenames = []
    for label in list(imgDict.keys()):
        for imgFilename in imgDict[label]:
            feat = features[label][imgFilename]
            if boL2Normalize:
                feat /= np.linalg.norm(feat, 2)
            feats.append(np.float32(feat))
            if lutLabel2Id == []:
                labels.append(label)
            else:
                labels.append(int(lutLabel2Id[label]))
            imgFilenames.append(imgFilename)
    return feats, labels, imgFilenames

def runClassifier(classifier, dnnOutput, imgDict = [],  lutLabel2Id = [], svmPath = [], svm_boL2Normalize = []):
    # Run classifier on all known images, if not otherwise specified
    if imgDict == []:
        imgDict = {}
        for label in list(dnnOutput.keys()):
            imgDict[label] = list(dnnOutput[label].keys())

    # Compute SVM classification scores
    if classifier.startswith('svm'):
        learner = readPickle(svmPath)
        feats, gtLabels, imgFilenames = getSvmInput(imgDict, dnnOutput, svm_boL2Normalize, lutLabel2Id)
        print("Evaluate SVM...")
        scoresMatrix = learner.decision_function(feats)

        # If binary classification problem then manually create 2nd column
        # Note: scoresMatrix is of size nrImages x nrClasses
        if len(scoresMatrix.shape) == 1:
            scoresMatrix = [[-scoresMatrix[i],scoresMatrix[i]] for i in range(len(scoresMatrix))]
            scoresMatrix = np.array(scoresMatrix)

    # Get DNN classification scores
    else:
        gtLabels = []
        scoresMatrix = []
        imgFilenames = []
        for label in list(imgDict.keys()):
            for imgFilename in imgDict[label]:
                scores = dnnOutput[label][imgFilename]
                if lutLabel2Id == []:
                    gtLabels.append(label)
                else:
                    gtLabels.append(int(lutLabel2Id[label]))
                scoresMatrix.append(scores)
                imgFilenames.append(imgFilename)
        scoresMatrix = np.vstack(scoresMatrix)
    return scoresMatrix, imgFilenames, gtLabels

def runClassifierOnImagePaths(classifier, dnnOutput, svmPath = [], svm_boL2Normalize = []):
    dnnOutputDict = {"dummy":{} }
    for i,feat in enumerate(dnnOutput):
        dnnOutputDict["dummy"][str(i)] = feat
    scoresMatrix, _, _ = runClassifier(classifier, dnnOutputDict, [], [], svmPath, svm_boL2Normalize)
    return scoresMatrix


################################
# Script-specific helper functions
################################
## 3_refineDNN
def cntkBalanceDataset(imgLabelList):
    duplicates = []
    counts = collections.Counter(getColumn(imgLabelList,1))
    print("Before balancing of training set:")
    for item in counts.items():
        print("   Class {:3}: {:5} exmples".format(*item))

    # Get duplicates to balance dataset
    targetCount = max(getColumn(counts.items(), 1))
    while min(getColumn(counts.items(),1)) < targetCount:
        for imgPath, label in imgLabelList:
            if counts[label] < targetCount:
                duplicates.append((imgPath, label))
                counts[label] += 1

    # Add duplicates to original dataset
    print("After balancing: all classes now have {} images; added {} duplicates to the {} original images.".format(targetCount, len(duplicates), len(imgLabelList)))
    imgLabelListDup = imgLabelList + duplicates
    counts = collections.Counter(getColumn(imgLabelListDup,1))
    assert(min(counts.values()) == max(counts.values()) == targetCount)
    return imgLabelListDup

## 5_trainSVM
def sklearnAccuracy(learner, feats, gtLabels):
    estimatedLabels = learner.predict(feats)
    confusionMatrix = metrics.confusion_matrix(gtLabels, estimatedLabels)
    return cmGetAccuracies(confusionMatrix, gtLabels)

def printFeatLabelInfo(title, feats, labels, preString = "   "):
    print(title)
    print(preString + "Number of examples: {}".format(len(feats)))
    print(preString + "Number of positive examples: {}".format(sum(np.array(labels) == 1)))
    print(preString + "Number of negative examples: {}".format(sum(np.array(labels) == 0)))
    print(preString + "Dimension of each example: {}".format(len(feats[0])))

## Jupyter Notebooks
# Make to different data structure. While not technically
# necessary, this improves code-readability.
def create_dataset(imgDict):
    imgCounter = 0
    dataset = Dataset()
    for label in imgDict:
        dataset.addLabel(label)
        for imgFilename in imgDict[label]:
            imgObj = DatasetImage(imgFilename, label, imgCounter)
            dataset.addImage(imgObj)
            imgCounter += 1
    return dataset

class DatasetImage(object):
    def __init__(self, filename, label, idVal):
        self.filename = filename
        self.label    = label
        self.idVal    = idVal
    def __str__(self):
        return ("Filename: {}, Ground truth label: {}, ID: {}, ".format(self.filename, self.label, self.idVal))

class Dataset(object):
    def __init__(self):
        self.images = []
        self.labels = []
    def addImage(self, image):
        self.images.append(image)
    def addLabel(self, label):
        self.labels.append(label)

def wImread(imgObj, imgOrigDir):
    imgPath = os.path.join(imgOrigDir, imgObj.label, imgObj.filename)
    if not os.path.exists(imgPath):
        raise Exception("Image {} does not exist.".format(imgPath))
    imgBytes = open(imgPath, "rb").read()
    return imgBytes

## Deployment notebook
def copyFiles(fileNames, srcFolder, dstFolder, boThrowFileDoesNotExistsError = False):
    for file in fileNames:
        src = os.path.join(srcFolder, file)
        dst = os.path.join(dstFolder, file)
        if os.path.exists(src):
            copyfile(src, dst)
        elif boThrowFileDoesNotExistsError:
            raise Exception("Source file {} does not exist.".format(src))

def deleteFiles(fileNames, rootDir, boThrowFileDoesNotExistsError = False):
    for filename in fileNames:
        path = os.path.join(rootDir, filename)
        if os.path.exists(path):
            os.remove(path)