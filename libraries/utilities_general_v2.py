# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from builtins import input
from builtins import map
from builtins import str
from builtins import zip
from builtins import range
from past.builtins import basestring
from past.utils import old_div
from builtins import object

import os, sys, pdb, random, collections, pickle, stat, codecs, itertools, shutil, datetime, importlib, requests
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from math import *
from functools import reduce
from itertools import cycle
from scipy import interp
from sklearn.metrics import *
from os.path import join as pathJoin
from os.path import exists as pathExists

###############################################################################
# Description:
#    This is a collection of general utility / helper functions.
#
# Typical meaning of variable names:
#    lines,strings = list of strings
#    line,string   = single string
#    table         = 2D row/column matrix implemented using a list of lists
#    row,list1D    = single row in a table, i.e. single 1D-list
#    rowItem       = single item in a row
#    list1D        = list of items, not necessarily strings
#    item          = single item of a list1D
###############################################################################


#################################################
# File access
#################################################
def readFile(inputFile):
    # Comment from Python 2: reading as binary, to avoid problems with end-of-text
    #    characters. Note that readlines() does not remove the line ending characters
    with open(inputFile,'rb') as f:
        lines = f.readlines()
        #lines = [unicode(l.decode('latin-1')) for l in lines]  convert to uni-code
    return [removeLineEndCharacters(s.decode('utf8')) for s in lines];

def readBinaryFile(inputFile):
    with open(inputFile,'rb') as f:
        bytes = f.read()
    return bytes

def readPickle(inputFile):
    with open(inputFile, 'rb') as filePointer:
         data = pickle.load(filePointer)
    return data

def readTable(inputFile, delimiter='\t', columnsToKeep=None):
    # Note: if getting memory errors then use 'readTableFileAccessor' instead
    lines = readFile(inputFile);
    if columnsToKeep != None:
        header = lines[0].split(delimiter)
        columnsToKeepIndices = listFindItems(header, columnsToKeep)
    else:
        columnsToKeepIndices = None;
    return splitStrings(lines, delimiter, columnsToKeepIndices)

def writeFile(outputFile, lines, header=None, encoding=None):
    if encoding == None:
        with open(outputFile,'w') as f:
            if header != None:
                f.write("%s\n" % header)
            for line in lines:
                f.write("%s\n" % line)
    else:
        with codecs.open(outputFile, 'w', encoding) as f:  # e.g. encoding=utf-8
            if header != None:
                f.write("%s\n" % header)
            for line in lines:
                f.write("%s\n" % line)

def writeTable(outputFile, table, header=None):
    lines = tableToList1D(table)
    writeFile(outputFile, lines, header)

def writeBinaryFile(outputFile, data):
    with open(outputFile,'wb') as f:
        bytes = f.write(data)
    return bytes

def writePickle(outputFile, data):
    p = pickle.Pickler(open(outputFile,"wb"))
    p.fast = True
    p.dump(data)

def getFilesInDirectory(directory, postfix = ""):
    if not os.path.exists(directory):
        return []
    fileNames = [s for s in os.listdir(directory) if not os.path.isdir(directory+"/"+s)]
    if not postfix or postfix == "":
        return fileNames
    else:
        return [s for s in fileNames if s.lower().endswith(postfix)]

def getFilesInSubdirectories(directory, postfix = ""):
    paths = []
    for subdir in getDirectoriesInDirectory(directory):
        for filename in getFilesInDirectory(os.path.join(directory, imgSubdir), postfix):
            paths.append(os.path.join(directory, subdir, filename))
    return paths

def getDirectoriesInDirectory(directory):
    return [s for s in os.listdir(directory) if os.path.isdir(directory+"/"+s)]

def makeDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def makeOrClearDirectory(directory):
    # Note: removes just the files in the directory, not recursive
    makeDirectory(directory)
    files = os.listdir(directory)
    for file in files:
        filePath = directory +"/"+ file
        os.chmod(filePath, stat.S_IWRITE )
        if not os.path.isdir(filePath):
            os.remove(filePath)

def removeWriteProtectionInDirectory(directory):
    files = os.listdir(directory)
    for file in files:
        filePath = directory +"/"+ file
        if not os.path.isdir(filePath):
            os.chmod(filePath, stat.S_IWRITE )

def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)

def deleteAllFilesInDirectory(directory, fileEndswithString, boPromptUser = False):
    if boPromptUser:
        userInput = eval(input('--> INPUT: Press "y" to delete files in directory ' + directory + ": "))
        if not (userInput.lower() == 'y' or userInput.lower() == 'yes'):
            print("User input is %s: exiting now." % userInput)
            exit()
    for filename in getFilesInDirectory(directory):
        if fileEndswithString == None or filename.lower().endswith(fileEndswithString):
            deleteFile(directory + "/" + filename)



#################################################
# 1D list
#################################################
def isList(var):
    return isinstance(var, list)

def toIntegers(list1D):
    return [int(float(x)) for x in list1D]

def toRounded(list1D):
    return [round(x) for x in list1D]

def toFloats(list1D):
    return [float(x) for x in list1D]

def toStrings(list1D):
    return [str(x) for x in list1D]

def max2(list1D):
    maxVal = max(list1D)
    indices = [i for i in range(len(list1D)) if list1D[i] == maxVal]
    return maxVal,indices

def pbMax(list1D): # depricated
    return max2(list1D)

def find(list1D, func):
    return [index for (index,item) in enumerate(list1D) if func(item)]

def listSort(list1D, reverseSort=False, comparisonFct=lambda x: x):
    indices = list(range(len(list1D)))
    tmp = sorted(zip(list1D,indices), key=comparisonFct, reverse=reverseSort)
    list1DSorted, sortOrder = list(map(list, list(zip(*tmp))))
    return (list1DSorted, sortOrder) 



#################################################
# 2D list (e.g. tables)
#################################################
def getColumn(table, columnIndex):
    return [row[columnIndex] for row in table]

def getRows(table, rowIndices):    
    return [table[rowIndex] for rowIndex in rowIndices]

def getColumns(table, columnIndices):
    return [[row[i] for i in columnIndices] for row in table]

def sortTable(table, sortColumnIndex, reverseSort=False, comparisonFct=lambda x: float(x[0])):
    if len(table) == 0:
        return []
    list1D = getColumn(table, sortColumnIndex)
    _, sortOrder = listSort(list1D, reverseSort, comparisonFct)
    return [table[i] for i in sortOrder]

def tableToList1D(table, delimiter='\t'):
    return [delimiter.join([str(s) for s in row]) for row in table]



#################################################
# String and chars
#################################################
def isString(var):
    return type(var) == type("")

def numToString(num, length, paddingChar = '0'):
    if len(str(num)) >= length:
        return str(num)[:length]
    else:
        return str(num).ljust(length, paddingChar)

def splitString(string, delimiter='\t', columnsToKeepIndices=None):
    if string == None:
        return None
    items = string.split(delimiter)
    if columnsToKeepIndices != None:
        items = getColumn(items, columnsToKeepIndices)
    return items;

def splitStrings(strings, delimiter, columnsToKeepIndices=None):
    table = [splitString(string, delimiter, columnsToKeepIndices) for string in strings]
    return table;

def removeLineEndCharacters(line):
    if line.endswith('\r\n'):
        return line[:-2]
    elif line.endswith('\n'):
        return line[:-1]
    else:
        return line



#################################################
# Randomize
#################################################
def getRandomNumber(low, high):
    return random.randint(low,high)

def getRandomNumbers(low, high):
    randomNumbers = list(range(low,high+1))
    random.shuffle(randomNumbers)
    return randomNumbers

def getRandomListElement(listND, containsHeader=False):
    if containsHeader:
        index = getRandomNumber(1, len(listND)-1)
    else:
        index = getRandomNumber(0, len(listND)-1)
    return listND[index]

def randomizeList(listND, containsHeader=False):
    if containsHeader:
        header = listND[0]
        listND = listND[1:]
    random.shuffle(listND)
    if containsHeader:
        listND.insert(0, header)
    return listND



#################################################
# Dictionaries
#################################################
def getDictionary(keys, values, boConvertValueToInt = True):
    dictionary = {}
    for key, value in zip(keys, values):
        if boConvertValueToInt:
            value = int(value)
        dictionary[key] = value
    return dictionary

def sortDictionary(dictionary, sortIndex=0, reverseSort=False):
    return sorted(list(dictionary.items()), key=lambda x: x[sortIndex], reverse=reverseSort)

def invertDictionary(dictionary):
    return {v: k for k, v in list(dictionary.items())}

def dictionaryToTable(dictionary):
    return (list(dictionary.items()))

def mergeDictionaries(dict1, dict2):
    tmp = dict1.copy()
    tmp.update(dict2)
    return tmp



#################################################
# Url
#################################################
def downloadFromUrl(url, boVerbose = True):
    data = []
    try:
        r = requests.get(url, timeout=5)
        data = r.content
    except:
        if boVerbose:
            print('Error downloading url {0}'.format(url))
    #if boVerbose and data == []: # and r.status_code != 200:
    #    print('Error {} downloading url {}'.format(r.status_code, url))
    return data



#################################################
# Confusion matrix and p/r curves
# Note: Let C be the confusion matrix. Then C_{i, j} is the number of observations known to be in group i but predicted to be in group j.
#################################################
def cmSanityCheck(confMatrix, gtLabels):
    for i in range(max(gtLabels)+1):
        assert(sum(confMatrix[i,:]) == sum([l == i for l in gtLabels])) 

def cmGetAccuracies(confMatrix, gtLabels = []):
    if gtLabels != []:
        cmSanityCheck(confMatrix, gtLabels)
    return [float(confMatrix[i, i]) / sum(confMatrix[i,:]) for i in range(confMatrix.shape[1])]

def cmPrintAccuracies(confMatrix, classes, gtLabels = []):
    columnWidth = max([len(s) for s in classes])
    accs = cmGetAccuracies(confMatrix, gtLabels)
    for cls, acc in zip(classes, accs):
        print(("Class {:<" + str(columnWidth) + "} accuracy: {:2.2f}%.").format(cls, 100 * acc))
    globalAcc = 100.0 * sum(np.diag(confMatrix)) / sum(sum(confMatrix))
    print("OVERALL accuracy: {:2.2f}%.".format(globalAcc))
    print("OVERALL class-averaged accuracy: {:2.2f}%.".format(100 * np.mean(accs)))
    return globalAcc, accs

def cmPlot(confMatrix, classes, normalize=False, title='Confusion matrix', cmap=[]):
    if normalize:
        confMatrix = confMatrix.astype('float') / confMatrix.sum(axis=1)[:, np.newaxis]
        confMatrix = np.round(confMatrix * 100,1)
    if cmap == []:
        cmap = plt.cm.Blues

    #Actual plotting of the values
    thresh = confMatrix.max() / 2.
    for i, j in itertools.product(range(confMatrix.shape[0]), range(confMatrix.shape[1])):
        plt.text(j, i, confMatrix[i, j], horizontalalignment="center",
                 color="white" if confMatrix[i, j] > thresh else "black")

    avgAcc = np.mean([float(confMatrix[i, i]) / sum(confMatrix[:, i]) for i in range(confMatrix.shape[1])])
    plt.imshow(confMatrix, interpolation='nearest', cmap=cmap)
    plt.title(title + " (avgAcc={:2.2f}%)".format(100*avgAcc))
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def rocComputePlotCurves(gtLabels, scoresMatrix, labels):
    #Code taken from Microsoft AML Workbench iris tutorial
    n_classes = len(labels)
    Y_score = scoresMatrix
    Y_onehot = []
    for i in range(len(gtLabels)):
        Y_onehot.append([])
        for j in range(len(labels)):
            Y_onehot[i].append(0)
        Y_onehot[i][gtLabels[i]] = 1
    Y_onehot = np.asarray(Y_onehot)

    fpr = dict()
    tpr = dict()
    thres = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], thres[i] = roc_curve(Y_onehot[:, i], Y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_onehot.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    # fig = plt.figure(figsize=(6, 5), dpi=75)
    # set lineweight
    lw = 2

    # plot micro average
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # plot macro average
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # plot ROC for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(labels[i], roc_auc[i]))

    # plot diagnal line
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    return (fpr, tpr, thres)



#################################################
# Math
#################################################
def intRound(item):
    return int(round(float(item)))

def softmax(vec):
    expVec = np.exp(vec)
    if max(expVec) != np.inf:
        outVec = expVec / np.sum(expVec)
    else:
        # Note: this is a hack to make softmax stable
        outVec = np.zeros(len(expVec))
        outVec[expVec == np.inf] = vec[expVec == np.inf]
        outVec = outVec / np.sum(outVec)
    return outVec

def softmax2D(w):
    # Note: could replace with np.exp(w – max(w)) to make numerically stable
    e = np.exp(w)
    dist = old_div(e, np.sum(e, axis=1)[:, np.newaxis])
    return dist



#################################################
# other
#################################################
def isTuple(var):
    return isinstance(var, tuple)
