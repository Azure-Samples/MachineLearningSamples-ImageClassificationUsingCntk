# -*- coding: utf-8 -*-
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("libraries")
sys.path.append("../libraries")
from helpers import *
from PARAMETERS import *


####################################
# Main
####################################
random.seed(0)
makeDirectory(rootDir)
makeDirectory(procDir)
amlLogger = getAmlLogger()
if amlLogger != []:
    amlLogger.log("amlrealworld.ImageClassificationUsingCntk.1_prepareData", "true")

imgDictTest  = dict()
imgDictTrain = dict()
subdirs = getDirectoriesInDirectory(imgOrigDir)
print("Directory used to read and write model/image files: " + rootDir)

# Load dedicated test split if provided
if pathExists(dedicatedTestSplitPath):
    print("Using dedicated test split.")
    deducatedTestSplitSet = set(["/".join(line).lower() for line in readTable(dedicatedTestSplitPath)])
    for subdir in subdirs:
        imgDictTest[subdir]  = []
        imgDictTrain[subdir] = []
else:
    deducatedTestSplitSet = []

# Split images into train and test
print("Split images into train or test...")
for subdir in subdirs:
    filenames = getFilesInDirectory(pathJoin(imgOrigDir, subdir), ".jpg")

    # Use dedicated test split
    if deducatedTestSplitSet != []:
        for filename in filenames:
            key = "/".join([subdir, filename]).lower()
            if key in deducatedTestSplitSet:
                imgDictTest[subdir].append(filename)
            else:
                imgDictTrain[subdir].append(filename)

    # Randomly assign images into train or test
    elif imagesSplitBy == 'filename':
        filenames  = randomizeList(filenames)
        splitIndex = int(ratioTrainTest * len(filenames))
        imgDictTrain[subdir] = filenames[:splitIndex]
        imgDictTest[subdir]  = filenames[splitIndex:]

    # Randomly assign whole subdirectories to train or test
    elif imagesSplitBy == 'subdir':
        if random.random() < ratioTrainTest:
            imgDictTrain[subdir] = filenames
        else:
            imgDictTest[subdir]  = filenames
    else:
        raise Exception("Variable 'imagesSplitBy' has to be either 'filename' or 'subdir', but is: " + imagesSplitBy)

    # Debug print
    if subdir in imgDictTrain:
        print("   Training: {:5} images in directory {}".format(len(imgDictTrain[subdir]), subdir))
    if subdir in imgDictTest:
        print("   Testing:  {:5} images in directory {}".format(len(imgDictTest[subdir]), subdir))


# Save assignments of images to train or test
writePickle(imgDictTrainPath, imgDictTrain)
writePickle(imgDictTestPath,  imgDictTest)

# Mappings label <-> id
lutId2Label = dict()
lutLabel2Id = dict()
for index, key in enumerate(imgDictTrain.keys()):
    lutLabel2Id[key] = index
    lutId2Label[index] = key
writePickle(lutLabel2IdPath, lutLabel2Id)
writePickle(lutId2LabelPath, lutId2Label)

# Sanity check: ensure train and test do not contain the same images
for label in imgDictTrain.keys():
    for filename in imgDictTrain[label]:
        assert(filename not in imgDictTest[label])

print("DONE.")
