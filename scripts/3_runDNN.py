# -*- coding: utf-8 -*-
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("libraries")
sys.path.append("../libraries")
from helpers import *
from helpers_cntk import *
from PARAMETERS import *


################################################
# MAIN
################################################
# Init
printDeviceType()
makeDirectory(workingDir)
node  = getModelNode(classifier)
model = load_model(cntkRefinedModelPath)
mapPath = pathJoin(workingDir, "rundnn_map.txt")
print("Directory used to read and write model/image files: " + rootDir)

amlLogger = getAmlLogger()
if amlLogger != []:
    amlLogger.log("amlrealworld.ImageClassificationUsingCntk.3_runDNN", "true")

# Compute dnn output for each image and write to disk
print("Running DNN for training set..")
dnnOutputTrain = runCntkModelAllImages(model, readPickle(imgDictTrainPath), imgOrigDir, mapPath, node, run_mbSize)
print("Running DNN for test set..")
dnnOutputTest  = runCntkModelAllImages(model, readPickle(imgDictTestPath),  imgOrigDir, mapPath, node, run_mbSize)

# Combine all dnn outputs
dnnOutput = dict()
for label in list(dnnOutputTrain.keys()):
    outTrain = dnnOutputTrain[label]
    outTest  = dnnOutputTest[label]
    dnnOutput[label] = mergeDictionaries(outTrain, outTest)

# Check if all DNN outputs are of expected size
for label in list(dnnOutput.keys()):
    for feat in list(dnnOutput[label].values()):
        assert(len(feat) == rf_modelOutputDimension)

# Save dnn output to file
print("Writting CNTK outputs to file %s ..." % dnnOutputPath)
writePickle(dnnOutputPath, dnnOutput)
print("DONE.")