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
makeDirectory(workingDir)
makeDirectory(resourcesDir)
makeDirectory("outputs")
print("Directory used to read and write model/image files: " + rootDir)
amlLogger = getAmlLogger()
if amlLogger != []:
    amlLogger.log("amlrealworld.ImageClassificationUsingCntk.2_refineDNN", "true")

# Download pre-trained ResNet model if not yet downloaded
if not os.path.exists(cntkPretrainedModelPath):
    modelUrl = urlPretrainedDNNs[rf_pretrainedModelFilename.lower()]
    print("One-time download of pretrained {} DNN from {} (this can take between 5-20 minutes)...".format(rf_pretrainedModelFilename, modelUrl))
    data = downloadFromUrl(modelUrl)
    writeBinaryFile(cntkPretrainedModelPath, data)

# If classifier is set to svm, then no need to run any training iterations
if classifier == 'svm':
    rf_maxEpochs = 0

# Load data
lutLabel2Id  = readPickle(lutLabel2IdPath)
lutId2Label  = readPickle(lutId2LabelPath)
imgDictTest  = readPickle(imgDictTestPath)
imgDictTrain = readPickle(imgDictTrainPath)

# Generate cntk test and train data, i.e. (image, label) pairs and write
# them to disk since in-memory passing is currently not supported by cntk
dataTest  = getImgLabelList(imgDictTest,  imgOrigDir, lutLabel2Id)
dataTrain = getImgLabelList(imgDictTrain, imgOrigDir, lutLabel2Id)

# Optionally add duplicates to balance dataset.
# Note: this should be done using data point weighting (as is done for svm training), rather than using explicit duplicates.
if rf_boBalanceTrainingSet:
    dataTrain = cntkBalanceDataset(dataTrain)

# Print training statistics
print("Statistics training data:")
counts = collections.Counter(getColumn(dataTrain,1))
for label in range(max(lutLabel2Id.values())+1):
    print("   Label {:10}({}) has {:4} training examples.".format(lutId2Label[label], label, counts[label]))

# Train model
# Note: Currently CNTK expects train/test splits to be provided as actual file, rather than in-memory
printDeviceType(boGpuRequired = True)
writeTable(cntkTestMapPath,  dataTest)
writeTable(cntkTrainMapPath, dataTrain)
model = train_model(cntkPretrainedModelPath, cntkTrainMapPath, cntkTestMapPath, rf_inputResoluton,
                    rf_maxEpochs, rf_mbSize, rf_maxTrainImages, rf_lrPerMb, rf_momentumPerMb, rf_l2RegWeight,
                    rf_dropoutRate, rf_boFreezeWeights)
model.save(cntkRefinedModelPath)
print("Stored trained model at %s" % cntkRefinedModelPath)

# Show training plot
if classifier != 'svm':
    print("Showing plot of DNN accuracy vs training epochs.")
    fig = plt.figure()
    fig.savefig('outputs/dnnTraining.jpg', bbox_inches='tight', dpi = 200)
    plt.show() # Accuracy vs training epochs plt
print("DONE.")