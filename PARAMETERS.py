# -*- coding: utf-8 -*-
import sys, os
sys.path.append("libraries")
sys.path.append("../libraries")
from helpers import getDirectoriesInDirectory, pathJoin


###################
# Parameters
###################
classifier = 'svm' #options: 'svm': to keep pre-trained DNN fixed and use a SVM as classifier (this does not require a DNN, all other options do)
                   #         'dnn': to refine the DNN and use it as the classifier
                   #         'svmDnnRefined': this first refines the DNN (like option 'dnn') but then trains a SVM classifier on its output(like option 'svm')

datasetName = "fashionTexture"

# Train and test splits (script: 1_prepareData.py)
ratioTrainTest = 0.75                   # Percentage of images used for training of the DNN and the SVM
imagesSplitBy  = 'filename'             # Options: 'filename' or 'subdir'. If 'subdir' is used, then all images in a subdir are assigned fully to train or test

# Model refinement parameters (script: 2_refineDNN.py)
rf_pretrainedModelFilename = "ResNet_18.model"  # Pre-trained ImageNet model
rf_inputResoluton = 224                 # DNN image input width and height in pixels. ALso try e.g. 4*224=896 pixels.
rf_dropoutRate    = 0.5                 # Droputout rate
rf_mbSize         = 16                  # Minibatch size (reduce if running out of memory)
rf_maxEpochs      = 45                  # Number of training epochs. Set to 0 to skip DNN refinement
rf_maxTrainImages = float('inf')        # Naximum number of training images per epoch. Set to float('inf') to use all images
rf_lrPerMb        = [0.01] * 20 + [0.001] * 20 + [0.0001]  # Learning rate schedule
rf_momentumPerMb  = 0.9                 # Momentum during gradient descent
rf_l2RegWeight    = 0.0005              # L2 regularizer weight during gradient descent
rf_boFreezeWeights      = False         # Set to 'True' to freeze all but the very last layer. Otherwise the full network is refined
rf_boBalanceTrainingSet = False         # Set to 'True' to duplicate images such that all labels have the same number of images

# Running the DNN model (script: 3_runDNN.py and 7_activeLearning_step1.py)
svm_boL2Normalize = True # Normalize 512-floats vector to be of unit length before SVM training
run_mbSize = 64          # Minibatch size when running the model. Higher values will run faster, but might model might not fit into memory

# SVM training params (script: 4_trainSVM.py)
svm_CVals = [10**-4, 10**-3, 10**-2, 0.1, 1, 10, 100] # Slack penality parameter C to try during SVM training

# Root directory where all data and temporary files are saved and loaded from. This is different from the project directory.
rootDir = os.path.expanduser('~') + "/Desktop/imgClassificationUsingCntk_data/"  \


###################
# Fixed parameters
# (do not modify)
###################
print("PARAMETERS: datasetName = " + datasetName)

# Directories
imgOrigDir      = pathJoin(rootDir,    "images",  datasetName + "/")
resourcesDir    = pathJoin(rootDir,    "resources/")
procDir         = pathJoin(rootDir,    "proc",    datasetName + "/")
resultsDir      = pathJoin(rootDir,    "results", datasetName + "/")
workingDir      = pathJoin(rootDir,    "tmp/")

# Files
dedicatedTestSplitPath  = pathJoin(imgOrigDir, "dedicatedTestImages.tsv")
imgUrlsPath             = "resources/fashionTextureUrls.tsv"
imgInfosTrainPath       = pathJoin(procDir, "imgInfosTrain.pickle")
imgInfosTestPath        = pathJoin(procDir, "imgInfosTest.pickle")
imgDictTrainPath        = pathJoin(procDir, "imgDictTrain.pickle")
imgDictTestPath         = pathJoin(procDir, "imgDictTest.pickle")
lutLabel2IdPath         = pathJoin(procDir, "lutLabel2Id.pickle")
lutId2LabelPath         = pathJoin(procDir, "lutId2Label.pickle")
if classifier == "svm":
   cntkRefinedModelPath = pathJoin(procDir, "cntk_fixed.model")
else:
   cntkRefinedModelPath = pathJoin(procDir,      "cntk_refined.model")
cntkTestMapPath         = pathJoin(workingDir,   "test_map.txt")
cntkTrainMapPath        = pathJoin(workingDir,   "train_map.txt")
cntkPretrainedModelPath = pathJoin(resourcesDir, rf_pretrainedModelFilename)
dnnOutputPath           = pathJoin(procDir,      "features_" + classifier + ".pickle")
svmPath                 = pathJoin(procDir,      classifier + ".np")

# Pretrained DNNs
urlPretrainedDNNs = {"resnet_18.model": "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet18_ImageNet_CNTK.model",
                     "resnet_34.model": "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet34_ImageNet_CNTK.model",
                     "resnet_50.model": "https://www.cntk.ai/Models/CNTK_Pretrained/ResNet50_ImageNet_CNTK.model"}

# Dimension of the DNN output, for "ResNet_18.model" this is 512 if using a SVM as classifier,
# otherwise the DNN output dimension equals the number of classes
assert(classifier in ['svm', 'dnn', 'svmDnnRefined'])
if os.path.exists(imgOrigDir):
    if classifier.startswith('dnn'):
        rf_modelOutputDimension = len(getDirectoriesInDirectory(imgOrigDir))
    elif rf_pretrainedModelFilename.lower() == "resnet_18.model" or rf_pretrainedModelFilename.lower() == "resnet_34.model":
        rf_modelOutputDimension = 512
    elif rf_pretrainedModelFilename.lower() == "resnet_50.model":
        rf_modelOutputDimension = 2048
    else:
        raise Exception("Model featurization dimension not specified.")