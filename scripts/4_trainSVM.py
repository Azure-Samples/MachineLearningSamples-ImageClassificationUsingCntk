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
amlLogger = getAmlLogger()
if amlLogger != []:
    amlLogger.log("amlrealworld.ImageClassificationUsingCntk.4_trainSVM", "true")

if not classifier.startswith('svm'):
    print("No need to train SVM since using the DNN directly as classifier.")
    
else:
    print("Directory used to read and write model/image files: " + rootDir)

    # Load training datasta
    print("Load data...")
    lutLabel2Id  = readPickle(lutLabel2IdPath)
    imgDictTest  = readPickle(imgDictTestPath)
    imgDictTrain = readPickle(imgDictTrainPath)
    dnnOutput  = readPickle(dnnOutputPath)

    # Prepare SVM inputs for training and testing
    feats_test,  labels_test,  _ = getSvmInput(imgDictTest,  dnnOutput, svm_boL2Normalize, lutLabel2Id)
    feats_train, labels_train, _ = getSvmInput(imgDictTrain, dnnOutput, svm_boL2Normalize, lutLabel2Id)
    printFeatLabelInfo("Statistics training data:", feats_train, labels_train)
    printFeatLabelInfo("Statistics test data:",     feats_test,  labels_test)

    # Train SVMs for different values of C, and keep the best result
    bestAcc = float('-inf')
    testAccs = []
    for svm_CVal in svm_CVals:
        print("Start SVM training  with C = {}..".format(svm_CVal))
        tstart = datetime.datetime.now()
        #feats_train = sparse.csr_matrix(feats_train) #use this to avoid memory problems
        learner = svm.LinearSVC(C=svm_CVal, class_weight='balanced', verbose=0)
        learner.fit(feats_train, labels_train)
        print("   Training time [labels_train]: {}".format((datetime.datetime.now() - tstart).total_seconds() * 1000))
        print("   Training accuracy    = {:3.2f}%".format(100 * np.mean(sklearnAccuracy(learner, feats_train, labels_train))))
        testAcc = np.mean(sklearnAccuracy(learner, feats_test,  labels_test))
        print("   Test accuracy        = {:3.2f}%".format(100 * np.mean(testAcc)))

        testAccs.append(testAcc)

        # Store best model. Note that this should use a separate validation set, and not the test set.
        if testAcc > bestAcc:
            print("   ** Updating best model. **")
            bestC = svm_CVal
            bestAcc = testAcc
            bestLearner = learner

    print("Best model has test accuracy {:2.2f}%, at C = {}".format(100 * bestAcc, bestC))
    writePickle(svmPath, bestLearner)
    print("Wrote svm to: " + svmPath + "\n")

    # Log accuracy and regularizer value using Azure ML
    if amlLogger != []:
        amlLogger.log("Regularization Rate", svm_CVals)
        amlLogger.log("Accuracy", testAccs)
print("DONE. ")