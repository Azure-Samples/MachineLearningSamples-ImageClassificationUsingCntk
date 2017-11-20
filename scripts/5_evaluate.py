# -*- coding: utf-8 -*-
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("libraries")
sys.path.append("../libraries")
from helpers import *
from PARAMETERS import *


####################################
# Parameters
####################################
boEvalOnTrainingSet = False  # Set to 'False' to evaluate on test set; 'True' to instead eval on training set


####################################
# Main
####################################
makeDirectory("outputs")
print("Classifier = " + classifier)
print("Directory used to read and write model/image files: " + rootDir)
amlLogger = getAmlLogger()
if amlLogger != []:
    amlLogger.log("amlrealworld.ImageClassificationUsingCntk.5_evaluate", "true")

# Load data
print("Loading data...")
dnnOutput   = readPickle(dnnOutputPath)
lutLabel2Id = readPickle(lutLabel2IdPath)
lutId2Label = readPickle(lutId2LabelPath)
if not boEvalOnTrainingSet:
    imgDict = readPickle(imgDictTestPath)
else:
    print("WARNING: evaluating on training set.")
    imgDict = readPickle(imgDictTrainPath)

# Predicted labels and scores
scoresMatrix, imgFilenames, gtLabels = runClassifier(classifier, dnnOutput, imgDict,  lutLabel2Id, svmPath, svm_boL2Normalize)
predScores = [np.max(scores)    for scores in scoresMatrix]
predLabels = [np.argmax(scores) for scores in scoresMatrix]
writePickle(pathJoin(procDir, "scoresMatrix.pickle"), scoresMatrix)
writePickle(pathJoin(procDir, "predLabels.pickle"),   predLabels)
writePickle(pathJoin(procDir, "gtLabels.pickle"),   gtLabels)
writePickle(pathJoin(procDir, "boEvalOnTrainingSet.pickle"), boEvalOnTrainingSet)

# Plot ROC curve
classes = [lutId2Label[i] for i in range(len(lutId2Label))]
fig = plt.figure(figsize=(14,6))
plt.subplot(121)
rocComputePlotCurves(gtLabels, scoresMatrix, classes)

# Plot confusion matrix
# Note: Let C be the confusion matrix. Then C_{i, j} is the number of observations known to be in group i but predicted to be in group j.
plt.subplot(122)
confMatrix = metrics.confusion_matrix(gtLabels, predLabels)

cmPlot(confMatrix, classes, normalize=False)
plt.show()
fig.savefig('outputs/rocCurve_confMat.jpg', bbox_inches='tight', dpi = 200)

# Print accuracy to console
globalAcc, classAccs = cmPrintAccuracies(confMatrix, classes, gtLabels)
if amlLogger != []:
    amlLogger.log("clasifier", classifier)
    amlLogger.log("Global accuracy", 100 * globalAcc)
    amlLogger.log("Class-average accuracy", 100 * np.mean(classAccs))
    for className, acc in zip(classes,classAccs):
        amlLogger.log("Accuracy of class %s" % className, 100 * np.mean(classAccs))

# Visualize results
for counter, (gtLabel, imgFilename, predScore, predLabel) in enumerate(zip(gtLabels, imgFilenames, predScores, predLabels)):
    if counter > 5:
        break
    if predLabel == gtLabel:
        drawColor = (0, 255, 0)
    else:
        drawColor = (0, 0, 255)
    img = imread(pathJoin(imgOrigDir, lutId2Label[gtLabel], imgFilename))
    img = imresizeToSize(img, targetWidth = 800)
    cv2.putText(img, "{} with score {:2.2f}".format(lutId2Label[predLabel], predScore), (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, drawColor, 2)
    drawCircle(img, (50, 50), 40, drawColor, -1)
    imshow(img, maxDim = 800, waitDuration=100)
    #imwrite(imresizeMaxDim(img,800)[0], "outputs/result_img{}.jpg".format(counter))
print("DONE.")

































#
# # ROC curve <-- wrong!
# #fpr, tpr, thresholds = metrics.roc_curve(gtLabels, predScores, pos_label = lutLabel2Id['positive'])
# #plt.figure()
# #lw = 2
# #plt.plot(fpr, tpr, color='darkorange', lw=lw) #, label='ROC curve (area = %0.2f)' % roc_auc[2])
# #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# #plt.xlim([0.0, 1.0])
# #plt.ylim([0.0, 1.05])
# #plt.title('ROC curve')
# #plt.ylabel('True Positive Rate (Sensitivity / Recall)')
# #plt.xlabel('False Positive Rate (1-Specificity)')
# #plt.legend(loc="lower right")
# #plt.show()
# #for fp, tp, thres in list(zip(fpr, tpr, thresholds))[::10]:
# #     #if fp > 0.95:
# #     print(tp, fp, thres)
#
#
# # For different threshold, output precision and recall
# thresholds = sorted(predScores)[::10] #sorted(scoresMatrix.flatten())[::30]
# #thresholds = thresholds[::int(len(thresholds) / 1000)]
# thresholds.insert(0, np.min(thresholds) - 0.01)
# thresholds.append(np.max(thresholds) + 0.01)
# eps = 0.00001
# posLabel = 0 #lutLabel2Id['positive']
#
#
# recallVec = []
# precisionVec = []
# fpRateVec = []
# for threshold in thresholds:
#     predLabelsThres = []
#
#     scores = getColumn(scoresMatrix,posLabel)
#     score = max(scores)
#     for score in scores:
#         if score <= threshold:
#             predLabelsThres.append(1)
#         else:
#             predLabelsThres.append(0)
#
#     #p  = len(np.where(np.array(gtLabels) == posLabel)[0])
#     tp = len(np.where((np.array(gtLabels) == posLabel) & (np.array(predLabelsThres) == posLabel))[0])
#     fp = len(np.where((np.array(gtLabels) != posLabel) & (np.array(predLabelsThres) == posLabel))[0])
#     fn = len(np.where((np.array(gtLabels) == posLabel) & (np.array(predLabelsThres) != posLabel))[0])
#     tn = len(np.where((np.array(gtLabels) != posLabel) & (np.array(predLabelsThres) != posLabel))[0])
#     if (tp + fp) > 0:
#         precision = 1.0 * tp / (tp + fp + eps)
#         recall    = 1.0 * tp / (tp + fn + eps)
#         fpRate    = 1.0 * fp / (fp + tn + eps)
#         precisionVec.append(precision)
#         recallVec.append(recall)
#         fpRateVec.append(fpRate)
#
#         #confMatrix = confusion_matrix(gtLabels, predLabelsThres)
#         #precision = precision_score(gtLabels, predLabelsThres, pos_label=0) # average=None)
#         #recall    = recall_score(   gtLabels, predLabelsThres, pos_label=0) #average=None)
#         print("Threshold {:2.2f}: precision={:2.2f}, recall={:2.2f}, fpRate = {:2.2f}".format(threshold, precision, recall, fpRate))
#         print("   tp={}, fp = {}, fn = {}".format(tp, fp, fn))
# #aucValue = auc(recallVec, precisionVec)
#
#
# # Compute confustion matrix and precision recall curve
# confMatrix = confusion_matrix(gtLabels, predLabels)
# classes = [lutId2Label[i] for i in range(len(lutId2Label))]
#
# #(precisionVec, recallVec, aucValue) = prComputeCurves(gtLabels, scoresMatrix)
# #(precisionVec, recallVec, aucValue) = prComputeCurves([abs(i-1) for i in gtLabels], -scoresMatrix) #, posLabel = lutLabel2Id['snowLeopard'])
# globalAcc, classAccs = cmPrintAccuracies(confMatrix, classes)
#
# if run_logger != []:
#     run_logger.log("Class-averaged accuracy", np.mean(classAccs))
#
# # Plot
# fig = plt.figure(figsize=(20,6))
# plt.subplot(131)
# prPlotCurves(precisionVec, recallVec) #, auc(recallVec, precisionVec))
# plt.subplot(132)
#
# rocPlotCurves(fpRateVec, recallVec)
# plt.subplot(133)
# cmPlot(confMatrix, classes=classes, normalize=False) #True)
# plt.draw()
# plt.show()
# fig.savefig('outputs/rovCurve.jpg', dpi=200)
#
#
# # Visualize results
# for counter, (gtLabel, imgFilename, predScore, predLabel) in enumerate(zip(gtLabels, imgFilenames, predScores, predLabels)):
#     if counter > 5:
#         break
#     if predLabel == gtLabel:
#         drawColor = (0, 255, 0)
#     else:
#         drawColor = (0, 0, 255)
#     img = imread(pathJoin(imgOrigDir, lutId2Label[gtLabel], imgFilename))
#     cv2.putText(img, "{} with score {:2.2f}".format(lutId2Label[predLabel], predScore), (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, drawColor, 2)
#     drawCircle(img, (50, 50), 40, drawColor, -1)
#     #imshow(img, maxDim = 800, waitDuration=500)
#     imwrite(imresizeMaxDim(img,800)[0], "outputs/result_img{}.jpg".format(counter))
# print("DONE.")
