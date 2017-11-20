# -*- coding: utf-8 -*-
import sys, os
sys.path.append(".")
sys.path.append("..")
sys.path.append("libraries")
sys.path.append("../libraries")
from helpers import *
from PARAMETERS import *
#locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameter
####################################
downloadTopNImages = sys.maxsize  #set to e.g. 50 to only download the first 50 of the 428 images
maxSize = 1000


####################################
# Main
####################################
makeDirectory(rootDir)
makeDirectory(imgOrigDir)
print("Directory used to read and write model/image files: " + rootDir)
amlLogger = getAmlLogger()
if amlLogger != []:
    amlLogger.log("amlrealworld.ImageClassificationUsingCntk.0_downloadData", "true")

# Read image urls
if os.path.exists(imgUrlsPath):
    imgUrls = readTable(imgUrlsPath)
else:
    imgUrls = readTable("../" + imgUrlsPath)
imgUrls = randomizeList(imgUrls)

# Download provided fashion images
counter = 0
for index, (label, url) in enumerate(imgUrls):
    # Skip image if was already downloaded
    outImgPath = pathJoin(imgOrigDir, label, str(index) + ".jpg")
    if pathExists(outImgPath):
        counter += 1
        continue

    # Download image
    print("Downloading image {} of {}: label={}, url={}".format(index, len(imgUrls), label, url))
    data = downloadFromUrl(url)
    if len(data) > 0:
        makeDirectory(pathJoin(imgOrigDir, label))
        writeBinaryFile(outImgPath, data)

        # Sanity check: delete image if it is corrupted
        # Otherwise, resize if above given pixel width/height
        try:
            img = imread(outImgPath)
            if max(imWidthHeight(img)) > maxSize:
                img, _ = imresizeMaxDim(img, maxSize)
                imwrite(img, outImgPath)
            counter += 1
        except:
            print("Removing corrupted image {}, url={}".format(outImgPath, url))
            os.remove(outImgPath)
print("Successfully downloaded {} of the {} image urls.".format(counter, len(imgUrls)))
print("DONE.")
