import sys, io, json, base64, datetime as dt
sys.path.append("tmp")
import matplotlib
matplotlib.use('Agg') #not sure if this include / 'Agg' is necessary
import cntk
from helpers_cntk import *


####################################
# Parameters
####################################
classifier   = 'svm' #must match the option used for model training
imgPath      = "uploadedImg.jpg" #"12.jpg"
resourcesDir = "tmp"

# Do not change
run_mbSize = 1
svm_boL2Normalize = True
if classifier == "svm":
   cntkRefinedModelPath = pathJoin(resourcesDir, "cntk_fixed.model")
else:
   cntkRefinedModelPath = pathJoin(resourcesDir, "cntk_refined.model")
workingDir           = pathJoin(resourcesDir, "tmp/")
svmPath              = pathJoin(resourcesDir, classifier + ".np")  #only used if classifier is set to 'svm'
lutId2LabelPath      = pathJoin(resourcesDir, "lutId2Label.pickle")



################
# API run() and
# init() methods
################
# API call entry point
def run(input_df):
    try:
        print("Python version: " + str(sys.version) + ", CNTK version: " + cntk.__version__)

        startTime = dt.datetime.now()
        print(str(input_df))

        # convert input back to image and save to disk
        base64ImgString = input_df['image base64 string'][0]
        print(base64ImgString)
        pil_img = base64ToPilImg(base64ImgString)
        print("pil_img.size: " + str(pil_img.size))
        pil_img.save(imgPath, "JPEG")
        print("Save pil_img to: " + imgPath)

        # Load model <---------- SHOULD BE DONE JUST ONCE
        print("Classifier = " + classifier)
        makeDirectory(workingDir)
        if not os.path.exists(cntkRefinedModelPath):
            raise Exception("Model file {} does not exist, likely because the {} classifier has not been trained yet.".format(cntkRefinedModelPath, classifier))
        model = load_model(cntkRefinedModelPath)
        lutId2Label = readPickle(lutId2LabelPath)

        # Run DNN
        printDeviceType()
        node = getModelNode(classifier)
        mapPath = pathJoin(workingDir, "rundnn_map.txt")
        dnnOutput = runCntkModelImagePaths(model, [imgPath], mapPath, node, run_mbSize)

        # Predicted labels and scores
        scoresMatrix = runClassifierOnImagePaths(classifier, dnnOutput, svmPath, svm_boL2Normalize)
        scores = scoresMatrix[0]
        predScore = np.max(scores)
        predLabel = lutId2Label[np.argmax(scores)]
        print("Image predicted to be '{}' with score {}.".format(predLabel, predScore))

        # Create json-encoded string of the model output
        executionTimeMs = (dt.datetime.now() - startTime).microseconds / 1000
        outDict = {"label": str(predLabel), "score": str(predScore), "allScores": str(scores),
                   "Id2Labels": str(lutId2Label), "executionTimeMs": str(executionTimeMs)}
        outJsonString = json.dumps(outDict)
        print("Json-encoded detections: " + outJsonString[:120] + "...")
        print("DONE.")

        return(str(outJsonString))

    except Exception as e:
        return(str(e))

# API initialization method
def init():
    try:
        print("Executing init() method...")
        print("Python version: " + str(sys.version) + ", CNTK version: " + cntk.__version__)
    except Exception as e:
        print("Exception in init:")
        print(str(e))


################
# Main
################
def main():
    from azureml.api.schema.dataTypes import DataTypes
    from azureml.api.schema.sampleDefinition import SampleDefinition
    from azureml.api.realtime.services import generate_schema
    import pandas

    # Create random 5x5 pixels image to use as sample input
    #base64ImgString = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAIAAAACDbGyAAAAFElEQVR4nGP8//8/AxJgYkAFpPIB6vYDBxf2tWQAAAAASUVORK5CYII="
    #pilImg = pilImread("C:/Users/pabuehle/Desktop/vienna/iris4/tiny.jpg")
    pilImg = Image.fromarray((np.random.rand(5, 5, 3) * 255).astype('uint8')) #.convert('RGB')
    base64ImgString = pilImgToBase64(pilImg)  #random 5x5 pixels image

    # Call init() and run() function
    init()
    df = pandas.DataFrame(data=[[base64ImgString]], columns=['image base64 string'])
    inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}
    resultString = run(df)
    print("resultString = " + str(resultString))

    # Genereate the schema
    generate_schema(run_func=run, inputs=inputs, filepath='service_schema.json')
    print("Schema generated.")

if __name__ == "__main__":
    main()



























# --- LOCAL DEPLOYMENT ---
# See links:
#  - https://github.com/Azure/ViennaDocs/blob/master/Documentation/tutorial-classifying-iris.md
# Steps (in this order):
# - az login
# - az account list -o table
# - az account set -s 0ca618d2-22a8-413a-96d0-0f1b531129c3     <--- Boston DS Dev, I whitelisted this.
# - az ml env setup -n pabuehleviennaenv2 -l eastus2
# - az ml env set -g pabuehleviennaenv2rg -n pabuehleviennaenv2
# - az ml env local
# - az ml service create realtime -f deploymain.py -s deployserviceschema.json -n imapp3 -v -r python -c conda_dependencies_my.yml  -d helpers.py -d helpers_cntk.py -d pabuehle_utilities_CVbasic_v2.py -d pabuehle_utilities_general_v2.py -d svm.np -d lutId2Label.pickle --model-file cntk_fixed.model
#           Also works if replacing "--model-file" with simply "-d"
# - az ml service run realtime -i imapp -d "{\"input_df\": [{\"image base64 string\": \"iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAIAAAACDbGyAAAAFElEQVR4nGP8//8/AxJgYkAFpPIB6vYDBxf2tWQAAAAASUVORK5CYII=\"}]}"


# --- CLUSTER DEPLOYMENT (OLD, USED TO WORK) ---
#   See here how to set up deployment target: https://github.com/Azure/ViennaDocs/blob/master/Documentation/Tutorial.md
#   Step 1 - test locally: az ml env local
#   Step 2 - set up cluster: az ml env setup --cluster -n pabuehleacs1
#                if error "No cluster available in your environment" then run: az ml env setup -c -n pabuehleacs1  (where the -c flag indicates that we want to create a cluster, and the name can be the same as the one before)
# az ml env cluster

# --- OTHER COMMANDS ---
# - Inspect docker message / deployment error:
#      docker ps -a
#      docker logs <containerid>
# - List all deployed services: az ml service list realtime
# - See more info for a specific deployed service: az ml service usage realtime --id imapp1


# ---- ACCOUT OUTPUTS ------
# C:\Users\pabuehle\Desktop\vienna\iris4>az account show
# {
#   "environmentName": "AzureCloud",
#   "id": "0ca618d2-22a8-413a-96d0-0f1b531129c3",
#   "isDefault": true,
#   "name": "Boston DS Dev",
#   "state": "Enabled",
#   "tenantId": "72f988bf-86f1-41af-91ab-2d7cd011db47",
#   "user": {
#     "name": "pabuehle@microsoft.com",
#     "type": "user"
#   }
# }
#
# C:\Users\pabuehle\Desktop\vienna\iris4>az ml env setup -n pabuehleviennaenv2 -l eastus2
# Subscription set to Boston DS Dev
# Continue with this subscription (Y/n)? Y
# Creating resource group pabuehleviennaenv2rg
# Provisioning compute resources...
# Resource creation submitted successfully.
# To see more information for your environment, run:
#   az ml env show -g pabuehleviennaenv2rg -n pabuehleviennaenv2
# You can set the new environment as your target context using:
#   az ml env set -g pabuehleviennaenv2rg -n pabuehleviennaenv2
#
# C:\Users\pabuehle\Desktop\vienna\iris4>az ml env show -g pabuehleviennaenv2rg -n pabuehleviennaenv2
# {
#   "Cluster Name": "pabuehleviennaenv2",
#   "Cluster Size": "N/A",
#   "Created On": "2017-09-15T14:18:27.378Z",
#   "Location": "eastus2",
#   "Provisioning State": "Creating",
#   "Resource Group": "pabuehleviennaenv2rg",
#   "Subscription": "0ca618d2-22a8-413a-96d0-0f1b531129c3"
# }
#
# C:\Users\pabuehle\Desktop\vienna\iris4>az ml env set -g pabuehleviennaenv2rg -n pabuehleviennaenv2
# Resource with group pabuehleviennaenv2rg and name pabuehleviennaenv2 cannot be set, as its provisioning state is Creating. Provisioning state succeeded is required.
#
# C:\Users\pabuehle\Desktop\vienna\iris4>az ml env set -g pabuehleviennaenv2rg -n pabuehleviennaenv2
# Compute set to pabuehleviennaenv2.

