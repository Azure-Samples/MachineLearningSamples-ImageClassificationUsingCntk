import sys, json, time
sys.path.append("libraries")
sys.path.append("../libraries")
from helpers import *


####################################
# Parameters
####################################
imgPath = "./resources/testImg.jpg"

# --- Edit and uncomment when calling a locally-deployed Rest API ---
#cluster_scoring_url = "http://127.0.0.1:32773/score."
#service_key         = None  # Set to None if it is local deployment

# --- Edit and uncomment when calling a cloud-deployed Rest API ---
#cluster_scoring_url = "http://12.34.567.89:80/api/v1/service/imgclassapi1/score"
#service_key         = "abcdefghijklmnopqrstuvw123456789"


####################################
# Main
####################################
amlLogger = getAmlLogger()
if amlLogger != []:
    amlLogger.log("amlrealworld.ImageClassificationUsingCntk.6_callWebservice", "true")

# Check if scoring url and service key are defined
try:
    cluster_scoring_url, service_key
except:
    print("ERROR: need to set 'cluster_scoring_url' and 'service_key' variables.")
    exit()

# Compile web service input
base64Img = pilImgToBase64(pilImread(imgPath))
headers = {'Content-Type': 'application/json'}
if service_key is not None and service_key is not []:
    headers['Authorization'] = 'Bearer ' + service_key
data = '{"input_df": [{"image base64 string": "' + base64Img + '"}]}'

# Repeat web-service call 5 times
print("Calling webservice at URL {}".format(cluster_scoring_url))
for i in range(5):
    startTime = time.time()
    res = requests.post(cluster_scoring_url, headers=headers, data=data)

    try:
        resDict = json.loads(res.json())
        apiDuration   = int(float(resDict['executionTimeMs']))
        localDuration = int(float(1000.0*(time.time() - startTime)))
        print("Webservice call took {:5} ms, pure computation time: {:5} ms, overhead (difference) = {:4} ms.".format(localDuration, apiDuration, (localDuration - apiDuration)))
        if i == 0:
            print("Image classified as '{}' with confidence score {}.".format(json.loads(res.json())["label"],
                                                                              json.loads(res.json())["score"]))
            print("Full web-service output:" + json.dumps(res.json(), indent=4))
    except:
        print("ERROR: webservice returned message " + res.text)
print("DONE")