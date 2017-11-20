# Image Classification using Microsoft Cognitive Toolkit (CNTK)


## Link to the Microsoft DOCS site

The detailed documentation for this image classification example includes the step-by-step walk-through:
[https://docs.microsoft.com/azure/machine-learning/preview/scenario-image-classification-using-cntk](https://docs.microsoft.com/azure/machine-learning/preview/scenario-image-classification-using-cntk)

## Link to the Gallery GitHub repository

The public GitHub repository for this image classification example contains all the code samples:
[https://github.com/azure/MachineLearningSamples-ImageClassificationUsingCntk](https://github.com/azure/MachineLearningSamples-ImageClassificationUsingCntk)

## Overview

A large number of problems in the computer vision domain can be solved using image classification approaches.
These include building models which answer questions such as, "Is an OBJECT present in the image?" (where OBJECT could for example be "dog", "car", "ship", etc.) as well as more complex questions, like "What class of eye disease severity is evinced by this patient's retinal scan?"

This tutorial will address solving such problems. We will show how to train, evaluate and deploy your own image classification model using the  [Microsoft Cognitive Toolkit (CNTK) ](https://www.microsoft.com/en-us/cognitive-toolkit/) for deep learning.
Example images are provided, but the reader can also bring their own dataset and train their own custom models.

The key steps required to deliver this solution are as follows:

1. Generate an annotated image dataset. Alternatively, the provided demo dataset can be used.
2. Train an image classifier using a pre-trained Deep Neural Network.
3. Evaluate and improve accuracy of this model.
4. Deploy the model as a REST API, either to the local machine or to the cloud.

## Key components needed to run this example

1. An [Azure account](https://azure.microsoft.com/free/) (free trials are available).
2. An installed copy of Azure Machine Learning Workbench with a workspace created.
3. A machine or VM running Windows.
4. A dedicated GPU is recommended, however not required.

## Data/Telemetry
This sample "Image Classification using CNTK" collects usage data and sends it to Microsoft to help improve our products and services. Read our [privacy statement](http://go.microsoft.com/fwlink/?LinkId=521839) to learn more.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
