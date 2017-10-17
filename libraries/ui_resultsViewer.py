import os
import numpy as np
from ipywidgets import widgets, Layout, IntSlider
import bqplot
from bqplot import pyplot as bqPyplot
from IPython.display import display
from utilities_general_v2 import softmax
from helpers import *


####################################
# ResultsUI class
####################################
# Init
class ResultsUI(object):
    # Init object and define instance variables
    def __init__(self, imgOrigDir, imgDict, predScores, predLabels, lutId2Label, boSoftmax=False):
        self.imgOrigDir = imgOrigDir
        self.dataset = create_dataset(imgDict)
        self.predScores = predScores
        self.predLabels = predLabels
        self.lutId2Label = lutId2Label
        assert (predScores.shape[1] == len(self.dataset.labels))
        assert (len(predLabels) == predScores.shape[0] == len(self.dataset.images))
        if boSoftmax:
            self.predScores = np.array([softmax(vec) for vec in self.predScores])

        # Init
        self.visImageIndex = 0
        self.labels = [lutId2Label[key] for key in sorted(lutId2Label.keys())]

        # Create UI
        self.ui = self.createUI()

    # Update / redraw all UI elements
    def updateUI(self):
        predLabel = self.predLabels[self.visImageIndex]
        imgObj = self.dataset.images[self.visImageIndex]
        scores = self.predScores[self.visImageIndex]
        self.wImageHeader.value = "Image index: {}".format(self.visImageIndex)
        self.wImg.value = wImread(imgObj, self.imgOrigDir)
        self.wGtLabel.value = imgObj.label
        self.wPredLabel.value = str(self.lutId2Label[predLabel])
        self.wPredScore.value = str(self.predScores[self.visImageIndex, predLabel])
        self.wIndex.value = str(self.visImageIndex)
        self.wFilename.value = imgObj.filename
        bqPyplot.clear()
        bqPyplot.bar(self.labels, scores, align='center', alpha=1.0, color=np.abs(scores),
                     scales={'color': bqplot.ColorScale(scheme='Blues', min=0)})

    # Create all UI elements
    def createUI(self):

        # ------------
        # Callbacks + logic
        # ------------
        # Return if image should be shown
        def image_passes_filters(imageIndex):
            boPredCorrect = self.dataset.images[imageIndex].label == self.lutId2Label[self.predLabels[imageIndex]]
            if (boPredCorrect and self.wFilterCorrect.value) or (not boPredCorrect and self.wFilterWrong.value):
                return True
            return False

        # Next / previous image button callback
        def button_pressed(obj):
            step = int(obj.value)
            self.visImageIndex += step
            self.visImageIndex = min(max(0, self.visImageIndex), int(len(self.predLabels)) - 1)
            while not image_passes_filters(self.visImageIndex):
                self.visImageIndex += step
                if self.visImageIndex <= 0 or self.visImageIndex >= int(len(self.predLabels)) - 1:
                    break
            self.visImageIndex = min(max(0, self.visImageIndex), int(len(self.predLabels)) - 1)
            self.wImageSlider.value = self.visImageIndex
            self.updateUI()

        # Image slider callback. Need to wrap in try statement to avoid errors when slider value is not a number.
        def slider_changed(obj):
            try:
                self.visImageIndex = int(obj['new']['value'])
                self.updateUI()
            except Exception as e:
                pass

        # ------------
        # UI - image + controls (left side)
        # ------------
        wNextImageButton = widgets.Button(description="Image +1")
        wNextImageButton.value = "1"
        wNextImageButton.layout = Layout(width='80px')
        wNextImageButton.on_click(button_pressed)
        wPreviousImageButton = widgets.Button(description="Image -1")
        wPreviousImageButton.value = "-1"
        wPreviousImageButton.layout = Layout(width='80px')
        wPreviousImageButton.on_click(button_pressed)

        self.wImageSlider = IntSlider(min=0, max=len(self.predLabels) - 1, step=1,
                                      value=self.visImageIndex, continuous_update=False)
        self.wImageSlider.observe(slider_changed)
        self.wImageHeader = widgets.Text("", layout=Layout(width="130px"))
        self.wImg = widgets.Image()
        imgWidth = 400
        self.wImg.layout.width = str(imgWidth) + 'px'  # '500px'
        wImageWithHeader = widgets.VBox(
            children=[widgets.HBox(children=[wPreviousImageButton, wNextImageButton, self.wImageSlider]),
                      self.wImg], width=imgWidth + 20)

        # ------------
        # UI - info (right side)
        # ------------
        wFilterHeader = widgets.HTML(value="Filters (use Image +1/-1 buttons for navigation):")
        self.wFilterCorrect = widgets.Checkbox(value=True, description='Correct classifications')
        self.wFilterWrong = widgets.Checkbox(value=True, description='Incorrect classifications')

        wGtHeader = widgets.HTML(value="Ground truth:")
        self.wGtLabel = widgets.Text(value="", description="Label:")

        wPredHeader = widgets.HTML(value="Prediction:")
        self.wPredLabel = widgets.Text(value="", description="Label:")
        self.wPredScore = widgets.Text(value="", description="Score:")

        wInfoHeader = widgets.HTML(value="Image info:")
        self.wIndex = widgets.Text(value="", description="Index:")
        self.wFilename = widgets.Text(value="", description="Name:")

        wScoresHeader = widgets.HTML(value="Classification scores:")
        self.wScores = bqPyplot.figure()
        self.wScores.layout.height = '250px'
        self.wScores.layout.width = '370px'
        self.wScores.fig_margin = {"top": 5, "bottom": 80, "left": 30, "right": 5}

        # Combine UIs into tab widget
        wInfoHBox = widgets.VBox(children=[wFilterHeader, self.wFilterCorrect, self.wFilterWrong, wGtHeader,
                                           self.wGtLabel, wPredHeader, self.wPredLabel, self.wPredScore,
                                           wInfoHeader, self.wIndex, self.wFilename, wScoresHeader,
                                           self.wScores])
        wInfoHBox.layout.padding = '20px'
        visTabsUI = widgets.Tab(children=[widgets.HBox(children=[wImageWithHeader, wInfoHBox])])
        visTabsUI.set_title(0, 'Results viewer')

        # Fill UI with content
        self.updateUI()

        return (visTabsUI)
