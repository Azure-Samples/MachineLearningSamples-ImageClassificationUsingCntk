import os
from ipywidgets import widgets, Layout, IntSlider
import shutil
from helpers import *

# ------------------------------------------------
# Helper functions
# ------------------------------------------------
def listSplit(list1D, n, method):
    if method.lower() == 'fillFirst'.lower():
        list2D = [list1D[i:i + n] for i in range(0, len(list1D), n)]
    else:
        raise Exception('Unknown list split method')
    return list2D



# ------------------------------------------------
# Class - Image annotation UI
# -------------------------------------------------
class AnnotationUI(object):

    # Init object and define instance variables
    def __init__(self, imgOrigDir, imgDict, lutLabel2Id, imgIndices=[], gridSize=(3, 2), wZoomImgWidth = 500):
        self.imgOrigDir    = imgOrigDir
        self.dataset       = create_dataset(imgDict)
        self.lutLabel2Id   = lutLabel2Id
        self.gridSize      = gridSize
        self.wZoomImgWidth = wZoomImgWidth

        # Set images to be shown (in that order)
        if imgIndices == []:
            imgIndices = list(range(len(self.dataset.images)))
            #random.shuffle(imgIndices)

        # Set labels and labelName->labelId mappings
        #self.labels = ["UNLABELED"] + sorted([l.name for l in dataset.labels]) + ["EXCLUDE"]
        self.labels = sorted(lutLabel2Id.keys())  # [l.name for l in self.dataset.labels]

        # Initialize what images are on what image page
        # (page == grid of images on the right side of the UI)
        self.pageIndex = 0
        self.pageImgIndices = listSplit(imgIndices, gridSize[0] * gridSize[1], method='fillFirst')

        # Create UI
        self.ui = self.createUI()

    # Update / redraw the zoom UI elements
    def updateZoomUI(self, imgObj):
        self.wZoomImg.value      = wImread(imgObj, self.imgOrigDir)
        self.wZoomHeader.value   = "Image id: {}".format(imgObj.idVal)
        self.wZoomTextArea.value = str(imgObj).replace(', ', '\n')
        self.wPageSlider.value   = str(self.pageIndex)


    # Update / redraw all UI elements
    def updateUI(self):
        self.boUpdatingUI = True # indicate code is in updating-UI state

        # Update image grid UI
        imgIndices = self.pageImgIndices[self.pageIndex]
        for i in range(self.gridSize[0] * self.gridSize[1]):
            wImg    = self.wImgs[i]
            wLabel  = self.wLabels[i]
            wButton = self.wButtons[i]

            if i < len(imgIndices):
                imgIndex = imgIndices[i]
                imgObj = self.dataset.images[imgIndex]
                wImg.layout.visibility    = 'visible'
                wButton.layout.visibility = 'visible'
                wLabel.layout.visibility  = 'visible'
                wImg.value = wImread(imgObj, self.imgOrigDir)
                wImg.description = str(imgIndex)
                wLabel.value = imgObj.label
                #wLabel.text = str(imgIndex)  # this property is ignored and not accessible later in code
                wLabel.description  = "Image " + str(imgIndex)
                wButton.description = "Zoom"
                wButton.value = str(imgIndex)
            else:
                wImg.layout.visibility    = 'hidden'
                wButton.layout.visibility = 'hidden'
                wLabel.layout.visibility  = 'hidden'

        # Update zoom image UI
        self.updateZoomUI(self.dataset.images[imgIndices[0]])
        self.boUpdatingUI = False


    # Create all UI elements
    def createUI(self):

        # ------------
        # Callbacks
        # ------------
        # Callback for image label dropdown menu
        def dropdown_changed(obj):
            # Note that updating the dropdown label in code (e.g. in the updateUI() function)
            # also triggers this change event. Hence need to check if self.boUpdatingUI is False.
            if obj['type'] == 'change' and obj['name'] == 'value' and not self.boUpdatingUI:
                imgIndex = int(obj['owner'].description[6:])
                imgObj   = self.dataset.images[imgIndex]
                newLabelName = obj['owner'].value
                oldLabelName = imgObj.label

                # physically move image to sub-directory of the new label
                imgObj.label = newLabelName
                imgPathSrc = os.path.join(self.imgOrigDir, oldLabelName, imgObj.filename)
                imgPathDst = os.path.join(self.imgOrigDir, newLabelName, imgObj.filename)
                if os.path.exists(imgPathDst):
                    raise Exception(
                        "Cannot more image from {} to {} since the destination already exists.".format(imgPathSrc,imgPathDst))
                shutil.move(imgPathSrc, imgPathDst)
                print("Moved image file from {} to {}.".format(imgPathSrc, imgPathDst))

        # Callback for "zoom" button
        def img_button_pressed(obj):
            imgIndex = int(obj.value)
            imgObj = self.dataset.images[imgIndex]
            self.updateZoomUI(imgObj)

        # Callback for "next images" or "previous images" buttons
        def page_button_pressed(obj):
            self.pageIndex += int(obj.value)
            self.pageIndex = max(0, self.pageIndex)
            self.pageIndex = min(self.pageIndex, len(self.pageImgIndices) - 1)
            self.updateUI()

        # Callback for "image page" slider
        def page_slider_changed(obj):
            try:
                self.pageIndex = int(obj['new']['value'])
                self.updateUI()
            except Exception as e:
                pass

        # Init
        self.boUpdatingUI = False

        # ------------
        # UI - image grid
        # ------------
        self.wImgs    = []
        self.wLabels  = []
        self.wButtons = []
        wImgLabelButtons = []

        for i in range(self.gridSize[0] * self.gridSize[1]):
            # Initialize images
            wImg = widgets.Image(width=150, description="")
            #wImg = widgets.Image(height=400, description="")
            self.wImgs.append(wImg)

            # Initialize dropdown menus
            wLabel = widgets.Dropdown(options=self.labels, value=self.labels[0], text="Image 0", description="Image 0")
            wLabel.layout.width = '200px'
            wLabel.observe(dropdown_changed, names='value')
            self.wLabels.append(wLabel)

            # Initialize zoom buttons
            wButton = widgets.Button(description="Image id: ", value="")
            wButton.layout.width = "100px"
            wButton.button_style = 'warning'
            wButton.on_click(img_button_pressed)
            self.wButtons.append(wButton)

            # combine into image grid widget
            wImgLabelButton = widgets.VBox(children=[wButton, wImg, wLabel])
            wImgLabelButton.width = '230px'
            wImgLabelButtons.append(wImgLabelButton)

        # Image grid widget
        wGridHBoxes = []
        for r in range(self.gridSize[0]):
            hbox = widgets.HBox(children=[wImgLabelButtons[r * self.gridSize[1] + c] for c in range(self.gridSize[1])])
            hbox.layout.padding = '10px'
            wGridHBoxes.append(hbox)
        wImgGrid = widgets.VBox(wGridHBoxes)

        # ------------
        # UI - zoom window
        # ------------
        wNextPageButton = widgets.Button(description="Next images", value="1")
        wNextPageButton.value = "1"  # should not be necessary but bug on some jupyter versions otherwise
        wNextPageButton.layout.width = '120px'
        wNextPageButton.button_style = 'primary'
        wNextPageButton.on_click(page_button_pressed)

        wPreviousPageButton = widgets.Button(description="Previous images", value="-1",
                                             layout=Layout(color='white', background_color='lightblue'))
        wPreviousPageButton.value = "-1"
        wPreviousPageButton.layout.width = '120px'
        wPreviousPageButton.button_style = 'primary'
        wPreviousPageButton.on_click(page_button_pressed)

        self.wPageSlider = IntSlider(min=0, max=len(self.pageImgIndices) - 1, step=1, value=self.pageIndex,
                                     continuous_update=False, description='Image page:')
        self.wPageSlider.observe(page_slider_changed)

        self.wZoomHeader = widgets.Text("")
        self.wZoomHeader.layout.width = "100px"
        self.wZoomHeader.layout.color = 'white'
        self.wZoomHeader.layout.background_color = 'orange'
        self.wZoomImg = widgets.Image()
        self.wZoomImg.layout.width = str(self.wZoomImgWidth) + 'px'
        self.wZoomTextArea = widgets.Textarea()
        self.wZoomTextArea.layout.width  = '500px'
        self.wZoomTextArea.layout.height = '100px'

        #wZoomButtonSlider = widgets.HBox([widgets.VBox([wNextPageButton, wPreviousPageButton]),
        #                                  self.wPageSlider])  # self.wZoomHeader
        wZoomButtonSlider = widgets.VBox([wNextPageButton, wPreviousPageButton, self.wPageSlider])
        wZoomButtonSlider.layout.width = str(self.wZoomImgWidth + 20) + 'px' # '420px'


        # ------------
        # UI - final
        # ------------
        annotationUI = widgets.HBox(children=[widgets.VBox(children=[wZoomButtonSlider, self.wZoomImg, self.wZoomTextArea], width=520),
                                              wImgGrid])
        annotationUI.layout.border_color = 'black'
        annotationUI.layout.border_style = 'solid'
        tabsUI = widgets.Tab(children=[annotationUI])
        tabsUI.set_title(0, 'Image Annotation')

        # Update UI with actual images
        self.updateUI()
        return (tabsUI)