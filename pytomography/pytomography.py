import vtk
from vtk.util import numpy_support
import slicer, qt
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import logging
import os
# from pytomography.io.SPECT import dicom, simind
# from pytomography.projectors import SPECTSystemMatrix
# from pytomography.transforms import SPECTAttenuationTransform, SPECTPSFTransform
# from pytomography.algorithms import OSEM
# from pytomography.transforms import GaussianFilter
# from torch import poisson
# import dicom
# import numpy as np

# from __main__ import qt
# from PyQt5.QtWidgets import QWidget
# from PyQt5.QtCore import pyqtSlot
# from PyQt5 import uic

#
# pytomography
#

class PyTomography(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "PyTomography"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Pytomography"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Luke Polson (QURIT), Maziar Sabouri (QURIT), Obed Dzikunu (QURIT), Shadab Ahamed (QURIT)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#pytomography">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # pytomography1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='pytomography',
        sampleName='pytomography1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'pytomography1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='pytomography1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='pytomography1'
    )

    # pytomography2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='pytomography',
        sampleName='pytomography2',
        thumbnailFileName=os.path.join(iconsPath, 'pytomography2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='pytomography2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='pytomography2'
    )


#
# pytomographyWidget
#

class PyTomographyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/pytomography.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = PyTomographyLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).

        # Buttons
        self.ui.PathLineEdit.connect('currentPathChanged(QString)', self.logic.onPhotoPeakButtonClicked)
        #self.ui.PathLineEdit.currentTextChanged.connect(self.logic.onPhotoPeakButtonClicked)

        # Make sure parameter node is initialized (needed for module reload)

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        pass
        # Make sure parameter node exists and observed
        # self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())


#
# pytomographyLogic
#

class PyTomographyLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        slicer.util.pip_install("pytomography")
        import pytomography
        from pytomography.io.SPECT import dicom, simind
        from pytomography.projectors import SPECTSystemMatrix
        from pytomography.transforms import SPECTAttenuationTransform, SPECTPSFTransform
        from pytomography.algorithms import OSEM
        from pytomography.transforms import GaussianFilter
        import numpy as np
        
        # a = torch.tensor([1])
        print("Im here")

    def onPhotoPeakButtonClicked(self, directory):
        from pytomography.io.SPECT import dicom, simind
    # Implement the code to allow the user to select a CT attenuation file
        if directory:
            self.object_meta, self.proj_meta = dicom.get_metadata(directory)
            self.projections = dicom.get_projections(directory)
            self.display_projections(self.projections)

    def display_projections(self, projections):
        # Load the array from the file
        array = projections[0].cpu().numpy()
        print(array.shape)
        # Create a scalar volume node from the NumPy array
        volume_node = slicer.vtkMRMLScalarVolumeNode()
    
    # Convert NumPy array to VTK array
        vtk_array = numpy_support.numpy_to_vtk(array.ravel(), array_type=numpy_support.get_vtk_array_type(array.dtype))
        vtk_image_data = vtk.vtkImageData()
        vtk_image_data.SetDimensions(array.shape[::-1])
        vtk_image_data.GetPointData().SetScalars(vtk_array)
        
        # Set VTK image data to the volume node
        volume_node.SetAndObserveImageData(vtk_image_data)
        
        # Add the volume node to the scene
        slicer.mrmlScene.AddNode(volume_node)
        
        # Create default display nodes for visualization
        volume_node.CreateDefaultDisplayNodes()
        # # Set the spacing and origin (you may need to adjust these based on your data)
        volume_node.SetSpacing([1.0, 1.0, 1.0])
        volume_node.SetOrigin([0.0, 0.0, 0.0])

        # # Display the volume in the 3D view
        # display_node = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(volume_node)
        # if not display_node:
        #     display_node = slicer.modules.volumerendering.logic().CreateVolumeRenderingDisplayNode()
        #     slicer.mrmlScene.AddNode(display_node)
        #     slicer.modules.volumerendering.logic().UpdateDisplayNodeFromVolumeNode(display_node, volume_node)

        # # Set the rendering parameters (you may need to adjust these based on your data)
        # display_node.SetVisibility3DFill(True)
        # display_node.SetVisibility2DFill(True)
        # display_node.SetVisibility2DFill(False)
        # display_node.SetVisibility2DOutline(True)
        # display_node.SetVisibility3DOutline(True)

        # Set the window/level (you may need to adjust these based on your data)
        #display_node.SetWindowLevel(255, 128)

        # # Fit the 3D view to the loaded volume
        # slicer.app.processEvents()
        # slicer.app.applicationLogic().FitSliceToAll()


#
# pytomographyTest
#

class PyTomographyTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_pytomography1()

    def test_pytomography1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('pytomography1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = pytomographyLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
