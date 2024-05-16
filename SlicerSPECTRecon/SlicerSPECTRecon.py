import logging
import os
from typing import Annotated, Optional
import vtk
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer import vtkMRMLScalarVolumeNode

class SlicerSPECTRecon(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SlicerSPECTRecon")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = ["Tomographic Reconstruction"]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Obed K. Dzikunu (QURIT), Luke Polson (QURIT), Maziar Sabouri (QURIT), Shadab Ahamed (QURIT)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#pytomography">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

class SlicerSPECTReconWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self._projectionList = None

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)
        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/SlicerSPECTRecon.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SlicerSPECTReconLogic()
        # Connections
        # # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.multiInput.connect('checkedNodesChanged()', self.updateParameterNodeFromGUI)
        self.ui.attenuationdata.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
        self.ui.spect_collimator_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.spect_scatter_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.photopeak_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.spect_upperwindow_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.spect_lowerwindow_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.algorithm_selector_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.osem_iterations_spinbox.connect('valueChanged(int)', self.updateParameterNodeFromGUI)
        self.ui.osem_subsets_spinbox.connect('valueChanged(int)', self.updateParameterNodeFromGUI)
        self.ui.outputVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
        # Buttons
        self.ui.osem_reconstruct_pushbutton.connect('clicked(bool)', self.onReconstructButton)
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        
    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()
        
    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)# self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)
        

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
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
        
    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)
        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
            
    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True
        inputVolume1 = self._parameterNode.GetNodeReference("InputVolume1")
        if inputVolume1 and self._parameterNode.GetParameter("Photopeak") :
            self.getProjectionData(inputVolume1)
        last_text={}
        lastPhotopeakSelection = last_text.get(self.ui.photopeak_combobox.objectName, "None")
        lastUpperWindowSelection = last_text.get(self.ui.spect_upperwindow_combobox.objectName, "None")
        lastLowerWindowSelection = last_text.get(self.ui.spect_lowerwindow_combobox.objectName, "None")
        if self.ui.photopeak_combobox.currentText !=  lastPhotopeakSelection:
            photopeak_value = self._parameterNode.GetParameter("Photopeak")
            photopeak_index = self.ui.photopeak_combobox.findText(photopeak_value)
            self.ui.photopeak_combobox.setCurrentIndex(photopeak_index)
            last_text[self.ui.photopeak_combobox.objectName] = self.ui.photopeak_combobox.currentText
        if self.ui.spect_upperwindow_combobox.currentText != lastUpperWindowSelection:
            upperwindow_value = self._parameterNode.GetParameter("UpperWindow")
            upperwindow_index = self.ui.spect_upperwindow_combobox.findText(upperwindow_value)
            self.ui.spect_upperwindow_combobox.setCurrentIndex(upperwindow_index)
            last_text[self.ui.spect_upperwindow_combobox.objectName] = self.ui.spect_upperwindow_combobox.currentText
        if self.ui.spect_lowerwindow_combobox.currentText != lastLowerWindowSelection:
            lowerwindow_value = self._parameterNode.GetParameter("LowerWindow")
            lowerwindow_index = self.ui.spect_lowerwindow_combobox.findText(lowerwindow_value)
            self.ui.spect_lowerwindow_combobox.setCurrentIndex(lowerwindow_index)
            last_text[self.ui.spect_lowerwindow_combobox.objectName] = self.ui.spect_lowerwindow_combobox.currentText
        if inputVolume1:
            self.ui.outputVolumeSelector.baseName = inputVolume1.GetName() + " reconstructed"
        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
        self._projectionList = self.ui.multiInput.checkedNodes()
        for counter, node in enumerate(self._projectionList, start=1):
            if node:
                nodeID = node.GetID()
                self._parameterNode.SetNodeReferenceID(f"InputVolume{counter}", nodeID)  
        self._parameterNode.SetNodeReferenceID("AttenuationData", self.ui.attenuationdata.currentNodeID)
        self._parameterNode.SetParameter("Collimator", self.ui.spect_collimator_combobox.currentText)
        self._parameterNode.SetParameter("Scatter", self.ui.spect_scatter_combobox.currentText)
        self._parameterNode.SetParameter("Photopeak", str(self.ui.photopeak_combobox.currentText))
        self._parameterNode.SetParameter("UpperWindow", self.ui.spect_upperwindow_combobox.currentText)
        self._parameterNode.SetParameter("LowerWindow", self.ui.spect_lowerwindow_combobox.currentText)
        self._parameterNode.SetParameter("Algorithm", self.ui.algorithm_selector_combobox.currentText)
        self._parameterNode.SetParameter("Iterations", str(self.ui.osem_iterations_spinbox.value))
        self._parameterNode.SetParameter("Subsets", str(self.ui.osem_subsets_spinbox.value))
        self._parameterNode.SetParameter("OutputVolume", self.ui.outputVolumeSelector.currentNodeID)
        self._parameterNode.EndModify(wasModified)

    def getProjectionData(self,node):
        inputdatapath = self.logic.pathFromNode(node)
        energy_window,_ = self.logic.getEnergyWindow(inputdatapath)
        self.ui.spect_upperwindow_combobox.clear()
        self.ui.spect_upperwindow_combobox.addItems(energy_window)
        self.ui.spect_lowerwindow_combobox.clear()
        self.ui.spect_lowerwindow_combobox.addItems(energy_window)
        self.ui.photopeak_combobox.clear()
        self.ui.photopeak_combobox.addItems(energy_window)

    def onReconstructButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            # Create new volume node, if not selected yet
            if not self.ui.outputVolumeSelector.currentNode():
                self.ui.outputVolumeSelector.addNode()
        recon_array, fileNMpaths= self.logic.reconstruct( 
            self._projectionList, self.ui.attenuationdata.currentNode(), self.ui.spect_collimator_combobox.currentText, 
            self.ui.spect_scatter_combobox.currentText, self.ui.photopeak_combobox.currentIndex, 
            self.ui.spect_upperwindow_combobox.currentIndex, self.ui.spect_lowerwindow_combobox.currentIndex,
            self.ui.algorithm_selector_combobox.currentText, self.ui.osem_iterations_spinbox.value, 
            self.ui.osem_subsets_spinbox.value)
        self.logic.stitchMultibed(recon_array, fileNMpaths, self.ui.outputVolumeSelector.currentNode())

class SlicerSPECTReconLogic(ScriptedLoadableModuleLogic):
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
        slicer.util.pip_install("pytomography==2.1.1")
        print("Im here")

    def getEnergyWindow(self, directory):
        # Import
        import numpy as np
        import pydicom
        # Implementation
        ds = pydicom.read_file(directory)
        energy_windows =[]
        lower_limits = []
        for energy_window_information in ds.EnergyWindowInformationSequence:
            lower_limit = energy_window_information.EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
            upper_limit = energy_window_information.EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
            energy_window_name = energy_window_information.EnergyWindowName
            lower_limits.append(lower_limit)
            energy_windows.append(f'{energy_window_name} ({lower_limit:.2f}keV - {upper_limit:.2f}keV)')
        idx_sorted = np.argsort(lower_limits)
        energy_windows = list(np.array(energy_windows)[idx_sorted])
        return energy_windows, idx_sorted

    def pathFromNode(self, node):
        #TODO: Review this function to handle the case where the data was dragged and dropped
        if node is not None:
            storageNode = node.GetStorageNode()
            if storageNode is not None: # loaded via drag-drop
                filepath = storageNode.GetFullNameFromFileName()
            else: # Loaded via DICOM browser
                instanceUIDs = node.GetAttribute("DICOM.instanceUIDs").split()
                filepath = slicer.dicomDatabase.fileForInstance(instanceUIDs[0])
        else: # Loaded via DICOM browser
            instanceUIDs = node.GetAttribute("DICOM.instanceUIDs").split()
            filepath = slicer.dicomDatabase.fileForInstance(instanceUIDs[0])
        return filepath

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Collimator"):
            parameterNode.SetParameter("Collimator", "Choose Collimator")
        if not parameterNode.GetParameter("Scatter"):
            parameterNode.SetParameter("Scatter", "Select Scatter Window")
        if not parameterNode.GetParameter("UpperWindow"):
            parameterNode.SetParameter("UpperWindow", "Select Upper Window")
        if not parameterNode.GetParameter("LowerWindow"):
            parameterNode.SetParameter("LowerWindow", "Select Lower Window")
        if not parameterNode.GetParameter("Algorithm"):
            parameterNode.SetParameter("Algorithm", "Select Algorithm")
        if not parameterNode.GetParameter("Iterations"):
            parameterNode.SetParameter("Iterations", "0")
        if not parameterNode.GetParameter("Subsets"):
            parameterNode.SetParameter("Subsets", "0")

    def reconstruct(self, files_NM, ct_file, collimator, scatter, photopeak, 
                    upperwindow, lowerwindow, algorithm, iter, subset): 
        from pytomography.io.SPECT import dicom
        from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
        from pytomography.projectors.SPECT import SPECTSystemMatrix
        from pytomography.likelihoods import PoissonLogLikelihood
        from pytomography.algorithms import OSEM
        fileNMpaths = []
        for fileNM in files_NM:
            path = self.pathFromNode(fileNM)
            fileNMpaths.append(path)
        _,idx = self.getEnergyWindow(fileNMpaths[0])
        index_peak = idx[photopeak]
        print(f'photopeak id: ', index_peak)
        index_upper = idx[upperwindow]
        print(f'upperwindow: {index_upper}')
        index_lower = idx[lowerwindow]
        print(f'lower: ', index_lower)
        print("Reconstructing volume...")
        import os
        ct_path = os.path.dirname(self.pathFromNode(ct_file))
        print(ct_path)
        files_CT = [os.path.join(ct_path, file) for file in os.listdir(ct_path)]
        projectionss = dicom.load_multibed_projections(fileNMpaths)
        recon_array = []
        for counter, fileNMpath in enumerate(fileNMpaths, start=0):
            projections = projectionss[counter]
            object_meta, proj_meta = dicom.get_metadata(fileNMpath, index_peak)
            photopeak = projections[index_peak].unsqueeze(0)
            scatter = dicom.get_scatter_from_TEW_projections(fileNMpath, projections, index_peak, index_lower, index_upper)
            attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT, fileNMpath, index_peak)
            energy_kev = 208 # TODO: needs to be center of photopeak window
            psf_meta = dicom.get_psfmeta_from_scanner_params(collimator, energy_kev)
            att_transform = SPECTAttenuationTransform(attenuation_map)
            psf_transform = SPECTPSFTransform(psf_meta)
            system_matrix = SPECTSystemMatrix(
                obj2obj_transforms = [att_transform,psf_transform],
                proj2proj_transforms = [],
                object_meta = object_meta,
                proj_meta = proj_meta)

            likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)

            if algorithm == "OSEM":
                reconstruction_algorithm = OSEM(likelihood)
            
            reconstructed_object = reconstruction_algorithm(n_iters=iter, n_subsets=subset)

            recon_array.append(reconstructed_object)

        return recon_array, fileNMpaths

    def stitchMultibed(self, recon_array, fileNMpaths, outputVolume):
        # Imports
        from pytomography.io.SPECT import dicom
        import torch
        from DICOMLib import DICOMUtils
        import tempfile
        import shutil
        # Code
        a = torch.stack(recon_array)
        recon_stitched = dicom.stitch_multibed(recons=torch.cat(recon_array), files_NM = fileNMpaths)
        reconstructedDCMInstances = dicom.save_dcm(save_path = None, object = recon_stitched, 
                                                   file_NM = fileNMpaths[0], recon_name = 'OSEM_4it_10ss', return_ds =True)
        temp_dir = tempfile.mkdtemp()
        for i, dataset in enumerate(reconstructedDCMInstances):
            temp_file_path = os.path.join(temp_dir, f"temp_{i}.dcm")
            dataset.save_as(temp_file_path)
        loadedNodeIDs = []
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(temp_dir, db)
            patientUIDs = db.patients()
            for patientUID in patientUIDs:
                loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))
        if loadedNodeIDs:
            volumeNode = slicer.mrmlScene.GetNodeByID(loadedNodeIDs[0])
        outputVolume.SetAndObserveImageData(volumeNode.GetImageData())
        outputVolume.CreateDefaultDisplayNodes() 
        colorTableID = volumeNode.GetDisplayNode().GetColorNodeID()
        outputVolume.GetDisplayNode().SetAndObserveColorNodeID(colorTableID)
        window = volumeNode.GetDisplayNode().GetWindow()
        level = volumeNode.GetDisplayNode().GetLevel()
        outputVolume.GetDisplayNode().SetWindow(window)
        outputVolume.GetDisplayNode().SetLevel(level)
        volumeMatrix = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(volumeMatrix)
        # Apply the same orientation matrix to the outputVolume
        outputVolume.SetRASToIJKMatrix(volumeMatrix)
        slicer.mrmlScene.RemoveNode(volumeNode)
        shutil.rmtree(temp_dir)
        print("Reconstruction successful")

class SlicerSPECTReconTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_pytomography1()

    def test_pytomography1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """
        import SampleData
        self.delayDisplay("Starting the test")
        # Get/create input data
        registerSampleData()
        inputVolume = SampleData.downloadSample("pytomography1")
        self.delayDisplay("Loaded test data set")
        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)
        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100
        # Test the module logic
        logic = SlicerSPECTReconLogic()
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
        self.delayDisplay("Test passed")