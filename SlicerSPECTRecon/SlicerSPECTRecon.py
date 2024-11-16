import slicer
import importlib

def ensure_packages_installed():
    """Ensure required packages are installed."""
    required_packages = {
        "pytomography": "3.2.4",
        "beautifulsoup4": None,
    }
    for package, version in required_packages.items():
        try:
            if version:
                package_name = f"{package}=={version}"
                slicer.util.pip_install(package_name)
            else:
                package_name = package
            importlib.import_module(package)
        except ImportError:
            slicer.util.pip_install(package_name)
ensure_packages_installed()

import pytomography
print(pytomography.__version__)
import logging
import os
from typing import Annotated, Optional
import vtk
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer import vtkMRMLScalarVolumeNode
from DICOMLib import DICOMUtils
import re
from Logic.SlicerSPECTReconLogic import SlicerSPECTReconLogic
from Logic.VtkkmrmlUtils import *
from Logic.MetadataUtils import *
from Logic.SimindToDicom import *
from Testing.Python.reconstruction import ReconstructionTest
from Testing.Python.simind_to_dicom import SimindToDicomConverterTest

__submoduleNames__ = [
    "SlicerSPECTReconLogic",
    "SlicerSPECTReconTest",
    "vtkkmrmlutils",
    "simindToDicomConverterTest",
    "testutils_builder",
    "transforms",
    "volumeutils",
    "systemMatrix",
    "simindToDicom",
    "priors",
    "algorithms",
    "getmetadatautils",
    "likelihood",
    "dicomvalues"
]
__package__ = "SlicerSPECTRecon"
mod = importlib.import_module("Logic", __name__)
importlib.reload(mod)
__all__ = ["SlicerSPECTRecon", "SlicerSPECTReconWidget", "SlicerSPECTReconLogic", "SlicerSPECTReconTest"]


class SlicerSPECTRecon(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        # TODO: make this more human readable by adding spaces
        self.parent.title = _("SlicerSPECTRecon")
        self.parent.categories = ["Tomographic Reconstruction"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Obed Korshie Dzikunu (QURIT, Canada)", 
            "Luke Polson (QURIT, Canada)", 
            "Maziar Sabouri (QURIT, Canada)", 
            "Shadab Ahamed (QURIT, Canada)"
            ]
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
            This module implements GPU accelerated SPECT image reconstruction. This module is for academic purposes. Please do not use for clinical work.
            See more information in <a href="https://github.com/PyTomography/slicer_spect_recon/blob/main/User_Manual.md">module documentation</a>.
            """)
        self.parent.acknowledgementText = _("""
            This software was originally developed by Obed Korshie Dzikunu, Luke Polson, Maziar Sabouri and Shadab Ahamed of the
            Quantitative Radiomolecular Imaging and Therapy Lab of the BC Cancer Research Institute, Canada.
            """) #replace with organization, grant and thanks
        try:
            slicer.selfTests
        except AttributeError:
            slicer.selfTests = {}
        slicer.selfTests["SlicerSPECTReconTest"] = self.runTest

    def runTest(self, msec=1000, **kwargs):
        """
        :param msec: delay to associate with :func:`ScriptedLoadableModuleTest.delayDisplay()`.
        """
        logging.info("\n******* Starting Tests of SlicerSPECTRecon **********\n")
        # Test SIMIND Converter
        testCase = SimindToDicomConverterTest()
        testCase.messageDelay = msec
        testCase.runTest(**kwargs)
        # # Test Reconstructions
        testCase = ReconstructionTest()
        testCase.messageDelay = msec
        testCase.runTest(**kwargs)
        logging.info("\n******* All tests passed **********\n")

# -------------------------------------------------
# ------------ SlicerSPECTReconWidget -------------
# -------------------------------------------------

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
        self.logic = SlicerSPECTReconLogic()
        # Connections
        # # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.vtkMRMLScene.NodeAddedEvent, self.onMRMLSceneNodeAdded)
        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.setupConnections()
        # initialize for loading data in case
        # Add uncertainty table
        self.uncertainty_table =  createTable(
            ['Image', 'Segmentation', 'Mask', 'Absolute Uncertainty', 'Percent Uncertainty'],
            ['string', 'string', 'string', 'float', 'float']
        )

    def filterNMVolumes(self):
        """Filter the projection data to show only Nuclear Medicine volumes."""
        self.ui.NM_data_selector.noneEnabled = False
        self.ui.NM_data_selector.addEnabled = False
        nmSOPClassUID = "1.2.840.10008.5.1.4.1.1.20"  # Standard SOPClassUID for PET images
        self.ui.NM_data_selector.addAttribute("vtkMRMLScalarVolumeNode", "SOPClassUID", nmSOPClassUID)
        self.ui.NM_data_selector.setMRMLScene(None)  # Clear the combobox first
        self.ui.NM_data_selector.setMRMLScene(slicer.mrmlScene)

    def filterCTVolumes(self):
        """Filter the attenuation data to show only CT volumes."""
        ctSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # Standard SOPClassUID for CT images
        sopclassuidtag = '0008,0016'
        for nodeIndex in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLScalarVolumeNode")):
            volNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, "vtkMRMLScalarVolumeNode")
            if volNode.GetAttribute("SOPClassUID"):
                pass
            else:
                uids = volNode.GetAttribute('DICOM.instanceUIDs')
                if uids is not None:
                    uid = uids.split()
                if len(uid)>1:
                    filepaths = [slicer.dicomDatabase.fileForInstance(instanceUID) for instanceUID in uid]
                    for filepath in filepaths:
                        sopClassUID = slicer.dicomDatabase.fileValue(filepath, sopclassuidtag)
                        volNode.SetAttribute("SOPClassUID", sopClassUID)
                else:
                    filepath = slicer.dicomDatabase.fileForInstance(uid)
                    sopClassUID = slicer.dicomDatabase.fileValue(filepath, sopclassuidtag)
                    volNode.SetAttribute("SOPClassUID", sopClassUID)
        self.ui.attenuationdata.addAttribute("vtkMRMLScalarVolumeNode", "SOPClassUID", ctSOPClassUID)
        
    def setupConnections(self):
        self.ui.attenuation_toggle.connect('toggled(bool)', self.hideShowItems)
        self.ui.psf_toggle.connect('toggled(bool)', self.hideShowItems)
        self.ui.scatter_toggle.connect('toggled(bool)', self.hideShowItems)
        self.ui.usePriorAnatomicalCheckBox.connect('toggled(bool)', self.hideShowItems)
        self.ui.algorithm_selector_combobox.connect('currentTextChanged(QString)', self.hideShowItems)
        self.ui.spect_scatter_combobox.connect('currentTextChanged(QString)', self.hideShowItems)
        self.ui.priorFunctionSelector.connect('currentTextChanged(QString)', self.hideShowItems)
        # Update info
        self.ui.NM_data_selector.connect('checkedNodesChanged()', self.updateParameterNodeFromGUI)
        self.ui.attenuationdata.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
        self.ui.anatomyPriorImageNode.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
        self.ui.UncImageMRMLNodeComboBox.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
        self.ui.UncImageSegmentSelectorWidget.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
        self.ui.spect_collimator_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.spect_scatter_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.photopeak_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.spect_upperwindow_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.spect_lowerwindow_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.IntrinsicResolutionSpinBox.connect('valueChanged(float)', self.updateParameterNodeFromGUI)
        self.ui.osem_iterations_spinbox.connect('valueChanged(int)', self.updateParameterNodeFromGUI)
        self.ui.osem_subsets_spinbox.connect('valueChanged(int)', self.updateParameterNodeFromGUI)
        # Default values
        self.ui.AttenuationGroupBox.setVisible(self.ui.attenuation_toggle.checked)
        self.ui.PSFGroupBox.setVisible(self.ui.psf_toggle.checked)
        self.ui.ScatterGroupBox.setVisible(self.ui.scatter_toggle.checked)
        self.ui.PriorGroupBox.setVisible(False)
        # Buttons
        self.ui.osem_reconstruct_pushbutton.connect('clicked(bool)', self.onReconstructButton)
        self.ui.computeUncertaintyPushButton.connect('clicked(bool)', self.onComputeUncertaintyButton)
        # Data converters
        self.ui.data_converter_comboBox.connect('currentTextChanged(QString)', self.hideShowItems)
        self.ui.postReconSelectionComboBox.connect('currentTextChanged(QString)', self.hideShowItems)
        self.ui.simind_nenergy_spinBox.connect('valueChanged(int)', self.hideShowItems)
        self.ui.simind_patientname_lineEdit.connect('textChanged(QString)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind_tperproj_doubleSpinBox.connect('valueChanged(double)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind_scale_doubleSpinBox.connect('valueChanged(double)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind_randomseed_spinBox.connect('valueChanged(int)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind2dicom_groupBox.setVisible(False)
        self.ui.data_converters_CollapsibleButton.collapsed = True
        for i in range(2,10):
            getattr(self.ui, f'PathLineEdit_w{i}').setVisible(False)
            getattr(self.ui, f'label_w{i}').setVisible(False)
        self.ui.simind_projections_pushButton.connect('clicked(bool)', self.saveSIMINDProjections)
        self.ui.simindSaveAmapPushButton.connect('clicked(bool)', self.saveSIMINDAmap)
        # Post reconstruction
        self.ui.PostReconCollapsibleButton.collapsed = True
        self.ui.uncertaintyEstimationGroupBox.setVisible(False)
        # Multi photopeak
        self.ui.multiPhotopeakGroupBox.setVisible(False)
        self.ui.multiScatterWindowGroupBox.setVisible(False)
        self.ui.multiPhotopeakCheckbox.connect('toggled(bool)', self.hideShowItems)
        self.ui.numPhotopeaksSpinBox.connect('valueChanged(int)', self.hideShowItems)
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        
    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()
        self.filterNMVolumes()

    def onMRMLSceneNodeAdded(self, caller, event):
        for nodeIndex in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLScalarVolumeNode")):
            volNode = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, "vtkMRMLScalarVolumeNode")
            uids = volNode.GetAttribute('DICOM.instanceUIDs')
            if uids:
                uid = uids.split()
                sopclassuidtag = '0008,0016'
                if len(uid) > 1:
                    filepaths = [slicer.dicomDatabase.fileForInstance(instanceUID) for instanceUID in uid]
                    for filepath in filepaths:
                        sopClassUID = slicer.dicomDatabase.fileValue(filepath, sopclassuidtag)
                        volNode.SetAttribute("SOPClassUID", sopClassUID)
                else:
                    filepath = slicer.dicomDatabase.fileForInstance(uid[0])
                    sopClassUID = slicer.dicomDatabase.fileValue(filepath, sopclassuidtag)
                    volNode.SetAttribute("SOPClassUID", sopClassUID)
        self.filterNMVolumes()
        
    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)
        
    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()
        self.filterNMVolumes()

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
        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
            
    def hideShowItems(self, called=None, event=None):
        self.ui.AttenuationGroupBox.setVisible(self.ui.attenuation_toggle.checked)
        self.filterCTVolumes()# Call filtering method
        self.ui.PSFGroupBox.setVisible(self.ui.psf_toggle.checked)
        self.ui.ScatterGroupBox.setVisible(self.ui.scatter_toggle.checked)
        # Scatter stuff
        if self.ui.spect_scatter_combobox.currentText=='Energy Window':
            self.ui.upperwindowLabel.setVisible(True)
            self.ui.lowerwindowLabel.setVisible(True)
            self.ui.spect_upperwindow_combobox.setVisible(True)
            self.ui.spect_lowerwindow_combobox.setVisible(True)
        # Algorithm stuff
        if self.ui.algorithm_selector_combobox.currentText!='OSEM':
            self.ui.PriorGroupBox.setVisible(True)
        elif self.ui.algorithm_selector_combobox.currentText=='OSEM':
            self.ui.PriorGroupBox.setVisible(False)
        # Prior stuff
        beta_show = delta_show = gamma_show = False
        if self.ui.priorFunctionSelector.currentText=='RelativeDifferencePenalty':
            beta_show = gamma_show = True
        elif self.ui.priorFunctionSelector.currentText=='Quadratic':
            beta_show = delta_show = True
        elif self.ui.priorFunctionSelector.currentText=='LogCosh':
            beta_show = delta_show = True
        # Now show priors
        self.ui.priorBetaLabel.setVisible(beta_show)
        self.ui.priorGammaLabel.setVisible(gamma_show)
        self.ui.priorDeltaLabel.setVisible(delta_show)
        self.ui.priorBetaSpinBox.setVisible(beta_show)
        self.ui.priorGammaSpinBox.setVisible(gamma_show)
        self.ui.priorDeltaSpinBox.setVisible(delta_show)
        self.ui.priorHyperparameterGroupbox.setVisible(self.ui.priorFunctionSelector.currentText!='None')
        self.ui.usePriorAnatomicalCheckBox.setVisible(self.ui.priorFunctionSelector.currentText!='None')
        self.ui.priorAnatomicalGroupBox.setVisible(self.ui.usePriorAnatomicalCheckBox.checked)
        # Data converters
        self.ui.simind2dicom_groupBox.setVisible(self.ui.data_converter_comboBox.currentText=='SIMIND to DICOM')
        n_windows = self.ui.simind_nenergy_spinBox.value
        for i in range(1,10):
            getattr(self.ui, f'PathLineEdit_w{i}').setVisible(i<=n_windows)
            getattr(self.ui, f'label_w{i}').setVisible(i<=n_windows)
        # Post reconstruction
        self.ui.uncertaintyEstimationGroupBox.setVisible(self.ui.postReconSelectionComboBox.currentText=='Uncertainty Estimation')
        # Multiphotopeak
        self.ui.multiPhotopeakGroupBox.setVisible(self.ui.multiPhotopeakCheckbox.checked)
        self.ui.multiScatterWindowGroupBox.setVisible(self.ui.multiPhotopeakCheckbox.checked)
        self.ui.scatterWindowGroupBox.setVisible(not self.ui.multiPhotopeakCheckbox.checked)
        self.ui.Photopeak.setVisible(not self.ui.multiPhotopeakCheckbox.checked)
        self.ui.photopeak_combobox.setVisible(not self.ui.multiPhotopeakCheckbox.checked)
        show3 = self.ui.numPhotopeaksSpinBox.value==3
        self.ui.multiPhotopeakLabel3.setVisible(show3)
        self.ui.multiPhotopeakComboBox3.setVisible(show3)
        self.ui.multiPhotopeakWeightLabel3.setVisible(show3)
        self.ui.multiPhotopeakWeightDoubleSpinBox1.setVisible(show3)
        self.ui.multiScatterUpperLabel3.setVisible(show3)
        self.ui.multiScatterLowerLabel3.setVisible(show3)
        self.ui.upperWindowMultiComboBox3.setVisible(show3)
        self.ui.lowerWindowMultiComboBox3.setVisible(show3)
        
    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._updatingGUIFromParameterNode:
            return
        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True
        inputVolume1 = self._parameterNode.GetNodeReference("InputVolume1")
        if inputVolume1 and self._parameterNode.GetParameter("PhotopeakIndex") :
            self.getProjectionData(inputVolume1)
        last_text = {}
        # Update photopeak
        self.ui.photopeak_combobox.setCurrentIndex(int(self._parameterNode.GetParameter("PhotopeakIndex")))
        self.ui.multiPhotopeakComboBox1.setCurrentIndex(int(self._parameterNode.GetParameter("MultiPhoto1Index")))
        self.ui.multiPhotopeakComboBox2.setCurrentIndex(int(self._parameterNode.GetParameter("MultiPhoto2Index")))
        self.ui.multiPhotopeakComboBox3.setCurrentIndex(int(self._parameterNode.GetParameter("MultiPhoto3Index")))
        self.ui.upperWindowMultiComboBox1.setCurrentIndex(int(self._parameterNode.GetParameter("UpperWindowMulti1Index")))
        self.ui.upperWindowMultiComboBox2.setCurrentIndex(int(self._parameterNode.GetParameter("UpperWindowMulti2Index")))
        self.ui.upperWindowMultiComboBox3.setCurrentIndex(int(self._parameterNode.GetParameter("UpperWindowMulti3Index")))
        self.ui.lowerWindowMultiComboBox1.setCurrentIndex(int(self._parameterNode.GetParameter("LowerWindowMulti1Index")))
        self.ui.lowerWindowMultiComboBox2.setCurrentIndex(int(self._parameterNode.GetParameter("LowerWindowMulti2Index")))
        self.ui.lowerWindowMultiComboBox3.setCurrentIndex(int(self._parameterNode.GetParameter("LowerWindowMulti3Index")))
        # Update photopeak1
        # photopeak_value = self._parameterNode.GetParameter("MultiPhoto1")
        # photopeak_index = self.ui.multiPhotopeakComboBox1.findText(photopeak_value)
        # self.ui.multiPhotopeakComboBox1.setCurrentIndex(photopeak_index)
        # last_text[self.ui.multiPhotopeakComboBox1.objectName] = self.ui.multiPhotopeakComboBox1.currentText
        # Scatter Stuff
        if self.ui.scatter_toggle.checked:
            upperwindow_value = self._parameterNode.GetParameter("UpperWindow")
            upperwindow_index = self.ui.spect_upperwindow_combobox.findText(upperwindow_value)
            self.ui.spect_upperwindow_combobox.setCurrentIndex(upperwindow_index)
            last_text[self.ui.spect_upperwindow_combobox.objectName] = self.ui.spect_upperwindow_combobox.currentText
            # Lower
            lowerwindow_value = self._parameterNode.GetParameter("LowerWindow")
            lowerwindow_index = self.ui.spect_lowerwindow_combobox.findText(lowerwindow_value)
            self.ui.spect_lowerwindow_combobox.setCurrentIndex(lowerwindow_index)
            last_text[self.ui.spect_lowerwindow_combobox.objectName] = self.ui.spect_lowerwindow_combobox.currentText
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
        self._projectionList = self.ui.NM_data_selector.checkedNodes()
        for counter, node in enumerate(self._projectionList, start=1):
            if node:
                nodeID = node.GetID()
                self._parameterNode.SetNodeReferenceID(f"InputVolume{counter}", nodeID)  
        self._parameterNode.SetNodeReferenceID("AttenuationData", self.ui.attenuationdata.currentNodeID)
        self._parameterNode.SetNodeReferenceID("AnatomyPriorImage", self.ui.anatomyPriorImageNode.currentNodeID)
        self._parameterNode.SetParameter("Collimator", self.ui.spect_collimator_combobox.currentText)
        self._parameterNode.SetParameter("Scatter", self.ui.spect_scatter_combobox.currentText)
        self._parameterNode.SetParameter("PhotopeakIndex", str(self.ui.photopeak_combobox.currentIndex))
        self._parameterNode.SetParameter("UpperWindow", self.ui.spect_upperwindow_combobox.currentText)
        self._parameterNode.SetParameter("LowerWindow", self.ui.spect_lowerwindow_combobox.currentText)
        self._parameterNode.SetParameter("Algorithm", self.ui.algorithm_selector_combobox.currentText)
        self._parameterNode.SetParameter("Iterations", str(self.ui.osem_iterations_spinbox.value))
        self._parameterNode.SetParameter("Subsets", str(self.ui.osem_subsets_spinbox.value))
        self._parameterNode.SetParameter("UncImage", self.ui.UncImageMRMLNodeComboBox.currentNodeID)
        self._parameterNode.SetParameter("UncImageSegmentNodeID", self.ui.UncImageSegmentSelectorWidget.currentNodeID())
        self._parameterNode.SetParameter("UncImageSegmentID", self.ui.UncImageSegmentSelectorWidget.currentSegmentID())
        # Multi energy window
        self._parameterNode.SetParameter("MultiPhoto1Index", str(self.ui.multiPhotopeakComboBox1.currentIndex))
        self._parameterNode.SetParameter("MultiPhoto2Index", str(self.ui.multiPhotopeakComboBox2.currentIndex))
        self._parameterNode.SetParameter("MultiPhoto3Index", str(self.ui.multiPhotopeakComboBox3.currentIndex))
        self._parameterNode.SetParameter("UpperWindowMulti1Index", str(self.ui.upperWindowMultiComboBox1.currentIndex))
        self._parameterNode.SetParameter("UpperWindowMulti2Index", str(self.ui.upperWindowMultiComboBox2.currentIndex))
        self._parameterNode.SetParameter("UpperWindowMulti3Index", str(self.ui.upperWindowMultiComboBox3.currentIndex))
        self._parameterNode.SetParameter("LowerWindowMulti1Index", str(self.ui.lowerWindowMultiComboBox1.currentIndex))
        self._parameterNode.SetParameter("LowerWindowMulti2Index", str(self.ui.lowerWindowMultiComboBox2.currentIndex))
        self._parameterNode.SetParameter("LowerWindowMulti3Index", str(self.ui.lowerWindowMultiComboBox3.currentIndex))
        self._parameterNode.EndModify(wasModified)

    def getProjectionData(self,node):
        inputdatapath = pathFromNode(node)
        energy_window,_,_ = getEnergyWindow(inputdatapath)
        all_combo_boxes = [
            self.ui.photopeak_combobox,
            self.ui.spect_upperwindow_combobox,
            self.ui.spect_lowerwindow_combobox,
            self.ui.multiPhotopeakComboBox1,
            self.ui.multiPhotopeakComboBox2,
            self.ui.multiPhotopeakComboBox3,
            self.ui.upperWindowMultiComboBox1,
            self.ui.upperWindowMultiComboBox2,
            self.ui.upperWindowMultiComboBox3,
            self.ui.lowerWindowMultiComboBox1,
            self.ui.lowerWindowMultiComboBox2,
            self.ui.lowerWindowMultiComboBox3,
        ]
        for combo_box in all_combo_boxes:
            combo_box.clear()
            combo_box.addItems(energy_window)
            
    def _get_photopeak_scatter_idxs(self, file_NM):
        _, mean_window_energies, idx_sorted = getEnergyWindow(file_NM)
        # Photopeak
        if self.ui.multiPhotopeakCheckbox.checked:
            num_peaks = self.ui.numPhotopeaksSpinBox.value
            photopeak_idx = [
                idx_sorted[self.ui.multiPhotopeakComboBox1.currentIndex],
                idx_sorted[self.ui.multiPhotopeakComboBox2.currentIndex],
                idx_sorted[self.ui.multiPhotopeakComboBox3.currentIndex],
            ][:num_peaks]
            upperwindow_idx = [
                idx_sorted[self.ui.upperWindowMultiComboBox1.currentIndex],
                idx_sorted[self.ui.upperWindowMultiComboBox2.currentIndex],
                idx_sorted[self.ui.upperWindowMultiComboBox3.currentIndex],
            ][:num_peaks]
            lowerwindow_idx = [
                idx_sorted[self.ui.lowerWindowMultiComboBox1.currentIndex],
                idx_sorted[self.ui.lowerWindowMultiComboBox2.currentIndex],
                idx_sorted[self.ui.lowerWindowMultiComboBox3.currentIndex],
            ][:num_peaks]
            if not self.ui.scatter_toggle: # enforce none if not checked
                upperwindow_idx = lowerwindow_idx = [None] * num_peaks
        else:
            photopeak_idx = idx_sorted[self.ui.photopeak_combobox.currentIndex]
            lowerwindow_idx = idx_sorted[self.ui.spect_lowerwindow_combobox.currentIndex]
            upperwindow_idx = idx_sorted[self.ui.spect_upperwindow_combobox.currentIndex]
            if not self.ui.scatter_toggle: # enforce none if not checked
                upperwindow_idx = lowerwindow_idx = None
        return photopeak_idx, upperwindow_idx, lowerwindow_idx

    def onReconstructButton(self):
        files_NM = get_filesNM_from_NMNodes(self._projectionList)
        photopeak_idx, upper_window_idx, lower_window_idx = self._get_photopeak_scatter_idxs(files_NM[0])
        recon_volume_node = self.logic.reconstruct( 
            files_NM = files_NM,
            attenuation_toggle = self.ui.attenuation_toggle.checked,
            CT_node = self.ui.attenuationdata.currentNode(),
            psf_toggle = self.ui.psf_toggle.checked,
            collimator_code = self.ui.spect_collimator_combobox.currentText, 
            intrinsic_resolution = self.ui.IntrinsicResolutionSpinBox.value,
            index_peak = photopeak_idx, 
            index_upper = upper_window_idx,
            index_lower = lower_window_idx,
            algorithm_name = self.ui.algorithm_selector_combobox.currentText,
            prior_type = self.ui.priorFunctionSelector.currentText,
            prior_beta = self.ui.priorBetaSpinBox.value,
            prior_delta = self.ui.priorDeltaSpinBox.value,
            prior_gamma = self.ui.priorGammaSpinBox.value,
            use_prior_image= self.ui.usePriorAnatomicalCheckBox.checked,
            prior_anatomy_image_node = self.ui.anatomyPriorImageNode.currentNode(),
            N_prior_anatomy_nearest_neighbours = self.ui.nearestNeighboursSpinBox.value,
            n_iters = self.ui.osem_iterations_spinbox.value, 
            n_subsets = self.ui.osem_subsets_spinbox.value,
            store_recons= self.ui.storeItersCheckBox.checked
        )
        print(recon_volume_node.GetID())
        self.logic.DisplayVolume(recon_volume_node)
        
    def onComputeUncertaintyButton(self):
        # Compute uncertainties
        recon_image_node = self.ui.UncImageMRMLNodeComboBox.currentNode()
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.ui.UncImageSegmentSelectorWidget.currentNode(),
            self.ui.UncImageSegmentSelectorWidget.currentSegmentID(),
            recon_image_node
        )
        uncertainty_abs, uncertainty_pct = self.logic.compute_uncertainty(mask, recon_image_node.GetID())
        # Update table
        recon_image_node = self.ui.UncImageMRMLNodeComboBox.currentNode()
        reconstruction_name = recon_image_node.GetName()
        segmentation_name = self.ui.UncImageSegmentSelectorWidget.currentNode().GetName()
        currentSegmentID = self.ui.UncImageSegmentSelectorWidget.currentSegmentID()
        currentSegment = self.ui.UncImageSegmentSelectorWidget.currentNode().GetSegmentation().GetSegment(currentSegmentID)
        mask_name = currentSegment.GetName()
        rowIndex = self.uncertainty_table.AddEmptyRow()
        self.uncertainty_table.GetTable().GetColumn(0).SetValue(rowIndex, reconstruction_name)
        self.uncertainty_table.GetTable().GetColumn(1).SetValue(rowIndex, segmentation_name)
        self.uncertainty_table.GetTable().GetColumn(2).SetValue(rowIndex, mask_name)
        self.uncertainty_table.GetTable().GetColumn(3).SetValue(rowIndex, uncertainty_abs)
        self.uncertainty_table.GetTable().GetColumn(4).SetValue(rowIndex, uncertainty_pct)
        self.uncertainty_table.GetTable().Modified()
        # Show table
        displayTable(self.uncertainty_table)
        
    # -------------------------------------------------
    # ------------ Data Converters --------------------
    # -------------------------------------------------
    
    def saveSIMINDProjections(self, called=None, event=None):
        n_windows = self.ui.simind_nenergy_spinBox.value
        headerfiles = []
        time_per_projection = self.ui.simind_tperproj_doubleSpinBox.value
        scale_factor = self.ui.simind_scale_doubleSpinBox.value
        random_seed = self.ui.simind_randomseed_spinBox.value
        n_windows = self.ui.simind_nenergy_spinBox.value
        for i in range(1,n_windows+1):
            headerfiles.append([getattr(self.ui, f'PathLineEdit_w{i}').currentPath])
        save_path = os.path.join(
            self.ui.simind_projection_folder_PathLineEdit.currentPath,
            self.ui.simind_projections_foldername_lineEdit.text
        )
        patient_name = self.ui.simind_patientname_lineEdit.text
        study_description = self.ui.simind_studydescription_lineEdit.text
        simind2DICOMProjections(
            headerfiles,
            time_per_projection, 
            scale_factor, 
            random_seed, 
            save_path,
            patient_name,
            study_description
        )
        
    def saveSIMINDAmap(self, called=None, event=None):
        save_path = os.path.join(
            self.ui.simindOutputFolderPathLineEdit.currentPath,
            'attenuation_map'
        )
        input_path = self.ui.simindAmapPathLineEdit.currentPath
        patient_name = self.ui.simind_patientname_lineEdit.text
        study_description = 'amap'
        simind2DICOMAmap(
            input_path,
            save_path, 
            patient_name,
            study_description
        )
        
    def changeSIMINDFolderStudyDescription(self, called=None, event=None):
        name = re.sub(r'\s+', '_', self.ui.simind_patientname_lineEdit.text)
        time = self.ui.simind_tperproj_doubleSpinBox.value
        scale = self.ui.simind_scale_doubleSpinBox.value
        random_seed = self.ui.simind_randomseed_spinBox.value
        self.ui.simind_projections_foldername_lineEdit.text = f'{name}_time{time:.0f}_scale{scale:.0f}_seed{random_seed}'
        self.ui.simind_studydescription_lineEdit.text = f'{name}_time{time:.0f}_scale{scale:.0f}_seed{random_seed}'