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
from DICOMLib import DICOMUtils
slicer.util.pip_install("--ignore-requires-python pytomography==3.0.0")
import pytomography
print(pytomography.__version__)
print("I'm here")
import re
import importlib
from Logic.SlicerSPECTReconLogic import SlicerSPECTReconLogic
from Logic.vtkkmrmlutils import *
from Logic import vtkkmrmlutils


__submoduleNames__ = [
    "SlicerSPECTReconLogic",
    "SlicerSPECTReconTest",
    "vtkkmrmlutils"
]

__package__ = "SlicerSPECTRecon"
mod = importlib.import_module("Logic", __name__)
importlib.reload(mod)
__all__ = ["SlicerSPECTRecon", "SlicerSPECTRecon", "SlicerSPECTReconLogic", "SlicerSPECTReconTest"]



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
        self.setupConnections()
        # self.filter_nodes_by_modality('CT')

    def filter_nodes_by_modality(self, comboBox, modality):
        all_nodes = getAllScalarVolumeNodes()
        mod = filterNodesByModality(all_nodes, modality)
        comboBox.removeNode()
        for node in mod:
            comboBox.addNode(node)


    def setupConnections(self):
        self.ui.attenuation_toggle.connect('toggled(bool)', self.hideShowItems)
        self.ui.psf_toggle.connect('toggled(bool)', self.hideShowItems)
        self.ui.scatter_toggle.connect('toggled(bool)', self.hideShowItems)
        self.ui.usePriorAnatomicalCheckBox.connect('toggled(bool)', self.hideShowItems)
        self.ui.algorithm_selector_combobox.connect('currentTextChanged(QString)', self.hideShowItems)
        self.ui.spect_scatter_combobox.connect('currentTextChanged(QString)', self.hideShowItems)
        self.ui.priorFunctionSelector.connect('currentTextChanged(QString)', self.hideShowItems)
        # SIMIND data converter
        self.ui.data_converter_comboBox.connect('currentTextChanged(QString)', self.hideShowItems)
        self.ui.simind_nenergy_spinBox.connect('valueChanged(int)', self.hideShowItems)
        self.ui.simind_patientname_lineEdit.connect('textChanged(QString)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind_tperproj_doubleSpinBox.connect('valueChanged(double)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind_scale_doubleSpinBox.connect('valueChanged(double)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind_randomseed_spinBox.connect('valueChanged(int)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind_poisson_checkBox.connect('toggled(bool)', self.hideShowItems)
        # Update info
        self.filter_nodes_by_modality(self.ui.NM_data_selector, 'NM')
        self.ui.NM_data_selector.connect('checkedNodesChanged()', self.updateParameterNodeFromGUI)
        self.ui.attenuationdata.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
        self.ui.anatomyPriorImageNode.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
        self.ui.spect_collimator_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.spect_scatter_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.photopeak_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.spect_upperwindow_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.spect_lowerwindow_combobox.connect('currentTextChanged(QString)', self.updateParameterNodeFromGUI)
        self.ui.IntrinsicResolutionSpinBox.connect('valueChanged(float)', self.updateParameterNodeFromGUI)
        self.ui.osem_iterations_spinbox.connect('valueChanged(int)', self.updateParameterNodeFromGUI)
        self.ui.osem_subsets_spinbox.connect('valueChanged(int)', self.updateParameterNodeFromGUI)
        self.ui.outputVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
        # Default values
        self.ui.data_converters_CollapsibleButton.checked = False
        self.ui.AttenuationGroupBox.setVisible(self.ui.attenuation_toggle.checked)
        self.ui.PSFGroupBox.setVisible(self.ui.psf_toggle.checked)
        self.ui.ScatterGroupBox.setVisible(self.ui.scatter_toggle.checked)
        self.ui.PriorGroupBox.setVisible(False)
        self.ui.simind2dicom_groupBox.setVisible(False)
        for i in range(2,10):
            getattr(self.ui, f'PathLineEdit_w{i}').setVisible(False)
            getattr(self.ui, f'label_w{i}').setVisible(False)
        # Buttons
        self.ui.osem_reconstruct_pushbutton.connect('clicked(bool)', self.onReconstructButton)
        self.ui.simind_projections_pushButton.connect('clicked(bool)', self.saveSIMINDProjections)
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

    def saveSIMINDProjections(self, called=None, event=None):
        n_windows = self.ui.simind_nenergy_spinBox.value
        headerfiles = []
        time_per_projection = self.ui.simind_tperproj_doubleSpinBox.value
        scale_factor = self.ui.simind_scale_doubleSpinBox.value
        n_windows = self.ui.simind_nenergy_spinBox.value
        for i in range(1,n_windows+1):
            headerfiles.append([getattr(self.ui, f'PathLineEdit_w{i}').currentPath])
        add_noise = self.ui.simind_poisson_checkBox.checked
        save_path = os.path.join(
            self.ui.simind_projection_folder_PathLineEdit.currentPath,
            self.ui.simind_projections_foldername_lineEdit.text
        )
        patient_name = self.ui.simind_patientname_lineEdit.text
        study_description = self.ui.simind_studydescription_lineEdit.text
        self.logic.simind2DICOMProjections(
            headerfiles,
            time_per_projection, 
            scale_factor, 
            add_noise, 
            save_path,
            patient_name,
            study_description
        )

    def saveSIMINDAmp(self, called=None, event=None):
        save_path = os.path.join(
            self.ui.simind_projection_folder_PathLineEdit.currentPath,
            self.ui.simind_projections_foldername_lineEdit.text
        )
        patient_name = self.ui.simind_patientname_lineEdit.text
        study_description = self.ui.simind_studydescription_lineEdit.text

    def changeSIMINDFolderStudyDescription(self, called=None, event=None):
        name = re.sub(r'\s+', '_', self.ui.simind_patientname_lineEdit.text)
        time = self.ui.simind_tperproj_doubleSpinBox.value
        scale = self.ui.simind_scale_doubleSpinBox.value
        random_seed = self.ui.simind_randomseed_spinBox.value if self.ui.simind_poisson_checkBox.checked else 'None'
        self.ui.simind_projections_foldername_lineEdit.text = f'{name}_time{time:.0f}_scale{scale:.0f}_seed{random_seed}'
        self.ui.simind_studydescription_lineEdit.text = f'{name}_time{time:.0f}_scale{scale:.0f}_seed{random_seed}'
            
    def hideShowItems(self, called=None, event=None):
        print(self.ui.attenuation_toggle.checked)
        self.ui.AttenuationGroupBox.setVisible(self.ui.attenuation_toggle.checked)
        self.ui.PSFGroupBox.setVisible(self.ui.psf_toggle.checked)
        self.ui.ScatterGroupBox.setVisible(self.ui.scatter_toggle.checked)
        self.ui.simind2dicom_groupBox.setVisible(self.ui.data_converter_comboBox.currentText=='SIMIND to DICOM')
        self.ui.simind_randomseed_label.setVisible(self.ui.simind_poisson_checkBox.checked)
        self.ui.simind_randomseed_spinBox.setVisible(self.ui.simind_poisson_checkBox.checked)
        # SIMIND2DICOM energy window stuff
        n_windows = self.ui.simind_nenergy_spinBox.value
        for i in range(1,10):
            getattr(self.ui, f'PathLineEdit_w{i}').setVisible(i<=n_windows)
            getattr(self.ui, f'label_w{i}').setVisible(i<=n_windows)
        # Scatter stuff
        if self.ui.spect_scatter_combobox.currentText=='Dual Energy Window':
            self.ui.upperwindowLabel.setVisible(False)
            self.ui.lowerwindowLabel.setVisible(True)
            self.ui.spect_upperwindow_combobox.setVisible(False)
            self.ui.spect_lowerwindow_combobox.setVisible(True)
        elif self.ui.spect_scatter_combobox.currentText=='Triple Energy Window':
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
    
    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._updatingGUIFromParameterNode:
            return
        print('update')
        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True
        inputVolume1 = self._parameterNode.GetNodeReference("InputVolume1")
        if inputVolume1 and self._parameterNode.GetParameter("Photopeak") :
            self.getProjectionData(inputVolume1)
        last_text = {}
        lastUpperWindowSelection = last_text.get(self.ui.spect_upperwindow_combobox.objectName, "None")
        lastLowerWindowSelection = last_text.get(self.ui.spect_lowerwindow_combobox.objectName, "None")
        # Update photopeak
        photopeak_value = self._parameterNode.GetParameter("Photopeak")
        photopeak_index = self.ui.photopeak_combobox.findText(photopeak_value)
        self.ui.photopeak_combobox.setCurrentIndex(photopeak_index)
        last_text[self.ui.photopeak_combobox.objectName] = self.ui.photopeak_combobox.currentText
        print(f'Photopeak: {photopeak_value}')
        print(f'Photopeak Index: {photopeak_index}')
        print(photopeak_index)
        # Attenuation Stuff
        # Scatter Stuff
        if self.ui.scatter_toggle.checked:
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
        self._projectionList = self.ui.NM_data_selector.checkedNodes()
        for counter, node in enumerate(self._projectionList, start=1):
            if node:
                nodeID = node.GetID()
                self._parameterNode.SetNodeReferenceID(f"InputVolume{counter}", nodeID)  
        self._parameterNode.SetNodeReferenceID("AttenuationData", self.ui.attenuationdata.currentNodeID)
        self._parameterNode.SetNodeReferenceID("AnatomyPriorImage", self.ui.anatomyPriorImageNode.currentNodeID)
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
        energy_window,_,_ = self.logic.getEnergyWindow(inputdatapath)
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
        #Scatter
        # Necessarily disable

        if not self.ui.scatter_toggle.checked:
            upper_window_idx = lower_window_idx = None
        elif self.ui.spect_scatter_combobox.currentText=='Dual Energy Window':
            upper_window_idx = None
            lower_window_idx = self.ui.spect_lowerwindow_combobox.currentIndex
        elif self.ui.spect_scatter_combobox.currentText=='Triple Energy Window':
            upper_window_idx = self.ui.spect_upperwindow_combobox.currentIndex
            lower_window_idx = self.ui.spect_lowerwindow_combobox.currentIndex
        recon_array, fileNMpaths= self.logic.reconstruct( 
            NM_nodes = self._projectionList,
            attenuation_toggle = self.ui.attenuation_toggle.checked,
            ct_file = self.ui.attenuationdata.currentNode(),
            psf_toggle = self.ui.psf_toggle.checked,
            collimator = self.ui.spect_collimator_combobox.currentText, 
            intrinsic_resolution = self.ui.IntrinsicResolutionSpinBox.value,
            peak_window_idx = self.ui.photopeak_combobox.currentIndex, 
            upper_window_idx = upper_window_idx,
            lower_window_idx = lower_window_idx,
            algorithm = self.ui.algorithm_selector_combobox.currentText,
            prior_type = self.ui.priorFunctionSelector.currentText,
            prior_beta = self.ui.priorBetaSpinBox.value,
            prior_delta = self.ui.priorDeltaSpinBox.value,
            prior_gamma = self.ui.priorGammaSpinBox.value,
            prior_anatomy_image_file=self.ui.anatomyPriorImageNode.currentNode(),
            N_prior_anatomy_nearest_neighbours = self.ui.nearestNeighboursSpinBox.value,
            iter = self.ui.osem_iterations_spinbox.value, 
            subset = self.ui.osem_subsets_spinbox.value
    )
        reconstructedDCMInstances = self.logic.stitchMultibed(recon_array, fileNMpaths)
        self.logic.saveVolumeInTempDB(reconstructedDCMInstances, self.ui.outputVolumeSelector.currentNode())