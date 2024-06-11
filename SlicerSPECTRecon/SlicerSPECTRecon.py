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
from pytomography.io.SPECT import dicom, simind
from pytomography.io.shared import dicom_creation
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM, BSREM, OSMAPOSL
from pytomography.priors import RelativeDifferencePrior, QuadraticPrior, LogCoshPrior, TopNAnatomyNeighbourWeight
import numpy as np
import pydicom
import torch
import tempfile
import shutil
import re
import copy
from pathlib import Path
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from datetime import datetime

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

    def simind2DICOMProjections(
            self,
            headerfiles, 
            time_per_projection, 
            scale, 
            add_noise, 
            save_path, 
            patient_name, 
            study_description
        ):
        object_meta, proj_meta = simind.get_metadata(headerfiles[0][0])
        dr, dz = proj_meta.dr
        Nproj, Nr, Nz = proj_meta.shape
        Nenergy = len(headerfiles)
        # Store lower/upper bound for each projection
        lowers = []
        uppers = []
        for i in range(Nenergy):
            with open(headerfiles[i][0]) as f:
                headerdata = f.readlines()
            headerdata = np.array(headerdata)
            lwr = simind.get_header_value(headerdata, 'energy window lower level', np.float32)
            upr = simind.get_header_value(headerdata, 'energy window upper level', np.float32)
            lowers.append(lwr)
            uppers.append(upr)
        # Create DICOM
        SOP_instance_UID = dicom_creation.generate_uid()
        SOP_class_UID = '1.2.840.10008.5.1.4.1.1.20' # NM data storage
        ds = dicom_creation.generate_base_dataset(SOP_instance_UID, SOP_class_UID)
        # required by DICOM standard
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.InstanceCreationDate = datetime.today().strftime("%Y%m%d")
        ds.InstanceCreationTime = datetime.today().strftime("%H%M%S.%f")
        ds.Manufacturer = "PyTomography"
        ds.ManufacturerModelName = f"PyTomography {pytomography.__version__}"
        ds.InstitutionName = "UBC"
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.ApprovalStatus = "UNAPPROVED"
        # date stuff
        ds.StudyDate = datetime.today().strftime('%Y%m%d')
        ds.StudyTime = datetime.today().strftime('%H%M%S.%f')
        ds.SeriesTime = datetime.today().strftime('%H%M%S.%f')
        ds.StudyDescription = study_description
        ds.SeriesDescription = study_description
        ds.StudyInstanceUID = dicom_creation.generate_uid()
        ds.SeriesInstanceUID = dicom_creation.generate_uid()
        ds.StudyID = ''
        ds.SeriesNumber = '1'
        ds.InstanceNumber = None
        ds.FrameofReferenceUID = ''
        # patient
        ds.PatientName = patient_name
        ds.PatientID = '001'
        # spacing
        ds.PixelSpacing = [dz, dr]
        ds.Rows = Nz
        ds.Columns = Nr
        ds.ImageOrientationPatient = [1,0,0,0,1,0]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.NumberOfFrames = Nenergy * Nproj
        energy_window_range_sequence = []
        for i in range(Nenergy):
            energy_window_information_sequence_element = Dataset()
            block = Dataset()
            block.EnergyWindowLowerLimit = lowers[i]
            block.EnergyWindowUpperLimit = uppers[i]
            energy_window_information_sequence_element.EnergyWindowRangeSequence = Sequence([Dataset(block)])
            energy_window_information_sequence_element.EnergyWindowName = f'Window{i+1}'
            energy_window_range_sequence.append(energy_window_information_sequence_element)
        ds.EnergyWindowInformationSequence = Sequence(energy_window_range_sequence)
        energy_window_vector = []
        for i in range(Nenergy):
            energy_window_vector += Nproj*[i+1]
        ds.EnergyWindowVector = energy_window_vector
        ds.NumberOfEnergyWindows = Nenergy
        ds.DetectorVector = Nproj*Nenergy*[1]
        ds.RotationVector = Nproj*Nenergy*[1]
        ds.AngularViewVector = Nproj*Nenergy*[1]
        ds.TypeOfDetectorMotion = 'STEP AND SHOOT'
        detector_information = Dataset()
        detector_information.CollimatorGridName = 'asd'
        detector_information.CollimatorType = 'PARA'
        detector_information.ImagePositionPatient = [-(Nr-1)/2*dr, -(Nr-1)/2*dr, Nz*dz]
        detector_information.ImageOrientationPatient = [1,0,0,0,0,-1]
        ds.DetectorInformationSequence = Sequence([detector_information])
        rotation_information = Dataset()
        rotation_information.TableHeight = 0
        rotation_information.TableTraverse = 0
        rotation_information.RotationDirection = 'CCW' #CHAGE THIS
        radius = simind.get_header_value(headerdata, 'Radius', np.float32)
        rotation_information.RadialPosition = Nproj*[radius]
        extent_of_rotation = simind.get_header_value(headerdata, 'extent of rotation', np.float32)
        start_angle = simind.get_header_value(headerdata, 'start angle', np.float32)
        rotation_information.ScanArc = extent_of_rotation
        rotation_information.StartAngle = start_angle + 180
        rotation_information.NumberOfFramesInRotation = Nproj
        rotation_information.AngularStep = extent_of_rotation / Nproj
        rotation_information.ActualFrameDuration = time_per_projection * 1000
        ds.RotationInformationSequence = Sequence([rotation_information])
        projections = simind.get_projections(headerfiles)
        if Nenergy==1:
            projections = projections.unsqueeze(0) # first dimension removed by default in pytomography
        projections_realization = np.random.poisson(projections.cpu().numpy()*time_per_projection*scale)
        projections_realization = np.transpose(projections_realization, (0,1,3,2))[:,:,::-1]
        ds.PixelData = projections_realization.astype(np.uint16).tobytes()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        ds.save_as(os.path.join(save_path, f'{ds.SeriesInstanceUID}.dcm'))
        dicomBrowser = slicer.modules.DICOMWidget.browserWidget.dicomBrowser
        dicomBrowser.importDirectory(save_path, dicomBrowser.ImportDirectoryAddLink)
        dicomBrowser.waitForImportFinished()
        # Create temp database so files can be loaded in viewer
        loadedNodeIDs = []
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(save_path, db)
            patientUIDs = db.patients()
            for patientUID in patientUIDs:
                loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

    def simind2DICOMAmap(
            self,
            amap_file,
            save_path,
            patient_name,
            study_description):
        amap = simind.get_attenuation_map(amap_file)
        scale_factor = (2**16 - 1) / amap.max()
        amap *= scale_factor #maximum dynamic range
        amap = amap.cpu().numpy().round().astype(np.uint16).transpose((2,1,0))
        with open(amap_file) as f:
            headerdata = np.array(f.readlines())
        dx = simind.get_header_value(headerdata, 'scaling factor (mm/pixel) [1]') / 10
        dy = simind.get_header_value(headerdata, 'scaling factor (mm/pixel) [2]') / 10
        dz = simind.get_header_value(headerdata, 'scaling factor (mm/pixel) [3]') / 10
        Nz, Ny, Nx = amap.shape

        # Create and save DICOM file
        Path(save_path).resolve().mkdir(parents=True, exist_ok=False)
        SOP_instance_UID = dicom_creation.generate_uid()
        SOP_class_UID = '1.2.840.10008.5.1.4.1.1.2' # CT
        ds = dicom_creation.generate_base_dataset(SOP_instance_UID, SOP_class_UID)
        # required by DICOM standard
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.InstanceCreationDate = datetime.today().strftime("%Y%m%d")
        ds.InstanceCreationTime = datetime.today().strftime("%H%M%S.%f")
        ds.Manufacturer = "PyTomography"
        ds.ManufacturerModelName = f"PyTomography {pytomography.__version__}"
        ds.InstitutionName = "UBC"
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.ApprovalStatus = "UNAPPROVED"
        # date stuff
        ds.StudyDate = datetime.today().strftime('%Y%m%d')
        ds.StudyTime = datetime.today().strftime('%H%M%S.%f')
        ds.SeriesTime = datetime.today().strftime('%H%M%S.%f')
        ds.StudyDescription = study_description
        ds.SeriesDescription = study_description
        ds.StudyInstanceUID = dicom_creation.generate_uid()
        ds.SeriesInstanceUID = dicom_creation.generate_uid()
        # patient
        ds.PatientName = patient_name
        ds.PatientID = '001'
        # image
        ds.RescaleSlope = 1/scale_factor
        ds.Rows = amap.shape[1]
        ds.Columns = amap.shape[2]
        ds.PixelSpacing = [dx, dy]
        ds.SliceThickness = dz
        ds.SpacingBetweenSlices = dz
        ds.ImageOrientationPatient = [1,0,0,0,1,0]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        dss = [] 
        for i in range(amap.shape[0]):
            ds_i = copy.deepcopy(ds)
            ds_i.InstanceNumber = i + 1
            ds_i.ImagePositionPatient = [-(Nx-1)/2*dx, -(Ny-1)/2*dy, i * dz]
            ds_i.SOPInstanceUID = f"{ds.SOPInstanceUID[:-3]}{i+1:03d}"
            ds_i.file_meta.MediaStorageSOPInstanceUID = ds_i.SOPInstanceUID
            ds_i.PixelData = amap[i].tobytes()
            dss.append(ds_i)  
        for ds_i in dss:
            ds_i.save_as(os.path.join(save_path, f'{ds_i.SOPInstanceUID}.dcm'))


    def getEnergyWindow(self, directory):
        # Import
        import numpy as np
        import pydicom
        # Implementation
        ds = pydicom.read_file(directory)
        window_names =[]
        mean_window_energies = []
        for energy_window_information in ds.EnergyWindowInformationSequence:
            lower_limit = energy_window_information.EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
            upper_limit = energy_window_information.EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
            energy_window_name = 'blank'#energy_window_information.EnergyWindowName
            mean_window_energies.append((lower_limit+upper_limit)/2)
            window_names.append(f'{energy_window_name} ({lower_limit:.2f}keV - {upper_limit:.2f}keV)')
        idx_sorted = np.argsort(mean_window_energies)
        window_names = list(np.array(window_names)[idx_sorted])
        mean_window_energies = list(np.array(mean_window_energies)[idx_sorted])
        return window_names, mean_window_energies, idx_sorted

    def pathFromNode(self, node):
        #TODO: Review this function to handle the case where the data was dragged and dropped
        if node is not None:
            storageNode = node.GetStorageNode()
            if storageNode is not None: # loaded via drag-drop
                filepath = storageNode.GetFullNameFromFileName()
            else: # Loaded via DICOM browser
                instanceUIDs = node.GetAttribute("DICOM.instanceUIDs").split()
                filepath = slicer.dicomDatabase.fileForInstance(instanceUIDs[0])
        return filepath
    
    def filesFromNode(self, node):
        #TODO: Review this function to handle the case where the data was dragged and dropped
        if node is not None:
            storageNode = node.GetStorageNode()
            if storageNode is not None: # loaded via drag-drop
                filepaths = storageNode.GetFullNameFromFileName()
            else: # Loaded via DICOM browser
                instanceUIDs = node.GetAttribute("DICOM.instanceUIDs").split()
                filepaths = [slicer.dicomDatabase.fileForInstance(instanceUID) for instanceUID in instanceUIDs]
            return filepaths
        else:
            return None

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

    def get_filesNM_from_NMNodes(self, NM_nodes):
        files_NM = []
        for NM_node in NM_nodes:
            path = self.pathFromNode(NM_node)
            files_NM.append(path)
        return files_NM

    def get_metadata_photopeak_scatter(self, bed_idx, files_NM, index_peak, index_lower=None, index_upper=None):
        file_NM = files_NM[bed_idx]
        object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak)
        projectionss = dicom.load_multibed_projections(files_NM)
        photopeak = projectionss[bed_idx][index_peak]
        # No scatter
        if (index_lower is None)*(index_upper is None):
            scatter = None
        # Dual or triple energy window
        else:
            scatter = dicom.get_energy_window_scatter_estimate_projections(file_NM, projectionss[bed_idx], index_peak, index_lower, index_upper)
        return object_meta, proj_meta, photopeak, scatter

    def reconstruct(
        self,
        NM_nodes,
        attenuation_toggle,
        ct_file,
        psf_toggle,
        collimator,
        intrinsic_resolution,
        peak_window_idx, 
        upper_window_idx,
        lower_window_idx,
        algorithm,
        prior_type,
        prior_beta,
        prior_delta,
        prior_gamma,
        prior_anatomy_image_file,
        N_prior_anatomy_nearest_neighbours,
        iter,
        subset
    ): 
        
        # Get data/metadata
        files_NM = self.get_filesNM_from_NMNodes(NM_nodes)
        _ , mean_window_energies, idx_sorted = self.getEnergyWindow(files_NM[0])
        index_peak = idx_sorted[peak_window_idx]
        index_upper = idx_sorted[upper_window_idx] if upper_window_idx is not None else None
        index_lower = idx_sorted[lower_window_idx] if lower_window_idx is not None else None
        print(index_upper)
        print(index_lower)
        # Loop over and reconstruct all bed positions
        recon_array = []
        for bed_idx in range(len(files_NM)):
            object_meta, proj_meta, photopeak, scatter = self.get_metadata_photopeak_scatter(bed_idx, files_NM, index_peak, index_lower, index_upper)
            # Transforms used for system modeling
            obj2obj_transforms = []
            if attenuation_toggle:
                files_CT = self.filesFromNode(ct_file)
                attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT, files_NM[bed_idx], index_peak)
                att_transform = SPECTAttenuationTransform(attenuation_map)
                obj2obj_transforms.append(att_transform)
            if psf_toggle:
                peak_window_energy = mean_window_energies[index_peak]
                psf_meta = dicom.get_psfmeta_from_scanner_params(collimator, peak_window_energy, intrinsic_resolution=intrinsic_resolution)
                psf_transform = SPECTPSFTransform(psf_meta)
                obj2obj_transforms.append(psf_transform)
            # Build system matrix
            system_matrix = SPECTSystemMatrix(
                obj2obj_transforms = obj2obj_transforms,
                proj2proj_transforms = [],
                object_meta = object_meta,
                proj_meta = proj_meta)
            # Build likelihood
            likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)
            if prior_type=='None':
                prior = None
            else:
                if prior_anatomy_image_file is not None:
                    files_CT = self.filesFromNode(prior_anatomy_image_file)
                    prior_anatomy_image = dicom.get_attenuation_map_from_CT_slices(files_CT, files_NM[bed_idx], keep_as_HU=True)
                    prior_weight = TopNAnatomyNeighbourWeight(prior_anatomy_image, N_neighbours=N_prior_anatomy_nearest_neighbours)
                else:
                    prior_weight = None
                # Now determine prior
                if prior_type=='RelativeDifferencePenalty':
                    prior = RelativeDifferencePrior(beta=prior_beta, gamma=prior_gamma, weight=prior_weight)
                elif prior_type=='Quadratic':
                    prior = QuadraticPrior(beta=prior_beta, delta=prior_delta, weight=prior_weight)
                elif prior_type=='LogCosh':
                    prior = LogCoshPrior(beta=prior_beta, delta=prior_delta, weight=prior_weight)
            # Build algorithm
            if algorithm == "OSEM":
                reconstruction_algorithm = OSEM(likelihood)
            elif algorithm == "BSREM":
                reconstruction_algorithm = BSREM(likelihood, prior=prior)
            elif algorithm == "OSMAPOSL":
                reconstruction_algorithm = OSMAPOSL(likelihood, prior=prior)
            # Reconstruct
            reconstructed_object = reconstruction_algorithm(n_iters=iter, n_subsets=subset)
            recon_array.append(reconstructed_object)
        return recon_array, files_NM

    def stitchMultibed(self, recon_array, fileNMpaths, outputVolume):
        if len(fileNMpaths)>1:
            # Get top bed position
            dss = np.array([pydicom.read_file(file_NM) for file_NM in fileNMpaths])
            zs = np.array(
                [ds.DetectorInformationSequence[0].ImagePositionPatient[-1] for ds in dss]
            )
            order = np.argsort(zs)
            print(torch.stack(recon_array).shape)
            recon_stitched = dicom.stitch_multibed(recons=torch.stack(recon_array), files_NM = fileNMpaths)
            fileNMpath_save = fileNMpaths[order[-1]]
        else:
            recon_stitched = recon_array[0]
            fileNMpath_save = fileNMpaths[0]
        reconstructedDCMInstances = dicom.save_dcm(save_path = None, object = recon_stitched, 
                                                   file_NM = fileNMpath_save, recon_name = 'slicer_recon', return_ds =True)
        temp_dir = tempfile.mkdtemp()
        for i, dataset in enumerate(reconstructedDCMInstances):
            temp_file_path = os.path.join(temp_dir, f"temp_{i}.dcm")
            dataset.save_as(temp_file_path)
        # Add saved files to DICOM browser
        dicomBrowser = slicer.modules.DICOMWidget.browserWidget.dicomBrowser
        dicomBrowser.importDirectory(temp_file_path, dicomBrowser.ImportDirectoryAddLink)
        dicomBrowser.waitForImportFinished()
        # Create temp database so files can be loaded in viewer
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
