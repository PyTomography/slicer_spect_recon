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
slicer.util.pip_install("--ignore-requires-python pytomography==3.2.2")
slicer.util.pip_install("beautifulsoup4")
import pytomography
print(pytomography.__version__)
import re
import importlib
from Logic.SlicerSPECTReconLogic import SlicerSPECTReconLogic
from Logic.SlicerSPECTReconTest import SlicerSPECTReconTest
from Logic.vtkkmrmlutils import *
from Logic.getmetadatautils import *
from Logic.simindToDicom import *
from Logic.reconstructSimindTest import reconstructSimindTest

__submoduleNames__ = [
    "SlicerSPECTReconLogic",
    "SlicerSPECTReconTest",
    "vtkkmrmlutils",
    "reconstructSimindTest",
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
            This is an example of scripted loadable module bundled in an extension.
            See more information in <a href="https://github.com/organization/projectname#pytomography">module documentation</a>.
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

    def runTest(self, msec=100, **kwargs):
            """
            :param msec: delay to associate with :func:`ScriptedLoadableModuleTest.delayDisplay()`.
            """
            logging.info("\n******* Starting Tests of SlicerSPECTRecon **********\n")
            # Test reconstructSimind
            testCase = reconstructSimindTest()
            testCase.messageDelay = msec
            testCase.runTest(**kwargs)
            # Test SlicerSPECTReconTest
            # name of the test case class is expected to be <ModuleName>Test
            module = importlib.import_module(self.__module__)
            className = self.moduleName + "Test"
            try:
                TestCaseClass = getattr(module, className)
            except AttributeError:
                # Treat missing test case class as a failure; provide useful error message
                raise AssertionError(
                    f"Test case class not found: {self.__module__}.{className} "
                )
            testCase = TestCaseClass()
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
        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.setupConnections()
        # initialize for loading data in case

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
        self.ui.AttenuationGroupBox.setVisible(self.ui.attenuation_toggle.checked)
        self.ui.PSFGroupBox.setVisible(self.ui.psf_toggle.checked)
        self.ui.ScatterGroupBox.setVisible(self.ui.scatter_toggle.checked)
        self.ui.PriorGroupBox.setVisible(False)
        # Buttons
        self.ui.osem_reconstruct_pushbutton.connect('clicked(bool)', self.onReconstructButton)
        # Data converters
        self.ui.data_converter_comboBox.connect('currentTextChanged(QString)', self.hideShowItems)
        self.ui.simind_nenergy_spinBox.connect('valueChanged(int)', self.hideShowItems)
        self.ui.simind_patientname_lineEdit.connect('textChanged(QString)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind_tperproj_doubleSpinBox.connect('valueChanged(double)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind_scale_doubleSpinBox.connect('valueChanged(double)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind_randomseed_spinBox.connect('valueChanged(int)', self.changeSIMINDFolderStudyDescription)
        self.ui.simind2dicom_groupBox.setVisible(False)
        for i in range(2,10):
            getattr(self.ui, f'PathLineEdit_w{i}').setVisible(False)
            getattr(self.ui, f'label_w{i}').setVisible(False)
        self.ui.simind_projections_pushButton.connect('clicked(bool)', self.saveSIMINDProjections)
        self.ui.simindSaveAmapPushButton.connect('clicked(bool)', self.saveSIMINDAmap)
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
        self.ui.PSFGroupBox.setVisible(self.ui.psf_toggle.checked)
        self.ui.ScatterGroupBox.setVisible(self.ui.scatter_toggle.checked)
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
        # Data converters
        self.ui.simind2dicom_groupBox.setVisible(self.ui.data_converter_comboBox.currentText=='SIMIND to DICOM')
        n_windows = self.ui.simind_nenergy_spinBox.value
        for i in range(1,10):
            getattr(self.ui, f'PathLineEdit_w{i}').setVisible(i<=n_windows)
            getattr(self.ui, f'label_w{i}').setVisible(i<=n_windows)
        
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
        inputdatapath = pathFromNode(node)
        energy_window,_,_ = getEnergyWindow(inputdatapath)
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
            files_NM = get_filesNM_from_NMNodes(self._projectionList),
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