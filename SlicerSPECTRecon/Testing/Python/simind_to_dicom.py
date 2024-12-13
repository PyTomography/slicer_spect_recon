import os
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from pytomography.io.SPECT import dicom
import numpy as np
import pydicom
import zipfile
import logging
from Logic.SlicerSPECTReconLogic import SlicerSPECTReconLogic
from Logic.SimindToDicom import *
from Testing.Python.utils import *

class SimindToDicomConverterTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        self.cleanUP()
        logger.info("SlicerSPECTRecon.simindToDicomConverterTest")
        module_dir = os.path.dirname(__file__)
        self.resources_dir = os.path.join(module_dir, './../../Resources')
        self.logic = SlicerSPECTReconLogic()

    def cleanUP(self):
        slicer.mrmlScene.Clear(0)
        slicer.app.processEvents()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_simindToDicom_conversion()
        self.cleanUP()

    def test_simindToDicom_conversion(self):
        ref_url = "https://zenodo.org/records/14190900/files/refs.zip?download=1"
        test_url = "https://zenodo.org/records/14172228/files/for_slicer_recon.zip?download=1"

        time_per_projection = 15.0
        scale = 1000.00
        random_seed = 0
        patient_name = "lu177_jaszak_test"
        study_description = f'{patient_name}_time{time_per_projection:.0f}_scale{scale:.0f}_seed{random_seed}'
        save_path = Path(slicer.app.temporaryPath)/"test_simind_dicom_file"
        if save_path.exists() and save_path.is_dir():
            for file in save_path.iterdir():
                if file.is_file():
                    file.unlink()
        self.delayDisplay("Downloading test jaszak simind data")
        
        self.tstDir = get_data_from_url(test_url, "test_data")
        simind_data_path = self.tstDir/'for_slicer_recon'/'SIMIND'
        headerfile_1 = [(simind_data_path/"tot_w1.h00").as_posix()]
        headerfile_2 = [(simind_data_path/"tot_w2.h00").as_posix()]
        headerfile_3 = [(simind_data_path/"tot_w3.h00").as_posix()]
        headerfiles = [headerfile_1, headerfile_2, headerfile_3]
        simind2DICOMProjections(headerfiles, time_per_projection, scale, random_seed, save_path, patient_name, study_description)
        if save_path.exists() and save_path.is_dir():
            file = next(save_path.iterdir())
            full_file_path = file.resolve()
        jaszak_simind_to_dicom_file = pydicom.read_file(full_file_path)
        ########Getting groundtruth dicom file##################
        self.delayDisplay("Downloading test jaszak dicom data")
        self.refDir = get_data_from_url(ref_url, "ref_data")
        dicom_data_path = self.refDir/'refs'/'simind_data'
        zip_path = dicom_data_path / 'test_jaszak_dicom.zip'
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dicom_data_path)
        dicom_path = dicom_data_path/"test_jaszak_dicom"/"dicom"/"lu177_jaszak_test_time15_scale1000_seed0"
        if dicom_path.exists() and dicom_path.is_dir():
            file = next(dicom_path.iterdir())
            dicom_full_file_path = file.resolve()
        jaszak_dicom_file = pydicom.read_file(dicom_full_file_path)
        mse_error = np.mean((jaszak_simind_to_dicom_file.pixel_array-jaszak_dicom_file.pixel_array)**2)
        self.assertTrue(mse_error<0.05)
        self.cleanUP()
        self.delayDisplay("Simind conversion to dicom test passed!")