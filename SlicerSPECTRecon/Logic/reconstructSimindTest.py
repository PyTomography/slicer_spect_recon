import logging
import os
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from DICOMLib import DICOMUtils
import pytomography
print(pytomography.__version__)
print("I'm here")
from pytomography.io.SPECT import dicom
import numpy as np
import pydicom
import torch
from Logic.SlicerSPECTReconLogic import SlicerSPECTReconLogic
from Logic.volumeutils import *
from Logic.likelihood import *
from Logic.algorithms import *
from Logic.systemMatrix import *
from Logic.priors import *
from Logic.transforms import *
from Logic.vtkkmrmlutils import *
from Logic.getmetadatautils import *
from Logic.simindToDicom import *
import json
from Logic.testutils_builder import *

class reconstructSimindTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        self.cleanUP()
        logger.info("SlicerSPECTRecon.reconstructSimindTest")
        module_dir = os.path.dirname(__file__)
        self.resources_dir = os.path.join(module_dir, './../Resources')
        self.logic = SlicerSPECTReconLogic()
        # try:
        #     self.delayDisplay("Initializing Dicom Database")
        #     initDICOMDatabase()
        # except:
        #     raise

    def cleanUP(self):
        slicer.mrmlScene.Clear(0)
        slicer.app.processEvents()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_simindToDicom_conversion()
        self.cleanUP()

    def test_simindToDicom_conversion(self):

        time_per_projection = 15.0
        scale = 1000.00
        add_noise = True
        random_seed = 0
        patient_name = "lu177_jaszak_test"
        study_description = f'{patient_name}_time{time_per_projection:.0f}_scale{scale:.0f}_seed{random_seed}'
        save_path = Path(slicer.app.temporaryPath)/"test_simind_dicom_file"

        if save_path.exists() and save_path.is_dir():
            for file in save_path.iterdir():
                if file.is_file():
                    file.unlink()
        self.delayDisplay("Downloading test jaszak simind data")
        simind_data_path = Path(slicer.app.temporaryPath) /"jaszak_phantom_simind_data"
        simind_data_path.mkdir(parents=True, exist_ok=True)
        file_id = "1L8NoXEA6AOekyKta6S0PwiB41XZYPK0d"
        zip_path = simind_data_path / '208keV_ME_jaszak.zip'
        status = download_file_from_google_drive(file_id, zip_path)
        if status:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(simind_data_path)
        else:
            logger.error(f"Failed to download jaszak simind data")
        headerfile_1 = [(simind_data_path/"208keV_ME_jaszak"/"tot_w1.h00").as_posix()]
        headerfile_2 = [(simind_data_path/"208keV_ME_jaszak"/"tot_w2.h00").as_posix()]
        headerfile_3 = [(simind_data_path/"208keV_ME_jaszak"/"tot_w3.h00").as_posix()]
        headerfiles = [headerfile_1, headerfile_2, headerfile_3]
        print(headerfiles)
        simind2DICOMProjections(headerfiles, time_per_projection, scale, add_noise, save_path, patient_name, study_description)
        if save_path.exists() and save_path.is_dir():
            file = next(save_path.iterdir())
            # file_name = file.name
            full_file_path = file.resolve()
        jaszak_simind_to_dicom_file = pydicom.read_file(full_file_path)


        self.delayDisplay("Downloading test jaszak dicom data")
        dicom_data_path = Path(slicer.app.temporaryPath) /"jaszak_phantom_dicom_data"
        dicom_data_path.mkdir(parents=True, exist_ok=True)
        
        file_id = "1PJ1mUsSmKkzKBkUXHJq776jE4TW48tyY"
        zip_path = dicom_data_path / 'test_jaszak_dicom.zip'
        status = download_file_from_google_drive(file_id, zip_path)
        if status:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dicom_data_path)
        else:
            logger.error(f"Failed to download jaszak dicom data")

        dicom_path = dicom_data_path/"test_jaszak_dicom"/"dicom"/"lu177_jaszak_test_time15_scale1000_seed0"
        if dicom_path.exists() and dicom_path.is_dir():
            file = next(dicom_path.iterdir())
            # file_name = file.name
            dicom_full_file_path = file.resolve()

        jaszak_dicom_file = pydicom.read_file(dicom_full_file_path)
        print((jaszak_dicom_file.pixel_array).shape)
        mse_error = np.mean((jaszak_simind_to_dicom_file.pixel_array-jaszak_dicom_file.pixel_array)**2)
        print(mse_error)
        self.assertTrue(mse_error<0.05)
        # self.cleanUP()
        self.delayDisplay("Load projection data test passed!")
        



    