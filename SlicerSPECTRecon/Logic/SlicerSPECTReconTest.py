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

class SlicerSPECTReconTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()
        module_dir = os.path.dirname(__file__)
        self.resources_dir = os.path.join(module_dir, './../Resources')

    def runTest(self):
        """Run as few or as many tests as needed here."""
        # self.setUp()
        # self.test_load_projection_data()
        # self.setUp()
        # self.test_projection_metadata()
        # self.setUp()
        # self.test_attenuation_map_alignment()
        self.setUp()
        self.test_psf_metadata()
        # self.test_slicerspectrecon1()
        # self.setUp()
        # self.test_slicerspectrecon2()

    def test_load_projection_data(self):

        self.delayDisplay("Load projection data test started!")
        files_NM = [os.path.join(self.resources_dir, 'DICOM/bed1_projections.dcm'),\
            os.path.join(self.resources_dir, 'DICOM/bed2_projections.dcm')]
        
        file_NM1 = pydicom.read_file(files_NM[0])
        file_NM2 = pydicom.read_file(files_NM[1])
        len_energy_window_NM1 = len(file_NM1.EnergyWindowInformationSequence)
        len_energy_window_NM2 = len(file_NM2.EnergyWindowInformationSequence)
        NM1_img_arr_shape = file_NM1.pixel_array.shape
        NM2_img_arr_shape = file_NM2.pixel_array.shape
        file_NM1_tensor = torch.tensor((file_NM1.pixel_array).astype(np.int16))
        file_NM2_tensor = torch.tensor((file_NM2.pixel_array).astype(np.int16))
        reshape_file_NM1_tensor = file_NM1_tensor.view(len_energy_window_NM1, int(NM1_img_arr_shape[0]/len_energy_window_NM1),
                                                       NM1_img_arr_shape[1], NM1_img_arr_shape[2])
        reshape_file_NM2_tensor = file_NM2_tensor.view(len_energy_window_NM2, int(NM2_img_arr_shape[0]/len_energy_window_NM2),
                                                       NM2_img_arr_shape[1], NM2_img_arr_shape[2])
        stack_files_NM = torch.stack([reshape_file_NM1_tensor, reshape_file_NM2_tensor], dim=0)
        projectionss = dicom.load_multibed_projections(files_NM)

        self.assertEqual(stack_files_NM.shape, projectionss.shape)
        self.delayDisplay("Load projection data test passed!")

    def test_projection_metadata(self):

        self.delayDisplay("Get projection metadata test started!")
        files_NM = [os.path.join(self.resources_dir, 'DICOM/bed1_projections.dcm'),\
            os.path.join(self.resources_dir, 'DICOM/bed2_projections.dcm')]
        with open(os.path.join(self.resources_dir, 'sampleDataMetaData.json'), mode="r", encoding="utf-8") as inputfilemeta:
            metadata = json.load(inputfilemeta)

        energy_window1, _, idx_sorted1 = getEnergyWindow(files_NM[0])
        energy_window2, _, idx_sorted2 = getEnergyWindow(files_NM[1])
        file_NM1 = pydicom.read_file(files_NM[0])
        file_NM2 = pydicom.read_file(files_NM[1])
        len_energy_window_NM1 = len(file_NM1.EnergyWindowInformationSequence)
        len_energy_window_NM2 = len(file_NM2.EnergyWindowInformationSequence)

        photopeak_value = metadata['inputDataMeta']['photopeak_value']
        peak_window_idx1 = energy_window1.index(photopeak_value)
        peak_window_idx2 = energy_window2.index(photopeak_value)
        photopeak1_idx = idx_sorted1[peak_window_idx1]
        photopeak2_idx = idx_sorted2[peak_window_idx2]

        self.assertEqual(len_energy_window_NM1, len(energy_window1))
        self.assertEqual(len_energy_window_NM2, len(energy_window2))
        self.assertEqual(photopeak1_idx, photopeak2_idx)
        self.assertEqual(1, photopeak1_idx)
        self.delayDisplay("Get projection metadata test passed!")

    def test_attenuation_map_alignment(self):

        self.delayDisplay("Attenuation map alignment test started!")
        ct_path = os.path.join(self.resources_dir, "DICOM/CT")
        files_CT = [os.path.join(ct_path, file) for file in os.listdir(ct_path)]
        files_NM = [os.path.join(self.resources_dir, 'DICOM/bed1_projections.dcm'),\
            os.path.join(self.resources_dir, 'DICOM/bed2_projections.dcm')]
        with open(os.path.join(self.resources_dir, 'sampleDataMetaData.json'), mode="r", encoding="utf-8") as inputfilemeta:
            metadata = json.load(inputfilemeta)

        photopeak_value = metadata['inputDataMeta']['photopeak_value']
        energy_window1, _, idx_sorted1 = getEnergyWindow(files_NM[0])
        energy_window2, _, idx_sorted2 = getEnergyWindow(files_NM[1])
        peak_window_idx1 = energy_window1.index(photopeak_value)
        peak_window_idx2 = energy_window2.index(photopeak_value)

        attenuation_map_path = os.path.join(self.resources_dir, "Attenuation_maps")
        attenuation_map_upper = torch.load(os.path.join(attenuation_map_path, "attenuation_map_upper.pt"))
        attenuation_map_lower = torch.load(os.path.join(attenuation_map_path, "attenuation_map_lower.pt"))
        attenuation_map1 = getAttenuationMap(files_CT, files_NM, bed_idx=0,
                                                index_peak=idx_sorted1[peak_window_idx1])
        attenuation_map2 = getAttenuationMap(files_CT, files_NM, bed_idx=1,
                                                index_peak=idx_sorted2[peak_window_idx2])
        mse_upper = torch.sum((attenuation_map_upper.to("cuda")-attenuation_map1)**2)/2
        mse_lower = torch.sum((attenuation_map_lower.to("cuda")-attenuation_map2)**2)/2

        self.assertEqual(mse_upper,0.0)
        self.assertEqual(mse_lower,0.0)
        self.delayDisplay("Attenuation map alignment test passed!")

    def test_psf_metadata(self):

        self.delayDisplay("PSF metadata test started!")
        with open(os.path.join(self.resources_dir, 'psfMeta.json'), mode="r", encoding="utf-8") as inputfilemeta:
            psf_metadata = json.load(inputfilemeta)
        files_NM = [os.path.join(self.resources_dir, 'DICOM/bed1_projections.dcm'),\
            os.path.join(self.resources_dir, 'DICOM/bed2_projections.dcm')]

        _, mean_window_energies1, _ = getEnergyWindow(files_NM[0])
        _, mean_window_energies2, _ = getEnergyWindow(files_NM[1])
        self.assertEqual(mean_window_energies1[1], mean_window_energies2[1])

        peak_window_energy = mean_window_energies1[1]
        print(peak_window_energy)
        collimator = psf_metadata['psfModellingMeta']['collimator']
        intrinsic_resolution = psf_metadata['psfModellingMeta']['intrinsic_resolution']
        sigma_fit_params = psf_metadata['sigma_fit_params']
        psf_meta = getPSFMeta(collimator, peak_window_energy, intrinsic_resolution)
        psf_meta_sigma_fit_params = [float('{:.3f}'.format(param)) for param in psf_meta.sigma_fit_params]

        self.assertEqual(sigma_fit_params, psf_meta_sigma_fit_params)
        self.delayDisplay("PSF metadata test passed!")

    # def test_scatter(self):
    #     with open('./../Resources/sampleDataMetaData.json', mode="r", encoding="utf-8") as inputfilemeta:
    #         metadata = json.load(inputfilemeta)
    #     self.upper_window_idx = energy_window1.index(upper_window_value)
    #     self.lower_window_idx = energy_window1.index(lower_window_value)
    #     upper_window_value = metadata['scatterCorrectionMeta']['upper_window_value']
    #     lower_window_value = metadata['scatterCorrectionMeta']['lower_window_value']
    #     photopeak, scatter = get_photopeak_scatter(bed_idx, files_NM, index_peak, index_lower, index_upper)

    # def recon_algorithms:
    #     pass

    # def test_data_converters:
    #     pass

    # def test_osem(self, algorithm_settings):
    #     algorithm = algorithm_settings['osem']['algorithm']
    #     iterations = algorithm_settings['osem']['iterations']
    #     subsets = algorithm_settings['osem']['subsets']
    #     return (algorithm, iterations, subsets)
    
    # def test_bsrem(self, algorithm_settings):
    #     beta = delta = gamma = nearest_neighbour = None
    #     algorithm = algorithm_settings['bsrem']['algorithm']
    #     iterations = algorithm_settings['bsrem']['iterations']
    #     subsets = algorithm_settings['bsrem']['subsets']
    #     prior_type = algorithm_settings['bsrem']['prior']
    #     if prior_type == "LogCosh":
    #         beta = algorithm_settings['prior_type']['logCosh']['beta']
    #         delta = algorithm_settings['prior_type']['logCosh']['delta']
    #     elif prior_type == "Quadratic":
    #         beta = algorithm_settings['prior_type']['quadratic']['beta']
    #         delta = algorithm_settings['prior_type']['quadratic']['delta']
    #     else:
    #         beta = algorithm_settings['prior_type']['relativeDifferencePenalty']['beta']
    #         gamma = algorithm_settings['prior_type']['relativeDifferencePenalty']['gamma']
    #     use_anatomical_information = json.loads(algorithm_settings['use_anatomical_information'].lower())
    #     if use_anatomical_information:
    #         nearest_neighbour = algorithm_settings['nearest_neighbour']        
    #     return (algorithm, iterations, subsets, prior_type, beta, delta, gamma, use_anatomical_information, nearest_neighbour)
    
    # def test_osmaposl(self, algorithm_settings):
    #     beta = delta = gamma = nearest_neighbour = None
    #     algorithm = algorithm_settings['osmaposl']['algorithm']
    #     iterations = algorithm_settings['osmaposl']['iterations']
    #     subsets = algorithm_settings['osmaposl']['subsets']
    #     prior_type = algorithm_settings['osmaposl']['prior']
    #     use_anatomical_information = json.loads(algorithm_settings['use_anatomical_information'].lower())
    #     if prior_type == "LogCosh":
    #         beta = algorithm_settings['prior_type']['logCosh']['beta']
    #         delta = algorithm_settings['prior_type']['logCosh']['delta']
    #     elif prior_type == "Quadratic":
    #         beta = algorithm_settings['prior_type']['quadratic']['beta']
    #         delta = algorithm_settings['prior_type']['quadratic']['delta']
    #     else:
    #         beta = algorithm_settings['prior_type']['relativeDifferencePenalty']['beta']
    #         gamma = algorithm_settings['prior_type']['relativeDifferencePenalty']['gamma']
    #     if use_anatomical_information:
    #         nearest_neighbour = algorithm_settings['nearest_neighbour']
    #     return (algorithm, iterations, subsets, prior_type, beta, delta, gamma, use_anatomical_information, nearest_neighbour)

    # def test_slicerspectrecon1(self):
    #     """Ideally you should have several levels of tests.  At the lowest level
    #     tests should exercise the functionality of the logic with different inputs
    #     (both valid and invalid).  At higher levels your tests should emulate the
    #     way the user would interact with your code and confirm that it still works
    #     the way you intended.
    #     One of the most important features of the tests is that it should alert other
    #     developers when their changes will have an impact on the behavior of your
    #     module.  For example, if a developer removes a feature that you depend on,
    #     your test should break so they know that the feature is needed.
    #     """



    #     self.delayDisplay("Starting the test")

    #     logic = SlicerSPECTReconLogic()
    #     self.delayDisplay("Test passed")
    #     files_NM = [os.path.join('./../Resources/DICOM', 'bed1_projections.dcm'),\
    #         os.path.join('./../Resources/DICOM', 'bed2_projections.dcm')]
    #     path_CT = os.path.join('./../Resources/DICOM', 'CT')
    #     files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]

    #     with open('./../Resources/sampleDataMetaData.json', mode="r", encoding="utf-8") as inputfilemeta:
    #         metadata = json.load(inputfilemeta)
    #     with open('./../Resources/algorithmTestSettings.json', mode="r", encoding="utf-8") as settingsfile:
    #         algorithm_settings = json.load(settingsfile) 

    #     photopeak_value = metadata['inputDataMeta']['photopeak_value']
    #     upper_window_value = metadata['scatterCorrectionMeta']['upper_window_value']
    #     lower_window_value = metadata['scatterCorrectionMeta']['lower_window_value']
    #     collimator = metadata['psfModellingMeta']['collimator']
    #     intrinsic_resolution = metadata['psfModellingMeta']['intrinsic_resolution']
    #     psf_toggle = json.loads(metadata['toggle']['psf_toggle'].lower())
    #     attenuation_toggle = json.loads(metadata['toggle']['attenuation_toggle'].lower())

    #     energy_window,_,_ = getEnergyWindow(files_NM[0])
    #     peak_window_idx = energy_window.index(photopeak_value)
    #     upper_window_idx = energy_window.index(upper_window_value)
    #     lower_window_idx = energy_window.index(lower_window_value)

    #     test_bsrem = False
    #     test_osmaposl = False

    #     if test_bsrem:
    #         algorithm, iter, subset, prior_type,\
    #         prior_beta, prior_delta, prior_gamma,\
    #         use_anatomical_information, N_prior_anatomy_nearest_neighbours = self.test_bsrem(algorithm_settings)
    #     elif test_osmaposl:
    #         algorithm, iter, subset, prior_type,\
    #         prior_beta, prior_delta, prior_gamma,\
    #         use_anatomical_information, N_prior_anatomy_nearest_neighbours = self.test_osmaposl(algorithm_settings)
    #     else:
    #         algorithm, iter, subset = self.test_osem(algorithm_settings)
    #     if use_anatomical_information:
    #         pass
    #     #TODO: Get sample image file
    #     prior_anatomy_image_file=None
    #     recon_array, fileNMpaths = logic.reconstruct(files_NM, attenuation_toggle, files_CT, psf_toggle, 
    #                                                 collimator, intrinsic_resolution,peak_window_idx, 
    #                                                 upper_window_idx, lower_window_idx, algorithm, prior_type, 
    #                                                 prior_beta, prior_delta, prior_gamma, prior_anatomy_image_file,
    #                                                 N_prior_anatomy_nearest_neighbours, iter, subset)
    #     reconstructedDCMInstances = self.logic.stitchMultibed(recon_array, fileNMpaths)
    #     if reconstructedDCMInstances:
    #         self.delayDisplay("Test passed")

    # def simind_to_dicom_conversionTest(self):
    #     pass