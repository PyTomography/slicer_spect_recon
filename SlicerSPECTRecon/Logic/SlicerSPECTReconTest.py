import os
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from DICOMLib import DICOMUtils
from pytomography.io.SPECT import dicom
import numpy as np
import pydicom
import torch
import ctk
import zipfile
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
from Logic.dicomvalues import DicomValues

class SlicerSPECTReconTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        self.cleanUP()
        logger.info("SlicerSPECTReconTests")
        module_dir = os.path.dirname(__file__)
        self.resources_dir = os.path.join(module_dir, './../Resources')
        self.logic = SlicerSPECTReconLogic()
        try:
            self.delayDisplay("Initializing Dicom Database")
            initDICOMDatabase()
        except:
            raise
        try:
            self.uploadtoDB()
        except:
            raise

    def cleanUP(self):
        slicer.mrmlScene.Clear(0)
        slicer.app.processEvents()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_load_projection_data()
        self.test_projection_metadata()
        self.test_attenuation_map_alignment()
        self.test_psf_metadata()
        self.test_scatter()
        self.run_test_osem()
        self.run_test_bsrem()
        self.run_test_osmaposl()
        self.cleanUP()

    def importDICOMFromURL(self, file_id, folder_name):
        dicomDir = Path(slicer.app.temporaryPath) /folder_name
        dicomDir.mkdir(parents=True, exist_ok=True)
        zip_path = dicomDir / 'dicom.zip'
        logger.info(f"Downloading {folder_name} DICOM files from drive to {dicomDir}")
        status = download_file_from_google_drive(file_id, zip_path)
        if status:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dicomDir)
        else:
            logger.error(f"Failed to download {folder_name} DICOM files")
            return
        indexer = ctk.ctkDICOMIndexer()
        indexer.addDirectory(slicer.dicomDatabase, str(dicomDir))
        indexer.waitForImportFinished()
        logger.info(f"Imported DICOM files from drive to database")

    def downloadAndImportNMFiles(self):
        file_id = "1bCz_hLgASAiQ38QrRlgrJ3lH_lOlqQb1"
        folder_name = "NM"
        self.importDICOMFromURL(file_id, folder_name)

    def downloadAndImportCTFiles(self):
        file_id = "1kbl4nqflovxOB0N9z8YrpmZHzTnPtnyu"
        folder_name = "CT"
        self.importDICOMFromURL(file_id, folder_name)
    
    def uploadtoDB(self):
        dicomValues = DicomValues()
        self.downloadAndImportNMFiles()
        self.downloadAndImportCTFiles()
        nm_series = slicer.dicomDatabase.seriesForStudy(dicomValues.NM_studyInstanceUID)
        nm_series_count = len([uid for uid in nm_series if uid in [dicomValues.NM1_seriesInstanceUID, dicomValues.NM2_seriesInstanceUID]])
        if nm_series_count != 2:
            raise Exception(f"Expected 2 NM series, but found {nm_series_count}")
        ct_series = slicer.dicomDatabase.seriesForStudy(dicomValues.CT_studyInstanceUID)
        ct_series_count = len([uid for uid in ct_series if uid in [dicomValues.CT_seriesInstanceUID]])
        if ct_series_count == 0:
            raise Exception(f"Expected at least 1 CT series, but found {ct_series_count}")
        logger.info(f"Successfully imported {nm_series_count} NM series and {ct_series_count} CT series")

    def test_load_projection_data(self):
        self.delayDisplay("Load projection data test started!")
        dicomValues = DicomValues()
        try:
            NM1_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM1_seriesInstanceUID])
            NM2_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM2_seriesInstanceUID])
            if not NM1_nodeID or not NM2_nodeID:
                raise Exception('Unable to load all series from the database')
        except Exception as e:
            logger.error(f'Could not load dicom files from database: {e}')
            raise
        files_NM_nodes = [getVolumeNode(NM1_nodeID), getVolumeNode(NM2_nodeID)]
        files_NM = [pathFromNode(files_NM_nodes[0]), pathFromNode(files_NM_nodes[1])]
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
        self.cleanUP()
        self.delayDisplay("Load projection data test passed!")

    def test_projection_metadata(self):
        self.delayDisplay("Get projection metadata test started!")
        dicomValues = DicomValues()
        try:
            NM1_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM1_seriesInstanceUID])
            NM2_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM2_seriesInstanceUID])
            if not NM1_nodeID or not NM2_nodeID:
                raise Exception('Unable to load all series from the database')
        except Exception as e:
            logger.error(f'Could not load dicom files from database: {e}')
            raise
        files_NM_nodes = [getVolumeNode(NM1_nodeID), getVolumeNode(NM2_nodeID)]
        files_NM = [pathFromNode(files_NM_nodes[0]), pathFromNode(files_NM_nodes[1])]
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
        self.cleanUP()
        self.delayDisplay("Get projection metadata test passed!")

    def test_attenuation_map_alignment(self):
        self.delayDisplay("Attenuation map alignment test started!")
        dicomValues = DicomValues()
        try:
            NM1_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM1_seriesInstanceUID])
            NM2_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM2_seriesInstanceUID])
            CT_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.CT_seriesInstanceUID])
            if not NM1_nodeID or not NM2_nodeID or not CT_nodeID:
                raise Exception('Unable to load all series from the database')
        except Exception as e:
            logger.error(f'Could not load dicom files from database: {e}')
            raise
        files_NM_nodes = [getVolumeNode(NM1_nodeID), getVolumeNode(NM2_nodeID)]
        files_NM = [pathFromNode(files_NM_nodes[0]), pathFromNode(files_NM_nodes[1])]
        ct_file_node = getVolumeNode(CT_nodeID)
        files_CT = filesFromNode(ct_file_node)
        with open(os.path.join(self.resources_dir, 'sampleDataMetaData.json'), mode="r", encoding="utf-8") as inputfilemeta:
            metadata = json.load(inputfilemeta)
        photopeak_value = metadata['inputDataMeta']['photopeak_value']
        energy_window1, _, idx_sorted1 = getEnergyWindow(files_NM[0])
        energy_window2, _, idx_sorted2 = getEnergyWindow(files_NM[1])
        peak_window_idx1 = energy_window1.index(photopeak_value)
        peak_window_idx2 = energy_window2.index(photopeak_value)
        attenuation_map_path = Path(slicer.app.temporaryPath) /"Attenuation_maps"
        attenuation_map_path.mkdir(parents=True, exist_ok=True)
        self.delayDisplay("Downloading test attenuation map data - lower and upper bed positions")
        lower_att_map_url_id = "1DKn3ixxcRONfmI22jL3WtDEpFcPwhVMp"
        status = download_file_from_google_drive(lower_att_map_url_id, os.path.join(attenuation_map_path/'attenuation_map_lower.pt'))
        if status:
            attenuation_map_lower = torch.load(os.path.join(attenuation_map_path, "attenuation_map_lower.pt"))
        else:
            logger.error(f"Failed to download attenuation_map_lower.pt")
        upper_att_map_url_id = "1RTh2W4j9ranreGDrAF-D0I6skb4eBwWF"
        status = download_file_from_google_drive(upper_att_map_url_id, os.path.join(attenuation_map_path/'attenuation_map_upper.pt'))
        if status:
            attenuation_map_upper = torch.load(os.path.join(attenuation_map_path, "attenuation_map_upper.pt"))
        else:
            logger.error(f"Failed to download attenuation_map_upper.pt")
        attenuation_map1 = getAttenuationMap(files_CT, files_NM, bed_idx=0,
                                                index_peak=idx_sorted1[peak_window_idx1])
        attenuation_map2 = getAttenuationMap(files_CT, files_NM, bed_idx=1,
                                                index_peak=idx_sorted2[peak_window_idx2])
        mse_upper = torch.sum((attenuation_map_upper.to("cuda")-attenuation_map1)**2)/2
        mse_lower = torch.sum((attenuation_map_lower.to("cuda")-attenuation_map2)**2)/2
        self.assertTrue(mse_upper<0.5)
        self.assertTrue(mse_lower<0.5)
        self.cleanUP()
        self.delayDisplay("Attenuation map alignment test passed!")

    def test_psf_metadata(self):
        self.delayDisplay("PSF metadata test started!")
        dicomValues = DicomValues()
        try:
            NM1_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM1_seriesInstanceUID])
            NM2_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM2_seriesInstanceUID])
            if not NM1_nodeID or not NM2_nodeID:
                raise Exception('Unable to load all series from the database')
        except Exception as e:
            logger.error(f'Could not load dicom files from database: {e}')
            raise
        files_NM_nodes = [getVolumeNode(NM1_nodeID), getVolumeNode(NM2_nodeID)]
        files_NM = [pathFromNode(files_NM_nodes[0]), pathFromNode(files_NM_nodes[1])]
        with open(os.path.join(self.resources_dir, 'psfMeta.json'), mode="r", encoding="utf-8") as inputfilemeta:
            psf_metadata = json.load(inputfilemeta)
        _, mean_window_energies1, _ = getEnergyWindow(files_NM[0])
        _, mean_window_energies2, _ = getEnergyWindow(files_NM[1])
        self.assertEqual(mean_window_energies1[1], mean_window_energies2[1])
        peak_window_energy = mean_window_energies1[1]
        collimator = psf_metadata['psfModellingMeta']['collimator']
        intrinsic_resolution = psf_metadata['psfModellingMeta']['intrinsic_resolution']
        sigma_fit_params = np.array(psf_metadata['sigma_fit_params'])
        psf_meta = getPSFMeta(collimator, peak_window_energy, intrinsic_resolution)
        psf_meta_sigma_fit_params = np.array([float('{:.3f}'.format(param)) for param in psf_meta.sigma_fit_params])
        error_margin = np.mean((sigma_fit_params - psf_meta_sigma_fit_params)**2)
        self.assertTrue(error_margin<0.05)
        self.cleanUP()
        self.delayDisplay("PSF metadata test passed!")

    def test_scatter(self):
        self.delayDisplay("Double and triple energy windows scatter data test started!")
        dicomValues = DicomValues()
        try:
            NM1_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM1_seriesInstanceUID])
            NM2_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM2_seriesInstanceUID])
            if not NM1_nodeID or not NM2_nodeID:
                raise Exception('Unable to load all series from the database')
        except Exception as e:
            logger.error(f'Could not load dicom files from database: {e}')
            raise
        files_NM_nodes = [getVolumeNode(NM1_nodeID), getVolumeNode(NM2_nodeID)]
        files_NM = [pathFromNode(files_NM_nodes[0]), pathFromNode(files_NM_nodes[1])]
        with open(os.path.join(self.resources_dir, 'sampleDataMetaData.json'), mode="r", encoding="utf-8") as inputfilemeta:
            metadata = json.load(inputfilemeta)
        photopeak_value = metadata['inputDataMeta']['photopeak_value']
        upper_window_value = metadata['scatterCorrectionMeta']['upper_window_value']
        lower_window_value = metadata['scatterCorrectionMeta']['lower_window_value']
        energy_window1, _, idx_sorted1 = getEnergyWindow(files_NM[0])
        energy_window2, _, idx_sorted2 = getEnergyWindow(files_NM[1])
        photopeak_idx = idx_sorted1[energy_window1.index(photopeak_value)]
        upper_window_idx1 = idx_sorted1[energy_window1.index(upper_window_value)]
        lower_window_idx1 = idx_sorted1[energy_window1.index(lower_window_value)]
        upper_window_idx2 = idx_sorted2[energy_window2.index(upper_window_value)]
        lower_window_idx2 = idx_sorted2[energy_window2.index(lower_window_value)]
        self.assertEqual(upper_window_idx1, upper_window_idx2)
        self.assertEqual(lower_window_idx1, lower_window_idx2)
        scatter_data_path = Path(slicer.app.temporaryPath) /"Test_scatter_data"
        scatter_data_path.mkdir(parents=True, exist_ok=True)
        _,upperbed_scatter_dew = get_photopeak_scatter(0, files_NM, photopeak_idx, lower_window_idx1, None)
        _,lowerbed_scatter_dew = get_photopeak_scatter(1, files_NM, photopeak_idx, lower_window_idx1, None)
        _,upperbed_scatter_tew = get_photopeak_scatter(0, files_NM, photopeak_idx, lower_window_idx1, upper_window_idx1)
        _,lowerbed_scatter_tew = get_photopeak_scatter(1, files_NM, photopeak_idx, lower_window_idx1, upper_window_idx1)
        self.delayDisplay("Downloading test double energy window scatter data")
        ###########Test double energy window scatter#############
        upperbed_scatter_dew_url_id = "1tnT08bYKBVD-H1B8PLVXhligzjCdQCsc"
        status = download_file_from_google_drive(upperbed_scatter_dew_url_id, os.path.join(scatter_data_path/'upperbed_scatter_dew.pt'))
        if status:
            test_upperbed_scatter_dew = torch.load(os.path.join(scatter_data_path, "upperbed_scatter_dew.pt"))
        else:
            logger.error(f"Failed to download upperbed_scatter_dew.pt")
        lowerbed_scatter_dew_url_id = "1RK-zB6BkPUAm8jZ6Ov1HHBhMGwfw933e"
        status = download_file_from_google_drive(lowerbed_scatter_dew_url_id, os.path.join(scatter_data_path/'lowerbed_scatter_dew.pt'))
        if status:
            test_lowerbed_scatter_dew = torch.load(os.path.join(scatter_data_path, "lowerbed_scatter_dew.pt"))
        else:
            logger.error(f"Failed to download lowerbed_scatter_dew.pt")
        mse_upper_dew = torch.sum((test_upperbed_scatter_dew.to("cuda")-upperbed_scatter_dew)**2)/2
        mse_lower_dew = torch.sum((test_lowerbed_scatter_dew.to("cuda")-lowerbed_scatter_dew)**2)/2
        self.assertTrue(mse_upper_dew<0.05)
        self.assertTrue(mse_lower_dew<0.05)
        self.delayDisplay("Downloading test triple energy window scatter data")
        ####################Test triple energy window scatter######################
        upperbed_scatter_tew_url_id = "1DM5Z8Q2-z5_Y8eYr5pwsdysIicMq8b_A"
        status = download_file_from_google_drive(upperbed_scatter_tew_url_id, os.path.join(scatter_data_path/'upperbed_scatter_tew.pt'))
        if status:
            test_upperbed_scatter_tew = torch.load(os.path.join(scatter_data_path, "upperbed_scatter_tew.pt"))
        else:
            logger.error(f"Failed to download upperbed_scatter_tew.pt")
        lowerbed_scatter_tew_url_id = "18znooiTYrjFcbp2JrRndoj9_lfDeUfqM"
        status = download_file_from_google_drive(lowerbed_scatter_tew_url_id, os.path.join(scatter_data_path/'lowerbed_scatter_tew.pt'))
        if status:
            test_lowerbed_scatter_tew = torch.load(os.path.join(scatter_data_path, "lowerbed_scatter_tew.pt"))
        else:
            logger.error(f"Failed to download lowerbed_scatter_tew.pt")
        mse_upper_tew = torch.sum((test_upperbed_scatter_tew.to("cuda")-upperbed_scatter_tew)**2)/2
        mse_lower_tew = torch.sum((test_lowerbed_scatter_tew.to("cuda")-lowerbed_scatter_tew)**2)/2
        self.assertTrue(mse_upper_tew<0.05)
        self.assertTrue(mse_lower_tew<0.05)
        self.cleanUP()
        self.delayDisplay("Double and triple energy windows scatter data test passed!")
     
    def test_osem(self, algorithm_settings):
        algorithm = algorithm_settings['osem']['algorithm']
        iter = algorithm_settings['osem']['iterations']
        subset = algorithm_settings['osem']['subsets']
        prior_beta = prior_delta = prior_gamma = N_prior_anatomy_nearest_neighbours = prior_anatomy_image_file = prior_type = None
        files_NM, attenuation_toggle, ct_file_node, psf_toggle, collimator,\
        intrinsic_resolution, peak_window_idx, upper_window_idx, lower_window_idx = self.get_proj_data()
        recon_array, fileNMpaths = self.logic.reconstruct(files_NM, attenuation_toggle, ct_file_node, psf_toggle, 
                                                    collimator, intrinsic_resolution, peak_window_idx, 
                                                    upper_window_idx, lower_window_idx, algorithm, prior_type, 
                                                    prior_beta, prior_delta, prior_gamma, prior_anatomy_image_file,
                                                    N_prior_anatomy_nearest_neighbours, iter, subset)
        reconstructedDCMInstances = self.logic.stitchMultibed(recon_array, fileNMpaths)
        return reconstructedDCMInstances

    def test_bsrem(self, algorithm_settings, prior_type):
        prior_beta = prior_delta = prior_gamma = N_prior_anatomy_nearest_neighbours = prior_anatomy_image_file = None
        algorithm = algorithm_settings['bsrem']['algorithm']
        iter = algorithm_settings['bsrem']['iterations']
        subset = algorithm_settings['bsrem']['subsets']
        files_NM, attenuation_toggle, ct_file_node, psf_toggle, collimator,\
        intrinsic_resolution, peak_window_idx, upper_window_idx, lower_window_idx = self.get_proj_data()
        if prior_type == "LogCosh":
            prior_beta = algorithm_settings['prior_type']['logCosh']['beta']
            prior_delta = algorithm_settings['prior_type']['logCosh']['delta']
        elif prior_type == "Quadratic":
            prior_beta = algorithm_settings['prior_type']['quadratic']['beta']
            prior_delta = algorithm_settings['prior_type']['quadratic']['delta']
        else:
            prior_beta = algorithm_settings['prior_type']['relativeDifferencePenalty']['beta']
            prior_gamma = algorithm_settings['prior_type']['relativeDifferencePenalty']['gamma']
        use_anatomical_information = json.loads(algorithm_settings['use_anatomical_information'].lower())
        if use_anatomical_information:
            N_prior_anatomy_nearest_neighbours = algorithm_settings['nearest_neighbours']
            prior_anatomy_image_file = ct_file_node
        recon_array, fileNMpaths = self.logic.reconstruct(files_NM, attenuation_toggle, ct_file_node, psf_toggle, 
                                                    collimator, intrinsic_resolution, peak_window_idx, 
                                                    upper_window_idx, lower_window_idx, algorithm, prior_type, 
                                                    prior_beta, prior_delta, prior_gamma, prior_anatomy_image_file,
                                                    N_prior_anatomy_nearest_neighbours, iter, subset)
        reconstructedDCMInstances = self.logic.stitchMultibed(recon_array, fileNMpaths)
        return reconstructedDCMInstances

    def test_osmaposl(self, algorithm_settings, prior_type):
        prior_beta = prior_delta = prior_gamma = N_prior_anatomy_nearest_neighbours = prior_anatomy_image_file = None
        algorithm = algorithm_settings['osmaposl']['algorithm']
        iter = algorithm_settings['osmaposl']['iterations']
        subset = algorithm_settings['osmaposl']['subsets']
        use_anatomical_information = json.loads(algorithm_settings['use_anatomical_information'].lower())
        files_NM, attenuation_toggle, ct_file_node, psf_toggle, collimator,\
        intrinsic_resolution, peak_window_idx, upper_window_idx, lower_window_idx = self.get_proj_data()
        if prior_type == "LogCosh":
            prior_beta = algorithm_settings['prior_type']['logCosh']['beta']
            prior_delta = algorithm_settings['prior_type']['logCosh']['delta']
        elif prior_type == "Quadratic":
            prior_beta = algorithm_settings['prior_type']['quadratic']['beta']
            prior_delta = algorithm_settings['prior_type']['quadratic']['delta']
        else:
            prior_beta = algorithm_settings['prior_type']['relativeDifferencePenalty']['beta']
            prior_gamma = algorithm_settings['prior_type']['relativeDifferencePenalty']['gamma']
        if use_anatomical_information:
            N_prior_anatomy_nearest_neighbours = algorithm_settings['nearest_neighbours']
            prior_anatomy_image_file = ct_file_node
        recon_array, fileNMpaths = self.logic.reconstruct(files_NM, attenuation_toggle, ct_file_node, psf_toggle, 
                                                    collimator, intrinsic_resolution, peak_window_idx, 
                                                    upper_window_idx, lower_window_idx, algorithm, prior_type, 
                                                    prior_beta, prior_delta, prior_gamma, prior_anatomy_image_file,
                                                    N_prior_anatomy_nearest_neighbours, iter, subset)
        reconstructedDCMInstances = self.logic.stitchMultibed(recon_array, fileNMpaths)
        return (reconstructedDCMInstances)

    def get_proj_data(self):
        dicomValues = DicomValues()
        try:
            NM1_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM1_seriesInstanceUID])
            NM2_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.NM2_seriesInstanceUID])
            CT_nodeID = DICOMUtils.loadSeriesByUID([dicomValues.CT_seriesInstanceUID])
            if not NM1_nodeID or not NM2_nodeID or not CT_nodeID:
                raise Exception('Unable to load all series from the database')
        except Exception as e:
            logger.error(f'Could not load dicom files from database: {e}')
            raise
        files_NM_nodes = [getVolumeNode(NM1_nodeID), getVolumeNode(NM2_nodeID)]
        files_NM = [pathFromNode(files_NM_nodes[0]), pathFromNode(files_NM_nodes[1])]
        ct_file_node = getVolumeNode(CT_nodeID)
        with open(os.path.join(self.resources_dir,'sampleDataMetaData.json'), mode="r", encoding="utf-8") as inputfilemeta:
            metadata = json.load(inputfilemeta)
        with open(os.path.join(self.resources_dir, 'psfMeta.json'), mode="r", encoding="utf-8") as inputfilemeta:
            psf_metadata = json.load(inputfilemeta)
        photopeak_value = metadata['inputDataMeta']['photopeak_value']
        upper_window_value = metadata['scatterCorrectionMeta']['upper_window_value']
        lower_window_value = metadata['scatterCorrectionMeta']['lower_window_value']
        collimator = psf_metadata['psfModellingMeta']['collimator']
        intrinsic_resolution = psf_metadata['psfModellingMeta']['intrinsic_resolution']
        psf_toggle = json.loads(psf_metadata['toggle']['psf_toggle'].lower())
        attenuation_toggle = json.loads(psf_metadata['toggle']['attenuation_toggle'].lower())
        energy_window,_,_ = getEnergyWindow(files_NM[0])
        peak_window_idx = energy_window.index(photopeak_value)
        upper_window_idx = energy_window.index(upper_window_value)
        lower_window_idx = energy_window.index(lower_window_value)
        return (files_NM, attenuation_toggle, ct_file_node, psf_toggle, collimator, intrinsic_resolution, peak_window_idx, 
                    upper_window_idx, lower_window_idx)

    def run_test_osem(self):
        self.delayDisplay("OSEM test started!")
        with open(os.path.join(self.resources_dir,'algorithmTestSettings.json'), mode="r", encoding="utf-8") as settingsfile:
            algorithm_settings = json.load(settingsfile) 
        reconstructedDCMInstances = self.test_osem(algorithm_settings)
        self.delayDisplay("Downloading test osem recon data")
        osem_recon_path = Path(slicer.app.temporaryPath) /"osem_recon"
        osem_recon_path.mkdir(parents=True, exist_ok=True)
        osem_recon_url_id = "132oc-3ZDs7Cvon7i2mTvGd-v1sBhENE-"
        zip_path = osem_recon_path / 'osem_1it_8ss.zip'
        status = download_file_from_google_drive(osem_recon_url_id, zip_path)
        if status:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(osem_recon_path)
            dicom_series=[]
            for filename in sorted(os.listdir(os.path.join(osem_recon_path, "osem_1it_8ss"))):
                filepath = os.path.join(osem_recon_path, f'./osem_1it_8ss/{filename}')
                dicom_file = pydicom.dcmread(filepath)
                dicom_series.append(dicom_file.pixel_array)
        else:
            logger.error(f"Failed to download osem recon data")
        test_recon_series = []
        for _, dataset in enumerate(reconstructedDCMInstances):
            test_recon_series.append(dataset.pixel_array)
        self.assertEqual(len(dicom_series), len(test_recon_series))
        mse_error = np.mean((np.array(dicom_series)-np.array(test_recon_series))**2)
        self.assertTrue(mse_error<0.05)
        self.delayDisplay("OSEM test passed!")
        self.cleanUP()
    
    def run_test_bsrem(self):
        self.delayDisplay("BSREM test started!")
        with open(os.path.join(self.resources_dir,'algorithmTestSettings.json'), mode="r", encoding="utf-8") as settingsfile:
            algorithm_settings = json.load(settingsfile)
        bsrem_recon_path = Path(slicer.app.temporaryPath) /"bsrem_recon"
        bsrem_recon_path.mkdir(parents=True, exist_ok=True)
        #############Test RelativeDifferencePenalty Prior#############
        test_relativeDifferencePenalty_prior = False
        if test_relativeDifferencePenalty_prior:
            prior_type = "RelativeDifferencePenalty"
            rdp_reconstructedDCMInstances = self.test_bsrem(algorithm_settings, prior_type)
            self.delayDisplay("Downloading test relative difference penalty bsrem recon data")
            bsrem_rdp_recon_url_id = "1WBgcnEyR_jQvCCd8TNLmSBjXME3zXvFR"
            zip_path = bsrem_recon_path/'bsrem_1it_8ss_rdp.zip'
            status = download_file_from_google_drive(bsrem_rdp_recon_url_id, zip_path)
            if status:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(bsrem_recon_path)
                bsrem_rdp_dicom_series=[]
                for filename in sorted(os.listdir(os.path.join(bsrem_recon_path, "bsrem_1it_8ss_rdp"))):
                    filepath = os.path.join(bsrem_recon_path, f'./bsrem_1it_8ss_rdp/{filename}')
                    dicom_file = pydicom.dcmread(filepath)
                    bsrem_rdp_dicom_series.append(dicom_file.pixel_array)
            else:
                logger.error(f"Failed to download brem relative difference penalty recon data")
            bsrem_rdp_test_recon_series = []
            for _, dataset in enumerate(rdp_reconstructedDCMInstances):
                bsrem_rdp_test_recon_series.append(dataset.pixel_array)
            self.assertEqual(len(bsrem_rdp_dicom_series), len(bsrem_rdp_test_recon_series))
            rdp_mse_error = np.mean((np.array(bsrem_rdp_dicom_series)-np.array(bsrem_rdp_test_recon_series))**2)
            self.assertTrue(rdp_mse_error<0.05)
            self.delayDisplay("BSREM relative difference prior test passed!")
        ################Test LogCosh Prior##################
        test_logCosh_prior = True
        if test_logCosh_prior:
            prior_type = "LogCosh"
            logcosh_reconstructedDCMInstances = self.test_bsrem(algorithm_settings, prior_type)
            self.delayDisplay("Downloading test logcosh bsrem recon data")
            bsrem_logcosh_recon_url_id = "1wOMuaxBFl3ntxvGdqR7LO6YXsdNMjrtz"
            zip_path = bsrem_recon_path / 'bsrem_1it_8ss_logcosh.zip'
            status = download_file_from_google_drive(bsrem_logcosh_recon_url_id, zip_path)
            if status:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(bsrem_recon_path)
                bsrem_logcosh_dicom_series=[]
                for filename in sorted(os.listdir(os.path.join(bsrem_recon_path, "bsrem_1it_8ss_logcosh"))):
                    filepath = os.path.join(bsrem_recon_path, f'./bsrem_1it_8ss_logcosh/{filename}')
                    dicom_file = pydicom.dcmread(filepath)
                    bsrem_logcosh_dicom_series.append(dicom_file.pixel_array)
            else:
                logger.error(f"Failed to download brem LogCosh prior recon data")
            bsrem_logcosh_test_recon_series = []
            for _, dataset in enumerate(logcosh_reconstructedDCMInstances):
                bsrem_logcosh_test_recon_series.append(dataset.pixel_array)
            self.assertEqual(len(bsrem_logcosh_dicom_series), len(bsrem_logcosh_test_recon_series))
            logcosh_mse_error = np.mean((np.array(bsrem_logcosh_dicom_series)-np.array(bsrem_logcosh_test_recon_series))**2)
            self.assertTrue(logcosh_mse_error<0.05)
            self.delayDisplay("BSREM log cosh prior test passed!")
        ################Test Quadratic Prior#####################
        test_quadratic_prior = False
        if test_quadratic_prior:
            prior_type = "Quadratic"
            quadratic_reconstructedDCMInstances = self.test_bsrem(algorithm_settings, prior_type)
            self.delayDisplay("Downloading test Quadratic bsrem recon data")
            bsrem_quadratic_recon_url_id = '1DJ0RNNQRKEDK1Wpworc9NuEWl_XHih7M'
            zip_path = bsrem_recon_path/'bsrem_1it_8ss_quadratic.zip'
            status = download_file_from_google_drive(bsrem_quadratic_recon_url_id, zip_path)
            if status:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(bsrem_recon_path)
                bsrem_quadratic_dicom_series=[]
                for filename in sorted(os.listdir(os.path.join(bsrem_recon_path, "bsrem_1it_8ss_quadratic"))):
                    filepath = os.path.join(bsrem_recon_path, f'./bsrem_1it_8ss_quadratic/{filename}')
                    dicom_file = pydicom.dcmread(filepath)
                    bsrem_quadratic_dicom_series.append(dicom_file.pixel_array)
            else:
                logger.error(f"Failed to download bsrem Quadratic prior recon data")
            bsrem_quadratic_test_recon_series = []
            for _, dataset in enumerate(quadratic_reconstructedDCMInstances):
                bsrem_quadratic_test_recon_series.append(dataset.pixel_array)
            self.assertEqual(len(bsrem_quadratic_dicom_series), len(bsrem_quadratic_test_recon_series))
            quadratic_mse_error = np.mean((np.array(bsrem_quadratic_dicom_series)-np.array(bsrem_quadratic_test_recon_series))**2)
            self.assertTrue(quadratic_mse_error<0.05)
            self.delayDisplay("BSREM Quadratic test passed!")
        self.cleanUP()

    def run_test_osmaposl(self):
        self.delayDisplay("OSMAPOSL test started!")
        with open(os.path.join(self.resources_dir,'algorithmTestSettings.json'), mode="r", encoding="utf-8") as settingsfile:
            algorithm_settings = json.load(settingsfile)
        osmaposl_recon_path = Path(slicer.app.temporaryPath) /"osmaposl_recon"
        osmaposl_recon_path.mkdir(parents=True, exist_ok=True)
        #############Test RelativeDifferencePenalty Prior#########################
        test_relativeDifferencePenalty_prior = False
        if test_relativeDifferencePenalty_prior:
            prior_type = "RelativeDifferencePenalty"
            rdp_reconstructedDCMInstances = self.test_osmaposl(algorithm_settings, prior_type)
            self.delayDisplay("Downloading test relative difference penalty osmaposl recon data")
            osmaposl_rdp_recon_url_id = "1nO9D_hP9K4Iz_1JS8aidOLd0Get6qMR5"
            zip_path = osmaposl_recon_path/'osmaposl_1it_8ss_rdp.zip'
            status = download_file_from_google_drive(osmaposl_rdp_recon_url_id, zip_path)
            if status:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(osmaposl_recon_path)
                osmaposl_rdp_dicom_series=[]
                for filename in sorted(os.listdir(os.path.join(osmaposl_recon_path, "osmaposl_1it_8ss_rdp"))):
                    filepath = os.path.join(osmaposl_recon_path, f'./osmaposl_1it_8ss_rdp/{filename}')
                    dicom_file = pydicom.dcmread(filepath)
                    osmaposl_rdp_dicom_series.append(dicom_file.pixel_array)
            else:
                logger.error(f"Failed to download osmaposl relative difference penalty recon data")
            osmaposl_rdp_test_recon_series = []
            for _, dataset in enumerate(rdp_reconstructedDCMInstances):
                osmaposl_rdp_test_recon_series.append(dataset.pixel_array)
            self.assertEqual(len(osmaposl_rdp_dicom_series), len(osmaposl_rdp_test_recon_series))
            rdp_mse_error = np.mean((np.array(osmaposl_rdp_dicom_series)-np.array(osmaposl_rdp_test_recon_series))**2)
            self.assertTrue(rdp_mse_error<0.05)
            self.delayDisplay("OSMAPOSL relative difference prior test passed!")
        ##############Test LogCosh Prior###################
        test_logCosh_prior = True
        if test_logCosh_prior:
            prior_type = "LogCosh"
            logcosh_reconstructedDCMInstances = self.test_osmaposl(algorithm_settings, prior_type)
            self.delayDisplay("Downloading test logcosh osmaposl recon data")
            osmaposl_logcosh_recon_url_id = "1Xz8v23Lxl3bXx4C66MUqN5kCGAaYQP75"
            zip_path = osmaposl_recon_path / 'osmaposl_1it_8ss_logcosh.zip'
            status = download_file_from_google_drive(osmaposl_logcosh_recon_url_id, zip_path)
            if status:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(osmaposl_recon_path)
                osmaposl_logcosh_dicom_series=[]
                for filename in sorted(os.listdir(os.path.join(osmaposl_recon_path, "osmaposl_1it_8ss_logcosh"))):
                    filepath = os.path.join(osmaposl_recon_path, f'./osmaposl_1it_8ss_logcosh/{filename}')
                    dicom_file = pydicom.dcmread(filepath)
                    osmaposl_logcosh_dicom_series.append(dicom_file.pixel_array)
            else:
                logger.error(f"Failed to download osmaposl LogCosh prior recon data")
            osmaposl_logcosh_test_recon_series = []
            for _, dataset in enumerate(logcosh_reconstructedDCMInstances):
                osmaposl_logcosh_test_recon_series.append(dataset.pixel_array)
            self.assertEqual(len(osmaposl_logcosh_dicom_series), len(osmaposl_logcosh_test_recon_series))
            logcosh_mse_error = np.mean((np.array(osmaposl_logcosh_dicom_series)-np.array(osmaposl_logcosh_test_recon_series))**2)
            self.assertTrue(logcosh_mse_error<0.05)
            self.delayDisplay("OSMAPOSL log cosh prior test passed!")
        ##################Test Quadratic Prior#######################
        test_quadratic_prior = False
        if test_quadratic_prior:
            prior_type = "Quadratic"
            quadratic_reconstructedDCMInstances = self.test_osmaposl(algorithm_settings, prior_type)
            self.delayDisplay("Downloading test Quadratic osmaposl recon data")
            osmaposl_quadratic_recon_url_id = '14OmIoeHh4pGd4qEglNPJqJ87sCn4Orlk'
            zip_path = osmaposl_recon_path/'osmaposl_1it_8ss_quadratic.zip'
            status = download_file_from_google_drive(osmaposl_quadratic_recon_url_id, zip_path)
            if status:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(osmaposl_recon_path)
                osmaposl_quadratic_dicom_series=[]
                for filename in sorted(os.listdir(os.path.join(osmaposl_recon_path, "osmaposl_1it_8ss_quadratic"))):
                    filepath = os.path.join(osmaposl_recon_path, f'./osmaposl_1it_8ss_quadratic/{filename}')
                    dicom_file = pydicom.dcmread(filepath)
                    osmaposl_quadratic_dicom_series.append(dicom_file.pixel_array)
            else:
                logger.error(f"Failed to download osmaposl Quadratic prior recon data")
            osmaposl_quadratic_test_recon_series = []
            for _, dataset in enumerate(quadratic_reconstructedDCMInstances):
                osmaposl_quadratic_test_recon_series.append(dataset.pixel_array)
            self.assertEqual(len(osmaposl_quadratic_dicom_series), len(osmaposl_quadratic_test_recon_series))
            quadratic_mse_error = np.mean((np.array(osmaposl_quadratic_dicom_series)-np.array(osmaposl_quadratic_test_recon_series))**2)
            self.assertTrue(quadratic_mse_error<0.05)
            self.delayDisplay("OSMAPOSL Quadratic test passed!")      
        self.cleanUP()