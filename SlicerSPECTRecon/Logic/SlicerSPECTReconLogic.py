import logging
import os
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
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
import copy
from pathlib import Path
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from datetime import datetime
from Logic.volumeutils import *


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
        saveFilesInBrowser(save_path)
        # Create temp database so files can be loaded in viewer
        _ = loadFromTempDB(save_path)

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
            energy_window_name = energy_window_information.EnergyWindowName
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

    def stitchMultibed(self, recon_array, fileNMpaths):
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
        return reconstructedDCMInstances
        
    def saveVolumeInTempDB(self, reconstructedDCMInstances, outputVolumeNode):
        temp_dir = createTempDir()
        for i, dataset in enumerate(reconstructedDCMInstances):
            temp_file_path = os.path.join(temp_dir, f"temp_{i}.dcm")
            dataset.save_as(temp_file_path)
        saveFilesInBrowser(temp_file_path)
        self.getAndDisplayVolume(temp_dir, outputVolumeNode)

    def getAndDisplayVolume(self, temp_dir, outputVolumeNode):
        loadedNodeIDs = loadFromTempDB(temp_dir)
        volumeNode = getVolumeNode(loadedNodeIDs[0])
        displayVolumeInViewer(volumeNode, outputVolumeNode)
        removeNode(volumeNode, temp_dir)
        print("Reconstruction successful")
