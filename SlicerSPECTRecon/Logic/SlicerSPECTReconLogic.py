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
from Logic.volumeutils import *
from Logic.likelihood import *
from Logic.algorithms import *
from Logic.systemMatrix import *
from Logic.priors import *
from Logic.transforms import *
from Logic.vtkkmrmlutils import *
from Logic.getmetadatautils import *
from Logic.simindToDicom import *


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

    def reconstruct(
        self,
        files_NM,
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
        _ , mean_window_energies, idx_sorted = getEnergyWindow(files_NM[0])
        index_peak = idx_sorted[peak_window_idx]
        index_upper = idx_sorted[upper_window_idx] if upper_window_idx is not None else None
        index_lower = idx_sorted[lower_window_idx] if lower_window_idx is not None else None
        # Loop over and reconstruct all bed positions
        recon_array = []
        for bed_idx in range(len(files_NM)):
            object_meta, proj_meta = get_object_meta_proj_meta(bed_idx, files_NM, index_peak)
            photopeak, scatter = get_photopeak_scatter(bed_idx, files_NM, index_peak, index_lower, index_upper)
            # Transforms used for system modeling
            obj2obj_transforms = []
            if attenuation_toggle:
                files_CT = filesFromNode(ct_file)
                attenuation_map = getAttenuationMap(files_CT, files_NM, bed_idx, index_peak)
                att_transform = spectAttenuationTransform(attenuation_map)
                obj2obj_transforms.append(att_transform)
            if psf_toggle:
                peak_window_energy = mean_window_energies[index_peak]
                psf_meta = getPSFMeta(collimator, peak_window_energy, intrinsic_resolution)
                psf_transform = spectPSFTransform(psf_meta)
                obj2obj_transforms.append(psf_transform)
            # Build system matrix
            system_matrix = spectSystemMatrix(obj2obj_transforms, object_meta, proj_meta)
            # Build likelihood
            likelihood = poissonLogLikelihood(system_matrix, photopeak, scatter)
            # Select prior
            prior = selectPrior(prior_type, prior_beta, prior_delta, prior_gamma, 
                                files_NM, bed_idx, index_peak, N_prior_anatomy_nearest_neighbours, 
                                prior_anatomy_image_file)
            # Build algorithm
            reconstruction_algorithm = selectAlgorithm(algorithm, likelihood, prior)
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
            recon_stitched = dicom.stitch_multibed(recons=torch.stack(recon_array), files_NM = fileNMpaths)
            fileNMpath_save = fileNMpaths[order[-1]]
        else:
            recon_stitched = recon_array[0]
            fileNMpath_save = fileNMpaths[0]
        reconstructedDCMInstances = dicom.save_dcm(save_path = None, object = recon_stitched, 
                                                   file_NM = fileNMpath_save, recon_name = 'slicer_recon', return_ds = True)
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
        volumeNode = getVolumeNode(loadedNodeIDs)
        displayVolumeInViewer(volumeNode, outputVolumeNode)
        removeNode(volumeNode, temp_dir)
        logging.info("Reconstruction successful")
