from __future__ import annotations
import logging
import os
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from pytomography.io.SPECT import dicom
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.projectors import ExtendedSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
import numpy as np
import pydicom
import torch
from pytomography.algorithms import PGAAMultiBedSPECT
from pytomography.callbacks import DataStorageCallback
from Logic.VolumeUtils import *
from Logic.Algorithms import *
from Logic.Priors import *
from Logic.VtkkmrmlUtils import *
from Logic.MetadataUtils import *
from Logic.SimindToDicom import *

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
        self.stored_recon_iters = {}
        
    def get_system_matrix(self, file_NM, attenuation_toggle, ct_file, psf_toggle, collimator, intrinsic_resolution, index_peak):
        object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak)
        obj2obj_transforms = []
        if attenuation_toggle:
            files_CT = filesFromNode(ct_file)
            attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM, index_peak)
            att_transform = SPECTAttenuationTransform(attenuation_map)
            obj2obj_transforms.append(att_transform)
        if psf_toggle:
            _ , mean_window_energies, _= getEnergyWindow(file_NM)
            peak_window_energy = mean_window_energies[index_peak]
            print(peak_window_energy)
            psf_meta = dicom.get_psfmeta_from_scanner_params(collimator, peak_window_energy, intrinsic_resolution=intrinsic_resolution)
            psf_transform = SPECTPSFTransform(psf_meta)
            obj2obj_transforms.append(psf_transform)
        system_matrix = SPECTSystemMatrix(
            obj2obj_transforms = obj2obj_transforms,
            proj2proj_transforms = [],
            object_meta = object_meta,
            proj_meta = proj_meta
        )
        return system_matrix

    def reconstruct(
        self,
        files_NM: Sequence[str],
        attenuation_toggle: bool,
        CT_node: slicer.vtkMRMLScalarVolumeNode,
        psf_toggle: bool,
        collimator_code: str,
        intrinsic_resolution: float,
        index_peak: int | Sequence[int], 
        index_upper: int | Sequence[int],
        index_lower: int | Sequence[int],
        algorithm_name: str,
        prior_type: str,
        prior_beta: float,
        prior_delta: float,
        prior_gamma: float,
        use_prior_image: bool,
        prior_anatomy_image_node: slicer.vtkMRMLScalarVolumeNode,
        N_prior_anatomy_nearest_neighbours: int,
        n_iters: int,
        n_subsets: int,
        store_recons: bool = False
    ): 
        if index_peak is None:
            logging.error("Please select a photopeak energy window")
            return
        # Loop over and reconstruct all bed positions
        recon_algorithms = []
        for bed_idx in range(len(files_NM)):
            # if multi photopeak
            if type(index_peak) is list: 
                photopeak = []
                scatter = []
                system_matrices = []
                for i in range(len(index_peak)):
                    photopeak_, scatter_ = get_photopeak_scatter(bed_idx, files_NM, index_peak[i], index_lower[i], index_upper[i])
                    system_matrix = self.get_system_matrix(files_NM[bed_idx], attenuation_toggle, CT_node, psf_toggle, collimator_code, intrinsic_resolution, index_peak[i])
                    photopeak.append(photopeak_)
                    scatter.append(scatter_)
                    system_matrices.append(system_matrix)
                photopeak = torch.stack(photopeak)
                scatter = torch.stack(scatter)
                system_matrix = ExtendedSystemMatrix(system_matrices)
            # if regular photopeak
            else: 
                photopeak, scatter = get_photopeak_scatter(bed_idx, files_NM, index_peak, index_lower, index_upper)
                system_matrix = self.get_system_matrix(files_NM[bed_idx], attenuation_toggle, CT_node, psf_toggle, collimator_code, intrinsic_resolution, index_peak)
            # Build likelihood
            likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)
            # Select prior
            prior = selectPrior(
                prior_type,
                prior_beta,
                prior_delta,
                prior_gamma, 
                files_NM,
                bed_idx,
                index_peak,
                use_prior_image,
                N_prior_anatomy_nearest_neighbours, 
                prior_anatomy_image_node
            )
            # Build algorithm
            recon_algorithm = selectAlgorithm(algorithm_name, likelihood, prior)
            recon_algorithms.append(recon_algorithm)
        recon_algorithm_all_beds = PGAAMultiBedSPECT(files_NM, recon_algorithms)  
        if store_recons:
            callbacks = [DataStorageCallback(r.likelihood, r.object_prediction) for r in recon_algorithm_all_beds.reconstruction_algorithms]
            reconstructed_image_multibed = recon_algorithm_all_beds(n_iters, n_subsets, callback=callbacks)
            volume_node = self.create_volume_node_from_recon(reconstructed_image_multibed, files_NM)
            # Store information for accessing later
            self.stored_recon_iters[volume_node.GetID()] = [recon_algorithm_all_beds, callbacks]
        else:
            reconstructed_image_multibed = recon_algorithm_all_beds(n_iters, n_subsets)
            volume_node = self.create_volume_node_from_recon(reconstructed_image_multibed, files_NM)
        return volume_node
    
    def compute_uncertainty(self, mask, recon_node_id):
        try:
            algorithm, callbacks = self.stored_recon_iters[recon_node_id]
        except:
            logging.error("No reconstruction stored for this volume node; to compute uncertainty, you need to store the reconstruction first")
            return
        uncertainty_abs, uncertainty_pct = algorithm.compute_uncertainty(
            torch.tensor(mask).permute(2,1,0).to(pytomography.device),
            callbacks,
            return_pct=True
        )
        return uncertainty_abs, uncertainty_pct

    def create_volume_node_from_recon(
        self,
        reconstructed_image_multibed,
        fileNMpaths,
    ):
        # Get top bed position
        if len(fileNMpaths)>1:
            dss = np.array([pydicom.read_file(file_NM) for file_NM in fileNMpaths])
            zs = np.array(
                [ds.DetectorInformationSequence[0].ImagePositionPatient[-1] for ds in dss]
            )
            order = np.argsort(zs)
            fileNMpath_save = fileNMpaths[order[-1]]
        else:
            fileNMpath_save = fileNMpaths[0]
        recon_ds = dicom.save_dcm(
            save_path = None,
            object = reconstructed_image_multibed,
            file_NM = fileNMpath_save,
            recon_name = 'slicer_recon',
            return_ds = True
        )
        temp_dir = createTempDir()
        for i, dataset in enumerate(recon_ds):
            temp_file_path = os.path.join(temp_dir, f"temp_{i}.dcm")
            dataset.save_as(temp_file_path)
        saveFilesInBrowser(temp_file_path)
        loadedNodeIDs = loadFromTempDB(temp_dir)
        volume_node = getVolumeNode(loadedNodeIDs)
        return volume_node        
        
    def DisplayVolume(self, volume_node):
        displayVolumeInViewer(volume_node)
        logging.info("Reconstruction successful")