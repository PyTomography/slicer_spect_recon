from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform

def getAttenuationMap(files_CT, files_NM, bed_idx, index_peak):
    if len(files_CT)==1:
        # Then user provided mu-map TODO: assumes the attenuation map is aligned with the reconstructed image, this mainly is used for SIMIND mu-maps
        attenuation_map = dicom.get_attenuation_map_from_file(files_CT[0])
    else:
        # User provided CT slices  
        attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT, files_NM[bed_idx], index_peak)
    return attenuation_map 

def spectAttenuationTransform(attenuation_map):
    return SPECTAttenuationTransform(attenuation_map)

def getPSFMeta(collimator, peak_window_energy, intrinsic_resolution):
    psf_meta = dicom.get_psfmeta_from_scanner_params(collimator, peak_window_energy, intrinsic_resolution=intrinsic_resolution)
    return psf_meta

def spectPSFTransform(psf_meta):
    return SPECTPSFTransform(psf_meta)