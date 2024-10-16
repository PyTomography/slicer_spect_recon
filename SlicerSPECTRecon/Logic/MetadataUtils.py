from pytomography.io.SPECT import dicom
import numpy as np
import pydicom

def getEnergyWindow(directory):
    ds = pydicom.dcmread(directory)
    window_names = []
    mean_window_energies = []
    for energy_window_information in ds.EnergyWindowInformationSequence:
        lower_limit = energy_window_information.EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
        upper_limit = energy_window_information.EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
        energy_window_name = energy_window_information.EnergyWindowName
        mean_window_energies.append((lower_limit+upper_limit)/2)
        window_names.append(f'{energy_window_name} ({lower_limit:.2f}keV - {upper_limit:.2f}keV)')
    idx_sorted = list(np.argsort(mean_window_energies))
    window_names = list(np.array(window_names)[idx_sorted])
    mean_window_energies = list(np.array(mean_window_energies)[idx_sorted])
    # Insert None option to beginning
    window_names.insert(0, 'None')
    idx_sorted.insert(0, None)
    return window_names, mean_window_energies, idx_sorted

def get_object_meta_proj_meta(bed_idx, files_NM, index_peak):
    file_NM = files_NM[bed_idx]
    object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak)
    return object_meta, proj_meta

def get_photopeak_scatter(bed_idx, files_NM, index_peak, index_lower=None, index_upper=None):
    projectionss = dicom.load_multibed_projections(files_NM)
    photopeak = projectionss[bed_idx][index_peak]
    # No scatter
    if (index_lower is None)*(index_upper is None):
        scatter = None
    # Dual or triple energy window
    else:
        file_NM = files_NM[bed_idx]
        scatter = dicom.get_energy_window_scatter_estimate_projections(file_NM, projectionss[bed_idx], index_peak, index_lower, index_upper)
    return photopeak, scatter