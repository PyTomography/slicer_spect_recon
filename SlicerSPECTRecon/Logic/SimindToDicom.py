
import os
from pathlib import Path
import numpy as np
import pytomography
from pytomography.io.SPECT import simind
from pytomography.io.shared import dicom_creation
from datetime import datetime
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import slicer

def simind2DICOMProjections(
    headerfiles, 
    time_per_projection, 
    scale, 
    random_seed, 
    save_path, 
    patient_name, 
    study_description
):
    _, proj_meta = simind.get_metadata(headerfiles[0][0])
    dr, dz = proj_meta.dr[0]*10, proj_meta.dr[1]*10 # to mm
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
    rotation_information.RotationDirection = 'CC' 
    radius = simind.get_header_value(headerdata, 'Radius', np.float32) * 10 # to mm
    rotation_information.RadialPosition = Nproj*[radius]
    extent_of_rotation = simind.get_header_value(headerdata, 'extent of rotation', np.float32)
    start_angle = simind.get_header_value(headerdata, 'start angle', np.float32)
    rotation_information.ScanArc = extent_of_rotation
    rotation_information.StartAngle = start_angle + 180
    rotation_information.NumberOfFramesInRotation = Nproj
    rotation_information.AngularStep = extent_of_rotation / Nproj
    rotation_information.ActualFrameDuration = time_per_projection * 1000 # to ms
    ds.RotationInformationSequence = Sequence([rotation_information])
    projections = simind.get_projections(headerfiles)
    if Nenergy==1:
        projections = projections.unsqueeze(0) # first dimension removed by default in pytomography
    np.random.seed(random_seed)
    projections_realization = np.random.poisson(projections.cpu().numpy()*time_per_projection*scale)
    projections_realization = np.transpose(projections_realization, (0,1,3,2))[:,:,::-1]
    ds.PixelData = projections_realization.astype(np.uint16).tobytes()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ds.save_as(os.path.join(save_path, f'{ds.SeriesInstanceUID}.dcm'))
    
def simind2DICOMAmap(
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
    SOP_class_UID = '1.2.840.10008.5.1.4.1.1.2'
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
    ds.RescaleIntercept = 0
    ds.Rows = amap.shape[1]
    ds.Columns = amap.shape[2]
    ds.PixelSpacing = [dx, dy]
    ds.SliceThickness = dz
    ds.SpacingBetweenSlices = dz
    ds.NumberOfSlices = Nz
    ds.NumberOfFrames = Nz
    ds.ImageOrientationPatient = [1,0,0,0,1,0]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = amap.tobytes()
    ds.ImagePositionPatient = [-(Nx-1)/2*dx, -(Ny-1)/2*dy, Nz * dz]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ds.save_as(os.path.join(save_path, f'{ds.SOPInstanceUID}.dcm'))