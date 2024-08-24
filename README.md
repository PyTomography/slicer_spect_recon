# SPECT Tomographic Reconstruction 3D Slicer Extension

This is the official repository for the `Slicer` extension `SlicerSPECTRecon`.

This module enables the reconstruction of raw SPECT projection data, providing customizable options for image modeling and image reconstruction. The module has a SIMIND to DICOM converter to permit reconstruction of SIMIND Monte Carlo data.

The module is divided into the following sections:

- Data Converters: Provides tools for converting data from various sources into the DICOM format.
- Input Data: Users can select data from multiple bed positions after loading the projection data into the 3D Slicer DICOM database. 
- System Modeling: Allows users to define transforms that are used to build the system matrix.
- Likelihood: Allows users to choose their preferred likelihood function.
- Reconstruction Algorithm: Provides the option of selecting a preferred reconstruction algorithm and their associated parameters

Please refer to the `User_Manual.md` file for further information

## User interface

- Inputs
  - Input volume: input SPECT/CT dicom files, simind file (convert to dicom using the data converter)
- Outputs
  - Reconstructed volume: The volume will be saved under the specified name (or as the dataset name appended with _reconstructed) and will be located within the Subject Hierarchy in the Data Module.

## Resources

The following link collection should facilitate understanding the code in this extension:

- [Slicer Utils](https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/util.py)
- [DICOM Introduction](https://www.slicer.org/wiki/Documentation/Nightly/FAQ/DICOM)
- [DICOM Structure](https://www.slicer.org/wiki/Documentation/4.0/Modules/CreateDICOMSeries)
- [DICOM Browser](https://dicom.innolitics.com/ciods)
- [Subject Hierarchy](https://www.slicer.org/wiki/Documentation/4.5/Modules/SubjectHierarchy)

## Sample Data

The links to the example data (sample patient and simind files) are in the sample_data.txt file in the `Resources` folder. 


## Contribute

If you'd like to contribute, you can find an orientation on the Slicer [documentation for developers](https://www.slicer.org/wiki/Documentation/Nightly/Developers).

Please read first the `CONTRIBUTING.md` file for further information on how to contribute.

## License

SlicerSPECTRecon is subject to the `MIT License`, which is in the project's root.


## Contact

Please post any questions to the [Pytomography Discourse Forum](https://pytomography.discourse.group/).