**DISCLAIMER: This plugin is for academic purposes only and it is not recommended for use in clinical environment.**

# Setup

**Prerequisite**: 

- Latest version and release of 3D Slicer (5.6.1)

## Installation

The *SlicerSPECTRecon* module, when released to the extension manager, shall be installed automatically from *View* -> *Extension manager* -> *Install Extensions* or directly from ![extensions](images/install_extensions.png) in Slicer Home. However, it can also be downloaded or cloned  from [https://github.com/PyTomography/slicer_spect_recon.git](https://github.com/PyTomography/slicer_spect_recon.git). 

**In the case of download**, the insallation proceeds as follows:

1. Download the zip file (see image below) and extract it in a personal directory folder. It will contain, among other things, the folder SlicerSPECTRecon with the SlicerSPECTRecon.py file
2. Open Slicer and click ![customize](images/customize.png), or alternatively *Edit* -> *Application Settings*. 
3. In *Modules* -> *Additional module paths*, add the path to the folder containing the module files.
4. Restart Slicer to apply the changes.

<img src="images/download.png" width="300"/>

**In the case of cloning from GitHub**, the installation proceeds as follows:
 
1. In GitHub copy URL from "Clone with HTTPS" (see screenshot below)
2. Open terminal in a personal directory folder and execute the line: <pre><code>git clone https://github.com/PyTomography/slicer_spect_recon.git </code></pre>
(i.e. "git clone" plus the copied URL).
This will download a folder called *slicer_spect_recon*, with  all the necessary files for the module 
3. Open Slicer and click ![customize](images/customize.png), or alternatively *Edit* -> *Application Settings*. 
4. In *Modules* -> *Additional module paths*, add the path to the folder containing the module files (see screenshot below).
5. Restart Slicer.

<img src="images/clone.png" width="300"/>

<img src="images/modules.png" width="300"/>

Once Slicer is restarted, the SlicerSPECTRecon module will be available from *All Modules* → *Tomographic Reconstruction* → *SlicerSPECTRecon* 

<img src="images/modules_list.png" width="300"/>

## Tutorial Data
The tutorial data can be obtained [at the following link](https://zenodo.org/records/14172228); it will be used in the subsequent examples in the user manual.

## Data Loading

First of all, the input images must be imported into Slicer and loaded in the Slicer scene; the images must be in DICOM format.

1. Go to *File* → *Add DICOM Data* or simply click ![dcm](images/dcm.png)
2. Click *Import DICOM Files*
3. Select the folder containing the DICOM files you wish to import, and click "Import" in the file explorer.
4. In the "Dicom Database" on the right hand side, click on the patient name, select the files you want to important, and click "Load" (see Figure 0).

<img src="images/data_loading.png" width="300"/>

# Description

This module enables the reconstruction of SPECT projection data, supporting both patient and Simind datasets.

1. **Data Converters:** Tools for converting projection data and attenuation maps from various formats into DICOM. The module currently supports conversion of the SIMIND data format. Once data has been converted, it will be saved to a DICOM file somewhere on the users computer, which then needs to be loaded via the Slicer DICOM image loader.
    * *SIMIND Data Converter*: The output of a SIMIND simulation may include multiple energy windows: users specify how many energy windows they want to include in the DICOM file in the "Number of energy windows" section. Since DICOM data is stored in units of counts, and SIMIND data is output in counts/second/MBq (or in some cases counts/second/(MBq/mL)) users must specify a time per projection (second), and scale factor (MBq or MBq/mL) to multiply the SIMIND data by. The noiseless SIMIND data is then sampled according to a Poisson distribution with the given "Random Seed". Users need to select the header files for each window they want to provide (".h00" files); these files must be in the same directory as the data files (".a00" files). Finally, the user must provide a patient name to give to the generated data, and choose a location to save on the computer. To convert a SIMIND attenuation map into DICOM format, users must select the header files (".hct"); this file must be in the same directory as the data file ("ict"). The user then specifies the location on their computer where to save the resulting files.

<img src="images/converters.png" width="550"/>

2. **Input Data:** Used to obtain photopeak projections from loaded DICOM data. Users can select multiple projection files corresponding to different bed positions of the same scan: the reconstructed object will automatically stitch the separate bed positions together.

<img src="images/input.png" width="550"/>

3. **System Modeling:** Used to build the system matrix that defines the particular SPECT system
    * *Attenuation Correction*: Provided the CT/attenuation map corresponding to the projectio data has been loaded into slicer, it can be selected here and used to enable attenuation correction.
    * *Collimator Detector Response Modeling*: Users specify here the code (see [here](https://pytomography.readthedocs.io/en/latest/external_data.html)) that corresponds to the collimator used during acquisition. The intrinsic resolution of the scintillator crystals (in FHWM) can also be included.
    * *Scatter Correction*: The module currently supports dual energy window and triple energy window scatter correction. Users select from the energy windows corresponding to the projection data loaded in in the "Input Data" section.

<img src="images/systemmodeling.png" width="550"/>

4. **Likelihood:** This section allows users to choose their preferred likelihood function. It's currently fixed to the `PoissonLogLikelihood` option, but future releases might expand on other options.

<img src="images/likelihood.png" width="550"/>

5. **Reconstruction Algorithm:** Used to select the reconstruction algorithm for reconstructing the projection data. Currently ordered subset expectation maximum (OSEM), block sequential regularized expectation maximum (BSREM), and ordered subset maximum a posterior one step late (OSMAPOSL) are supported.
    * Regularized algorithms can include a regularizer from options such as `RelativeDifferencePenalty`, `LogCosh`, and `Quadratic` priors. Parameters for each prior are specified in the [PyTomography documentation](https://pytomography.readthedocs.io/en/latest/autoapi/pytomography/priors/index.html)
    * Regularizers may also use an additional anatomical image for weighting the contribution from nearby voxels. The anatomical image will automatically be aligned with the SPECT reconstruction geometry.

<img src="images/algorithms.png" width="550"/>

6. **Output:** This section allows users to specify the name of the resulting volume created after reconstruction in 3D slicer.

<img src="images/output.png" width="550"/>

# Limitations

1. The module requires some basic knowledge of Slicer modules from the user, specifically DICOM import and data management.
2. Currently, the software only supports dual head SPECT systems with parallel hole collimators (and has only been tested on Siemens Symbia and GE Discovery scanners). 
3. The module requires data to be in the DICOM format (but has a converter for other formats). Currently the only supported format for conversion to DICOM is SIMIND.

# Limitations

1. The module requires some basic knowledge of Slicer modules from the user, specifically DICOM import and data management.
2. Currently, the software only supports dual head SPECT systems with parallel hole collimators (and has only been tested on Siemens Symbia and GE Discovery scanners). 
3. The module requires data to be in the DICOM format (but has a converter for other formats). Currently the only supported format for conversion to DICOM is SIMIND.
