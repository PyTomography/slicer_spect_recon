---
title: 'SlicerSPECTRecon: A 3D Slicer Extension for SPECT Image Reconstruction'
tags:
  - 3D slicer
  - nuclear medicine
  - tomography
  - spect
  - image reconstruction
authors:
  - name: Obed K. Dzikunu
    orcid: 0000-0002-1122-0629
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "2, 3" # (Multiple affiliations must be quoted)
  - name: Maziar Sabouri
    orcid: 0000-0001-7525-8952
    affiliation: "1, 2"
  - name: Shadab Ahamed
    orcid: 0000-0002-2051-6085
    affiliation: "1, 2"
  - name: Carlos Uribe
    orcid: 0000-0003-3127-7478
    affiliation: "2, 4, 5"
  - name: Arman Rahmim
    orcid: 0000-0002-9980-2403
    affiliation: "1, 2, 3, 5"
  - name: Lucas A. Polson
    orcid: 0000-0002-3182-2782
    affiliation: "1, 2"    
affiliations:
 - name: Deparment of Physics & Astronomy, University of British Columbia, Vancouver, Canada
   index: 1
 - name: Department of Integrative Oncology, BC Cancer Research Institute, Vancouver, Canada
   index: 2
 - name: School of Biomedical Engineering, University of British Columbia, Vancouver, Canada
   index: 3
 - name: Department of Radiology, University of British Columbia, Vancouver, Canada
   index: 4
 - name: Molecular Imaging and Therapy Department, BC Cancer Research Institute, Vancouver, Canada
   index: 5
date: 25 November 2024
bibliography: paper.bib
---

# Summary

`SlicerSPECTRecon` is a `3D Slicer` [@Kikinis2014-3DSlicer] extension designed for Single Photon Emission Computed Tomography (SPECT) image reconstruction. It offers a range of popular reconstruction algorithms and requires raw projection data from clinical or Monte Carlo simulated scanners. Built with the PyTomography Python library [@pytomography], it features GPU-accelerated functionality for fast reconstruction. The extension includes a graphical user interface for the selection of reconstruction parameters, and reconstructed images can be post-processed using all available `3D Slicer` functionalities.


# Statement of need

SPECT imaging is used to measure the 3D distribution of a radioactive molecule within a patient; it requires (i) acquisition of 2D `projection` images at different angles using a gamma camera followed by (ii) the use of a tomographic image reconstruction algorithm to obtain a 3D radioactivity distribution consistent with the acquired data. While there exist many commercial products for image reconstruction, as well as multiple low-level software packages [@pytomography; @STIR; @castor], there is currently no open source graphic user interface (GUI) that provides access to the latest reconstruction algorithms. Due to continuing research interest in the implications of different reconstruction techniques on various clinical tasks [@hermes_spect_ap; @spect_ai], there is a need for a open-source GUI for SPECT reconstruction.


# Overview of SlicerSPECTRecon

Typical reconstruction algorithms in nuclear medicine attempt to maximize a likelihood function via
\begin{equation}
  \hat{x} = \underset{x;A}{\text{argmax}}~L(y|x,H,s)
\end{equation}
where $\hat{x}$ is the 3D image estimate, $A$ is the reconstruction algorithm, $L$ is a statistical likelihood function that describes the acquired data $y$, $H$ is a linear operator that models the imaging system, $s$ is an additive term that corrects for scatter, and $\text{argmax}_{x;A}$ signifies maximization of the likelihood with respect to $x$ using algorithm $A$. Based on this formalism, SlicerSPECTRecon is partitioned into four main components: input data, system modeling, likelihoods, and reconstruction algorithms. There are additional modules for input data conversion as well as post reconstruction applications.

Once the raw data is loaded into the 3D slicer interface, the user can select the projection data $y$ to reconstruct in the `Input Data` tab. Data from multiple bed positions can be selected at this stage; resulting reconstructions will contain a single 3D image where all the bed positions are stitched. Upon selection of the projection data, necessary information about the acquired energy windows becomes stored internally. Users then select which energy window they want to reconstruct via the `Photopeak` option; it is possible to to reconstruct data from multiple photopeaks simultaneously provided the relative weighting between calibration factors is provided for each peak.

The `System Modeling` component considers the system matrix $H$, as well as additional corrections such as scatter $s$. Selecting `Attenuation Correction` enables attenuation correction during reconstruction; users must specify a corresponding CT image used to generate a mu-map. Selecting `Collimator Detector Response Modeling` enables modeling of the collimator and detector spatial resolution in image reconstruction. Users must specify the collimator code, defined on [the PyTomography data page](https://pytomography.readthedocs.io/en/latest/data_tables/collimator_data.html#collimator-data-index), as well as the intrinsic spatial resolution of the scintillator crystals. Selecting `Scatter Correction` enables scatter correction during reconstruction; users must select the scatter correction method as well as supporting data required for the method.

The third component, likelihoods, considers the likelihood function $L$. Currently, the extension only supports the `PoissonLogLikelihood` likelihood, which correctly describes the data acquired in SPECT imaging. It may be desirable in the future to test alternative likelihood functions, so this is left as a seperate module.

The fourth component, algorithms, considers the reconstruction algorithm $A$. Currently, the extension supports the ordered subset expectation maximum [OSEM, @osem], block sequential regularized expectation maximum [BSREM, @BSREM] and ordered subset maximum a posteriori one step late [OSMAPOSL, @OSL] algorithms. Regularized algorithms can use the quadratic, log-cosh, and relative difference penalty [@RDP] priors; these priors can also utilize a provided anatomical image to modify the weighting by only including contributions from the top N neighbours. Additional algorithms may be added based on community request.

 The `Data Converters` component permits the conversion of raw SPECT data from various file formats, such as `SIMIND` [@simind] and `GATE` [@gate], into DICOM format so it can be loaded using the native data readers of `3D Slicer`. While the extension currently has support for the conversion of `SIMIND` data, more data converters may be added in the future depending on community request. The `Post-Reconstruction` component contains functionality that is applicable for use on reconstructed images. One such example is the computation of uncertainty on total counts within segmented regions of interest.

\autoref{fig:fig1} contains a screenshot of the extension along with a sample reconstructed image in the 3D Slicer viewer.   

![Left: user interface for the proposed extension. Right: reconstructed coronal slice from a patient receiving ${}^{177}$Lu-PSMA-617 radiopharmaceutical therapy (color) overlayed on a corresponding CT (greyscale). Raw SPECT data consisted of two bed positions that were automatically stitched together after each was reconstructed; the raw data was acquired on a GE Discovery 670 camera.\label{fig:fig1}](with_recon.png)


# Acknowledgements

We would like to acknowledge Peyman Sh.Zadeh from the Faculty of Medicine at the Tehran University of Medical Science for providing the patient data used in the paper. Also, we acknowledge funding from the Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grant RGPIN-2019-06467, and Canadian Institutes of Health Research (CIHR) Project Grant PJT-162216.

# References
