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
    orcid: 0000-0002-1162-0629
    equal-contrib: true
    affiliation: "2, 3" # (Multiple affiliations must be quoted)
  - name: Maziar Sabouri
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2"
  - name: Shadab Ahamed
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2"
  - name: Carlos Uribe
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "2, 4, 5"
  - name: Arman Rahmim
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2, 3, 5"
  - name: Luke Polson
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2"    
affiliations:
 - name: Deparment of Physics & Astronomy, University of British Columbia, Vancouver Canada
   index: 1
 - name: Department of Integrative Oncology, BC Cancer Research Institute, Vancouver Canada
   index: 2
 - name: School of Biomedical Engineering, University of British Columbia, Vancouver Canada
   index: 3
 - name: Department of Radiology, University of British Columbia, Vancouver Canada
   index: 4
 - name: Molecular Imaging and Therapy Department, BC Cancer, Vancouver Canada
   index: 5
date: 05 September 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`SlicerSPECTRecon` is a `3D Slicer` [@Kikinis2014-3DSlicer] extension designed for Single Photon Emission Computed Tomography (SPECT) tomographic image reconstruction. It offers a range of popular reconstruction algorithms and requires raw projection data from clinical or Monte Carlo simulated scanners. Built with the PyTomography python library [@pytomography], it features GPU-accelerated functionality for fast reconstruction. The extension includes a graphical user interface for the selection of reconstruction parameters, and reconstructed images can be post-processed using all available `3D Slicer` functionality.


# Statement of need

SPECT imaging is used to measure the 3D distribution of a radioactive molecule within a patient; it requires (i) acquisition of 2D `projection` images at different angles using a gamma camera followed by (ii) use of a tomographic image reconstruction algorithm to obtain a 3D radioactivity distribution consistent with the acquired data. While there exist many commercial products for image reconstruction, as well as multiple low-level software packages [@pytomography] [@STIR] [@castor], there currently is no graphic user interface (GUI) that provides access to the latest reconstruction algorithms. Due to continuing research interest in the implications of different reconstruction techniques on various clinical tasks [@hermes_spect_ap] [@spect_ai], there is a need for a open-source GUI for SPECT reconstruction.


# Overview of SlicerSPECTRecon

Typical reconstruction algorithms in nuclear medicine attempt to maximize a liklihood function via
\begin{equation}
  \hat{x} = \underset{x;A}{\text{argmax}}~L(y|x,H,s)
\end{equation}
where $\hat{x}$ is the 3D image estimate, $A$ is the reconstruction algorithm, $L$ is a statistical likelihood function that describes the acquired data $y$, $H$ is a linear operator that models the imaging system, $s$ is an additive term that corrects for scatter, and $\text{argmax}_{x;A}$ signifies maximization of the liklihood with respect to $x$ using algorithm $A$. Based on this formalism, SlicerSPECTRecon is partioned into four main components: data converters / input data, system modeling, likelihoods, and reconstruction algorithms.

The first component, `Data Converters` / `Input Data`, considers the raw data $y$. For medical imaging data in nuclear medicine, `3D Slicer` provides native support for the DICOM data format. While this is sufficient for clinical data, there exist other data formats, such as Monte Carlo data from `SIMIND` [@simind] and `GATE`, which are not supported by `3D Slicer`. The purpose of the data converters tab is to provide tools for the conversion of data from alternative sources into DICOM format. Once converted into the DICOM format, data can be loaded into `3D Slicer` using the native `Data` module. While the extension currently has support for the conversion of `SIMIND` data, more data converters will be added in the future dependent on community request. Once the raw data is loaded into the 3D slicer interface, the user can select the projection data to reconstruct in the `Input Data` tab. Data from multiple bed positions can be selected at this stage; resulting reconstructions will contain a single 3D image where all the bed positions are stitched. When the projection data is selected, information on all energy windows becomes stored, and users can select which energy window they want to reconstruct in the `Photopeak` option.

The second component, `System Modeling`, considers the system matrix $H$ as well as additional corrections such as scatter $s$. Selecting `Attenuation Correction` enables attenuation correction during reconstruction; users must specify a corresponding CT image used to generate a mu-map. Selecting `Collimator Detector Response Modeling` enables modeling of the collimator and detector spatial resolution in image reconstruction. Users must specify the collimator code as well as the intrinsic spatial resolution of the scintillator crystals. Selecting `Scatter Correction` enables scatter correction during reconstruction; users must select the scatter correction method as well as supporting data required for the method.

The third component, likelihoods, considers the the likelihood function $L$. Currently, the extension only supports the `PoissonLogLikelihood` likelihood, which correctly describes the data acquired in SPECT imaging. It may be desirable in the future to test alternative likelihood functions, so this is left as a seperate module.

The fourth component, algorithms, considers the reconstruction algorithm $A$. Currently, the extension supports the ordered subset expectation maximum (OSEM) [@osem], block sequential regularized expectation maximum (BSREM) [@BSREM] and ordered subset maximum a posteriori one step late (OSMAPOSL) [@OSL] algorithms. Regularized algorithms can use the quadratic, log-cosh, and relative difference penalty prior [@RDP] functions. Regularizers may also use an additional anatomical image to modify the weighting by only including contributions from the top N neighbours. Additional algorithms will be added based on community request.

\autoref{fig:fig1} contains a screenshot of the extension along with a sample reconstructed image in the 3D Slicer viewer.   

![Left: Extension options. Right: reconstructed coronal slice of a ${}^{177}$Lu-PSMA-617 patient (two bed positions); the raw data was acquired on a GE Discovery 670 camera.\label{fig:fig1}](images/with_recon.png)


# Acknowledgements

We would like to acknowledge Peyman Sh.Zadeh from the Faculty of Medicine at the Tehran University of Medical Science for providing the patient data used in the paper.

# References