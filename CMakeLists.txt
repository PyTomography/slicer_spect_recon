cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SPECTRecon)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/PyTomography/slicer_spect_recon")
set(EXTENSION_CATEGORY "Tomographic Reconstruction")
set(EXTENSION_CONTRIBUTORS "Obed Dzikunu (QURIT), Luke Polson (QURIT), Maziar Sabouri (QURIT),  Shadab Ahamed (QURIT)")
set(EXTENSION_DESCRIPTION "This module implement GPU accelerated SPECT image reconstruction. User Manual: https://github.com/PyTomography/slicer_spect_recon/blob/main/User_Manual.md")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/PyTomography/slicer_spect_recon/main/Pytomography.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/PyTomography/slicer_spect_recon/main/images/screenshot.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerSPECTRecon)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
