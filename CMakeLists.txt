cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(Pytomography)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/Pytomography")
set(EXTENSION_CATEGORY "Pytomography")
set(EXTENSION_CONTRIBUTORS "Obed Dzikunu (QURIT), Luke Polson (QURIT), Maziar Sabouri (QURIT),  Shadab Ahamed (QURIT)")
set(EXTENSION_DESCRIPTION "GPU accelerated SPECT and PET reconstruction application")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/Pytomography.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/Pytomography/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(pytomography)
add_subdirectory(pytomography)
add_subdirectory(pytomography)
add_subdirectory(pytomography)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
