#-----------------------------------------------------------------------------
set(MODULE_NAME SlicerSPECTRecon)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  Logic/algorithms.py
  Logic/getmetadatautils.py
  Logic/likelihood.py
  Logic/priors.py
  Logic/simindToDicom.py
  Logic/SlicerSPECTReconLogic.py
  Logic/SlicerSPECTReconTest.py
  Logic/systemMatrix.py
  Logic/transforms.py
  Logic/volumeutils.py
  Logic/vtkkmrmlutils.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/Attenuation_maps/attenuation_map_lower.pt
  Resources/Attenuation_maps/attenuation_map_upper.pt
  Resources/DICOM/bed1_projections.dcm
  Resources/DICOM/bed2_projections.dcm
  Resources/algorithmTestSettings.json
  Resources/psfMeta.json
  Resources/sampleDataMetaData.json
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  # slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Logic)
endif()
