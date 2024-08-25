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
  Logic/simindToDicomConverterTest.py
  Logic/testutils_builder.py
  Logic/dicomvalues.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
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
