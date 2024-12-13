# Contributing to SlicerSPECTRecon

First off, thanks for your interest in contributing! There are many potential options for contributions to this library: most notably, implementation of conversion from alternative sources to dicom format, implementation of different prior functions, implementation of additional reconstruction algorithms, etc. If you wish to contribute:

**Bugs**
1. Open an issue. 
2. Provide as much context as you can about what you're running into.
3. Provide project and platform versions.

**New Features**

The recommended method of contributing is as follows:
1. Create an issue on the [issues page](https://github.com/PyTomography/slicer_spect_recon/issues)
2. Fork the repository to your own account
3. Fix issue and push changes to your own fork on GitHub
4. Create a pull request from your fork (whatever branch you worked on) to the development branch in the main repository.

## Setup

To activate and use this module please follow these steps:

   1.  Either `unzip` the source code or `clone` it from GitHub.
   2.  Add the respective path in `Slicer` by *Edit -> Application settings -> Modules -> Additional module paths -> Add -> OK*.
   3.  `Restart` Slicer to apply the changes.
   4.  `Choose` the module under *Modules -> Tomographic Reconstruction -> SlicerSPECTRecon*.

## Testing

The tests in `Logic/SlicerSPECTReconTest` can be run in `Slicer` itself. Therefore, the module has to be already imported and activated.

Additionally, you have to enable the `developer mode` under *Edit -> Application settings -> Developer -> Enable developer mode -> OK*.

After reloading Slicer, you should see a column `Reload and Test`. To run the tests and changes to the source code, just click this button. This reimports the module with the current changes and runs the code against the given tests.

## Sample Data

The links to the example data (sample patient and simind files) are in the sample_data.txt file in the `Resources` folder. 

## Contribute

If you'd like to contribute, you can find an orientation on the Slicer [documentation for developers](https://www.slicer.org/wiki/Documentation/Nightly/Developers).

## License

SlicerSPECTRecon is subject to the `MIT License`, which can be found in the project's root.
