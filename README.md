# CaImAn-API
Python API for the CaImAn package. Adaptation of the 2-photon data analysis pipeline. To download the CaImAn package, see https://github.com/flatironinstitute/CaImAn.
### Requirements
See import statements in `caimanAPI.py`.
### Example code
The `CaImAn` class allows the user to process microscopy data using the motion correction and segmentation algorithms efficiently, with very few code lines. Default parameters are included within the `CaImAn` class and should be modified accordingly to fit data.

```
caiman = CaImAn('/path/', 'data.tif')
caiman.correctMotion()
caiman.saveFilm('corrected.tif')
caiman.computeSegmentation()
caiman.saveResults()
caiman.deleteMemoryMappedFiles()
caiman.stopCluster()
```
### Credits
Code and examples by Antoine Légaré. All credits go to the CaImAn developpers.
