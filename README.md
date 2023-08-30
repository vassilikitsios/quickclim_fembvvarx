# quickclim_fembvvarx

A machine learning approach to rapidly project climate responses under a multitude of carbon concentration pathways.

## Overview

The international Coupled Model Inter-comparison Projects, simulate a small selection of potential future carbon concentration pathways using numerous climate models. These simulations provide climate data key to assessing physical risk. We have developed a machine learning approach that exploits these datasets, to rapidly generate output mimicking the climate simulations. This is done at a fraction of the computational cost. We refer to this approach as QuickClim. With QuickClim one now has the ability to rapdily simulate a multitude of potential future climate scenarios. This enables a broader assessment of the potential future climates, and the associated risks.


## Directory Structure

The directories in this repo are as follows:

quickclim_fembvvarx.ipynb
        Jupyter notebook containing all of the QuickClim and plotting code to reproduce the results in the manuscript.

./images/
        All figures produced in the notebook are output to PDF here

./inputs/
        Contains the CO2e concentration files (*.dat) and pre-processed CMIP5 inputs (ds_model.nc) required to reproduce the results in section 3 of the notebook.
        One can instead obtain the original CMIP5 data directly from https://pcmdi.llnl.gov/mips/cmip5/data-portal.html.

./results/
        Pre-calculated outputs using QuickClim required to reproduce the figures in section 4 and section 5 of the notebook.

./src/
        Third party software required to run the FEM-BV-VARX code. Users of the software in this directory should cite the following:
                paper - Quinn, C., D. Harries, and T. J. O’Kane, 2021: Dynamical Analysis of a Reduced Model for the North Atlantic Oscillation. J. Atmos. Sci., 78, 1647–1671, https://doi.org/10.1175/JAS-D-20-0282.1.
                code - https://zenodo.org/record/4035644

## License

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
