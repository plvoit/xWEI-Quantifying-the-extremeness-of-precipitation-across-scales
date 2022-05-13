# WEI and xWEI calculation 


doi of this repo:  
*will be added soon for first release*

doi of according publication:  
*XXXXXXXXXXXXXXXXX

This repository provides example model code according to: 

XXX Change Paper title
**Voit, P. ; Heistermann, M.: Quantifying .....**

Contact: [voit@uni-potsdam.de](voit@uni-potsdam.de)

ORCIDs of first author:   
P. Voit:  [0000-0003-1005-0979](https://orcid.org/0000-0003-1005-0979)   
 

For a detailed description please refer to the publication.

This repository contains examplary code and data to calculate the WEI and xWEI for all three methods of GEV-parameter
estimation (cellwise-GEV, Region-of-interest, duration dependent GEV) for one selected event.

# Installation
The code was implemented in `Python 3.9`. Used packages are Numpy (Harris et al., 2020), `Pandas`
(McKinney, 2010), `Scikit-Learn` (Pedregosa et al., 2011), `Matplotlib` (Hunter, 2007),
`xarray` (Hoyer and Hamann, 2017) and `netcdf4`. To view the notebooks you need to have jupyter installed.  
The user can use the included `xwei_env.yaml` file to create a conda environment which includes all the necessary
packages. This can be done by:  
`conda config --add channels conda-forge`  
`conda config --set channel_priority strict`  
`conda env create -f xwei_env.yml`

Once the environment is installed it needs to be activated. Start the terminal of your choice and type:
`conda activate xwei_env`  
Then start Jupyter Notebook by typing:
`jupyter notebook`  
Now you can select and run the supplied notebooks in the browser.  
Alternatively you can manually install all the necessary packages without using a conda environment but this way it
can not be guaranteed that the packages work correctly.

# Included files
## example_xwei.ipynb
In this notebook we demonstrate the application of WEI and xWEI using three different methods
(cellwise GEV, ROI, dGEV) for an examplary rainfall event (CatRaRE ID: 16058). This event caused
the massive floodings in Braunsbach in May 2017. For methods the WEI and xWEI is computed and plotted.
The used functions can be found in the script "xwei_functions.py".

## roi_parameter_fitting.ipynb
This contains the functions which we used to derive the GEV parameters with Region-of-Interest method
according to Burn (1990). Because this calculation is quite time consuming we supplied the parameter sets in
"data/roi_parameters". The results achieved with this script will slightly differ from the parameter sets
that we supplied. This is because we just use a small subset for one event here, 
instead of whole Germany and therfore the ROI on the sides of the subset  contains no values.
Running this script could take about 1.5 hours.

## xwei_functiony.py
This script contains all the functions that are used to calculate and visualize the WEI according to
Müller and Kaspar (2014) and xWEI (Voit & Heistermann,XX).
The supplied functions can be applied to any 2D or 3D-data xarrays.
Documentation is included.

## xwei_env.yml
This file can be used to create a conda environment that includes all the necessary scripts to run the supplied
scripts.

# Included Data
## event_16058_200.nc
This netcdf is a subset of the RADKLIM dataset (precipitation in hourly resolution) of 200 km bounding box around the centroid of the event
(as specified in CatRaRE) from May 28 00:50 to 31 May 17:50. These extended temporal boundaries (compared to 
the ones specified in CatRaRE) allow for a moving window 72h-aggregation for every hour of the event.
This file is used in the process of calculation the EtA for every duration and for every hour of the event.

## yearmax_2001_2020
This folder contains the yearly maxima of the RADKLIM dataset in netcdf format for the area of event_16058. This files can
be used to derive the GEV parameters with all three methods.

## dgev_parameters
This folder contains the dGEV parameters derived with the R-package "IDF" (Ulrich et al., 2019) for the region of
the event_16058.

## gev_parameters
This folder contains the GEV parameters derived with the R-package "extRemes" (Gilleland and Katz, 2016) for the region of
the event_16058.

## roi_parameters
This folder contains the GEV parameters for the region of the event_16058. This parameters were achieved
by using the functionalities contained in "roi_parameter_fitting.ipynb" for the whole RADKLIM dataset which
were then subsetted for the region of the event.

# References
Burn, D. H.: Evaluation of regional flood frequency analysis with a region of influence approach, Water Resources Research, 26, 2257–2265,
publisher: Wiley Online Library, 1990.

Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.

Hoyer, S. & Hamman, J., (2017). xarray: N-D labeled Arrays and Datasets in Python. Journal of Open Research Software. 5(1), p.10. DOI: https://doi.org/10.5334/jors.148

Data structures for statistical computing in python, McKinney, Proceedings of the 9th Python in Science Conference, Volume 445, 2010

Müller, M. and Kaspar, M.: Event-adjusted evaluation of weather and climate extremes, Natural Hazards and Earth System Sciences, 14,
473–483, publisher: Copernicus GmbH, 2014

Pedregosa, F. et al., 2011. Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), pp.2825–2830.

Ulrich, J., Ritschel, C., Mack, L., Jurado, O. E., Fauer, F. S., Detring, C., and Joedicke, S.: IDF: Estimation and Plotting of IDF Curves, R
package version, 2, 2019.

Gilleland, E. and Katz, R. W.: extRemes 2.0: an extreme value analysis package in R, Journal of Statistical Software, 72, 1–39, 2016.


