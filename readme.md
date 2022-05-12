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
Two Notebooks are included.  
DESCREIBE NOTEBOOKS

All the functions, including documentation, that are used are included in the file "xwei_functions.py".

The user can use the included .yaml file to create a conda environment which includes all the necessary
packages. This can be done by:
conda env create -f xwei.yml

The code was implemented in Python 3.9. Used packages are Numpy (van der Walt et al., 2011), Pandas 
(McKinney, 2010; Reback et al., 2020),
Scikit-Learn (Pedregosa et al., 2011), Matplotlib (Hunter, 2007), xarray (Hoyer and Hamann, 2017) and netcdf4. To view the
notebooks you need to have jupyter installed.

