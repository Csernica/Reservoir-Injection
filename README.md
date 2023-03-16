# Reservoir-Injection

This repository is split into 3 sets of files, as follows:

1) the "DataAnalyzerMN" and "DataScreen" files read in and screen for common errors (respectively) in input .txt or .isox files. We have two versions; one for FTStat and one for IsoX. We recommend the IsoX version be used for any future work. The IsoX version is used to process Perchlorate, while the FTStat is for methionine and acetone. 

2) The 'readAcetone', 'readPerchlorate', and 'readMethionine' .py files. Each reads in the relevant data (files can be found on the Caltech Data repository cited in the paper) and calculates basic .json outputs. 

3) the 'Acetone Figures', 'Perchlorate Figures', and 'Methionine Figures and Anomaly Screening Figures' .ipynb generate the figures presented in the text from the input files. By downloading the data files and running this scripts, one can recreate and modify/investigate any of the results presented in the paper. 
