#!/usr/bin/python3

import numpy as np
occname  = "AlpOri270S99"
# What are the wavelengths?
lambdas  = np.loadtxt('/data/vimsSaturnOccs/data/lambda.txt')[-256:]
# Location of Files
cubdir   = "../data/cubs/"+occname+"/"
cubfiles = open(cubdir+'/cubs.txt').read().splitlines()
flatdir  = "../data/FlatField/"
PRFfile  = "../data/PRFscans/makePRF270.sav"
# SAMPLE CHANNEL FOR MOVIE
visible  = False
continua = (np.arange(10), np.arange(60,78), np.arange(100,120), np.arange(146,159), np.arange(189,220))
# STAR LOCATIONS TODO: automate by making aperture around brightest pixel
starpixx = (1,6)
starpixy = (1,5)
slope    = 4.09 # urad/cube
offset   = -5    # mrad
# Debugging
backgroundcheck = True
skipcol1        = True
# Spatial Background
spaback  = "Additive" #"Sensitivity" # "Additive"
# clipping and binning
normclip = 250 # number of frames to clip for normalization of spectrum (if first few are bad)
binning  =  10 # temporal binning for binned spectra
smooth   =   1 # Number of frames to smooth together on either side of the current one (rolling average) for center-finding algorithm
transwindow = 20 # window size for pixel transitions finder
# CUBES TO ZOOM IN ON TODO: automate by finding region of high stddev?
zoomin   = 500
zoomax   = 750
# MOVIES?
movies	 = True
gamma    = 0.3
# Figure size settings
figsize=(30,30)
dpi=300
fontsize=28
