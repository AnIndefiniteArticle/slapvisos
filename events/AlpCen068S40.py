#!/usr/bin/python3

import numpy as np
occname  = "AlpCen068S40"
# What are the wavelengths?
lambdas  = np.loadtxt('/data/vimsSaturnOccs/data/lambdavis.txt')[-352:]
# Location of Files
cubdir   = "../data/cubs/"+occname+"/"
cubfiles = open(cubdir+'/cubs.txt').read().splitlines()
flatdir  = "../data/FlatField/"
PRFfile  = "../data/PRFscans/makePRF270.sav"
# SAMPLE CHANNEL FOR MOVIE
visible  = True
continua = (np.arange(90), np.arange(96,106), np.arange(156,174), np.arange(196,216), np.arange(242,255), np.arange(285,316))
# STAR LOCATIONS TODO: automate by making aperture around brightest pixel
starpixx = (3,11)
starpixy = (0,3)
slope    = 3.37 # urad/cube
offset   = -5    # mrad
# Debugging
backgroundcheck = True
skipcol1        = False
# Spatial Background
spaback  = "Additive" #"Sensitivity" # "Additive"
# clipping and binning
normclip =   0 # number of frames to clip for normalization of spectrum (if first few are bad)
binning  =  10 # temporal binning for binned spectra
smooth   =   1 # Number of frames to smooth together on either side of the current one (rolling average) for center-finding algorithm
transwindow = 20 # window size for pixel transitions finder
# CUBES TO ZOOM IN ON TODO: automate by finding region of high stddev?
zoomin   = 100
zoomax   = 200
# MOVIES?
movies	 = True
gamma    = 0.3
# Figure size settings
figsize=(30,30)
dpi=300
fontsize=28
prfplots=False
