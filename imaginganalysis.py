#!/usr/bin/env python3
# Filename: analysis.py
# Produces plots of saturn occultations from .cub files

# prep:
# put .cub files in a directory called cubfiles, with a cubs.txt file listing all desired files
# mkdir -p ./figs/{spectra,frames} NOTE: there are more directories needed now, grep this file for savefigs to find them all

#TODO: use times instead of cube numbers in plots

import pysis as ps
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import readsav
import occultationFuncs as oF
from datetime import datetime
import os
import shutil
from config import *
import sys

# DEBUGGING STATEMENTS delete when arguments are confirmed to have been passed correctly
print(sys.argv)
sys.exit()

# read in outdir as an argument
try:
  outdir = sys.argv[1]
except:
  print("invalid output directory (passed as first and only argument)")
  sys.exit(1)

# Welcome Message
print("\nBeginning analysis of "+occname+"\n")

  #####################
  #     READ IN DATA  #
  #####################

# Get cube-size and allocate space
shape    = ps.CubeFile(cubdir+cubfiles[0]).shape
if visible:
  print("Visible and IR cubes")
  ncubs    = np.int64(len(cubfiles)/2)
  nspec    = shape[0] + ps.CubeFile(cubdir+cubfiles[1]).shape[0]
else:
  print("IR cubes")
  ncubs    = len(cubfiles)
  nspec    = shape[0]
height   = shape[1]
width    = shape[2]
maxpix   = np.max((height,width))
# Read in data, or load from save file
try:
  cubdata = np.load(occname+"data.npy")
  print("loaded previous save file")
except:
  cubdata = oF.readVIMSimaging(cubdir, cubfiles, ncubs, nspec, height, width, visible)
  np.save(occname+"data.npy", cubdata)

# Get pixel dimensions
if ps.CubeFile(cubdir+cubfiles[0]).label['IsisCube']['Instrument']['SamplingMode'] == 'HI-RES':
  Xpixelwidth = 0.25 # mr
  Zpixelwidth = 0.5  # mr
  flatfield   = ps.CubeFile(flatdir+"ir_hires_flatfield_v0002.cub").data
  mode        = "HiRes"
  print("High-Resolution Frames")
else:
  Xpixelwidth = Zpixelwidth = 0.5 # mr
  flatfield   = ps.CubeFile(flatdir+"ir_flatfield_v0002.cub").data
  mode        = "LoRes"
  print("Low-Resolution Frames")

# read in PRF Scans
PRFs     = readsav(PRFfile)
XscanVal = PRFs['PRF1']
XscanXs  = PRFs['XPOS1']
XscanZs  = PRFs['ZPOS1']
ZscanVal = PRFs['PRF2']
ZscanXs  = PRFs['XPOS2']
ZscanZs  = PRFs['ZPOS2']

  #####################
  #  CREATE APERTURE  #
  #####################

# Square aperture set by starpix variable in config file
aper = np.zeros(cubdata.shape) 
aper[:,:,starpixy[0]:starpixy[1], starpixx[0]:starpixx[1]] += 1
#aper[:,:,2] = 1

  #####################
  #  CENTERING ALLOC  #
  #####################

# Allocate space for centering frames
indices                    = np.indices((height,width))
nconts                     = len(continua)
brightestPixel	           = np.zeros((ncubs, 2, nconts))
binbrightestPixels         = np.zeros((ncubs, 2, nconts))
stretchbrightestPixel      = np.zeros((ncubs, 2, nconts))
binstretchbrightestPixels  = np.zeros((ncubs, 2, nconts))
brightestPixel             = np.zeros((brightestPixel.shape)).astype(np.int64)
pixeltransitions           = [None]*nconts
bettercenters              = brightestPixel.copy().astype(np.float64)

  #####################
  #  GENERATE SPECTRA #
  #####################

# Perform Background Corrections
print("Performing background corrections")

#cubdata -= cubdata.min()

# flatfield is corrected first
flatcor = oF.flatField(cubdata, flatfield, mode, outdir)
# vims flatfield seems to already be corrected out

# Subtract spatial background gradient
if spaback == "Sensitivity":
  spatcor = oF.spatialBackgroundSensitivity(cubdata, zoomax, -1)
elif spaback == "Additive":
  spatcor = oF.spatialBackgroundSignal(cubdata, zoomax, -1)

# Subtract sky level
cordata, normphotometry = oF.temporalBackground(spatcor, starpixx, starpixy, aper, normclip, zoomin)

  #####################
  # FOR EACH CONTINUA #
  #####################

# FOR LOOP OVER CONTINUA SEGMENTS
for i in range(nconts):
  print("continuum band %d"%i)
  # Set the continuum array of channels to the first entry in the continua tuple from config
  continuum = continua[i]

  #####################
  #     MAKE FRAMES   #
  #####################

  # Create frames by summing over desired wavelength channels
  frames  = cordata[:,continuum].sum(axis=1)

  #####################
  # MAKE SUMMED FRAME #
  #####################
  if backgroundcheck == True:
    # Sum frames and plt.imshow
    summedframe = frames.sum(axis=0)
    sfmin       = summedframe.min()
    sfmax       = summedframe.max()
    print(sfmin,sfmax)
    print("Summed frame for continuum %d ranges from %d to %d DN"%(i,sfmin,sfmax))
    plt.imshow(summedframe, interpolation='none', vmin=sfmin, vmax=-sfmin/2, cmap='copper')
    plt.colorbar(extend="both")
    plt.xlim(-0.5,width-0.5)
    plt.ylim(-0.5,height-0.5)
    plt.title("%s Summed Frame, %f-%f microns"%(occname, lambdas[continua[i][0]], lambdas[continua[i][-1]]))
    plt.savefig(outdir+'/summedframecont%d.png'%(i))
    plt.clf()
    plt.close()


  # Making array of smoothed frames
  smoothframes = np.zeros(frames.shape)
  for j in range(len(frames)):
    # left-edge
    if j-smooth < 0:
      smoothframes[j] = frames[       0:j+smooth].sum(axis=0)
    # right-edge
    elif j+smooth > len(frames)-1:
      smoothframes[j] = frames[j-smooth:      -1].sum(axis=0)
    # smoothing with adjacent frames
    else:
      smoothframes[j] = frames[j-smooth:j+smooth].sum(axis=0)

  #####################
  #  CENTER  FINDING  #
  #####################
  
    # Brightest Pixel Method (with smoothed frames)
    if skipcol1 == True:
      try:
        brightestPixel[j,:,i] = np.where(smoothframes[j,] == smoothframes[j,:,1:].max())
      except:
        brightestPixel[j,:,i] = (np.where(smoothframes[j,] == smoothframes[j,:,1:].max())[0][0], np.where(smoothframes[j,] == smoothframes[j,:,1:].max())[1][0])
    else:
      try: 
        brightestPixel[j,:,i] = np.where(smoothframes[j] == smoothframes[j].max())
      except:
        brightestPixel[j,:,i] = (np.where(smoothframes[j] == smoothframes[j].max())[0][0], np.where(smoothframes[j] == smoothframes[j].max())[1][0])

  pixeltransitions[i]         = oF.transitionfinder(brightestPixel[:,:,i], transwindow)
  bettercenters[:,:,i]        = oF.twopixcenters(smoothframes, pixeltransitions[i], PRFs, Xpixelwidth, Zpixelwidth)

#####################
#       PLOTS       #
#####################

  #####################
  #  LIGHTCURVE PLOTS # Brightest Pixel Centering
  #####################

  # Make Lightcurves
  print("Making Lightcurve and Centering plots for Each Channel")
  plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
  plt.rcParams.update({'font.size': fontsize})
  # Unzoomed lightcurve
  plt.subplot(2,1,1)
  plt.plot(normphotometry[:,continuum].sum(axis=1)/len(continuum))
  plt.title("%s %f:%f micron Lightcurve"%(occname, lambdas[continua[i][0]], lambdas[continua[i][-1]]))
  plt.xlabel("Cube Number")
  plt.ylabel("Summed Brightness")

  # zoomed lightcurve
  #plt.subplot(3,2,2)
  #plt.plot(np.arange(zoomin,zoomax), normphotometry[zoomin:zoomax,continuum].sum(axis=1)/len(continuum))
  #plt.title("%s %f:%f micron Lightcurve"%(occname, lambdas[continua[i][0]], lambdas[continua[i][-1]]))
  #plt.xlabel("Cube Number")
  #plt.ylabel("Summed Brightness")

  #####################
  # Z CENTERING PLOTS # Brightest Pixel Centering
  #####################
  ''' 
  # Plot centering
  plt.subplot(3,1,2)
  plt.plot(brightestPixel[:,0,i]*Zpixelwidth, 'b.')
  plt.plot(bettercenters[:,0,i]*Zpixelwidth, 'r.')
  print(pixeltransitions[i])
  plt.plot(pixeltransitions[i][:,0], pixeltransitions[i][:,1]*Zpixelwidth, 'k+')
  plt.ylim(-0.5*Zpixelwidth, height*Zpixelwidth)
  plt.title("%s Z Center v Time, %f:%f microns"%(occname, lambdas[continua[i][0]], lambdas[continua[i][-1]]))
  plt.xlabel("Frame Number")
  plt.ylabel("Star's Z Positon, mrad from edge of frame")
  '''
  # zoomed centering
  #plt.subplot(3,2,4)
  #plt.plot(np.arange(zoomin,zoomax), brightestPixel[zoomin:zoomax,0, i]*Zpixelwidth, 'b.')
  #plt.plot(np.arange(zoomin,zoomax), bettercenters[zoomin:zoomax,0, i]*Zpixelwidth, 'r,')
  #plt.plot(pixeltransitions[i][:,0], pixeltransitions[i][:,1]*Zpixelwidth, 'k+')
  #plt.ylim(-0.5*Zpixelwidth, height*Zpixelwidth)
  #plt.title("%s Z Center v Time - Zoomed, %f:%f microns"%(occname, lambdas[continua[i][0]], lambdas[continua[i][-1]]))
  #plt.xlabel("Frame Number")
  #plt.ylabel("Star's Z Position, mrad from edge of frame")

  #####################
  # X CENTERING PLOTS # Brightest Pixel Centering
  #####################
  
  # Plot centering
  plt.subplot(2,1,2)
  plt.plot(brightestPixel[:,1,i]*Xpixelwidth, 'g.')
  plt.plot(bettercenters[:,1,i]*Xpixelwidth, 'r.')
  plt.plot(pixeltransitions[i][:,0], pixeltransitions[i][:,2]*Xpixelwidth, 'k+')
  x = np.arange(len(brightestPixel))
  plt.plot(x, slope/1000 * x + offset, 'k')
  plt.ylim(-0.5*Xpixelwidth, width*Xpixelwidth)
  plt.title("%s X Center v Time, %f:%f microns"%(occname, lambdas[continua[i][0]], lambdas[continua[i][-1]]))
  plt.xlabel("Frame Number")
  plt.ylabel("Star's X Position, mrad from edge of frame")

  # zoomed centering
  #plt.subplot(3,2,6)
  #plt.plot(np.arange(zoomin,zoomax), brightestPixel[zoomin:zoomax,1, i]*Xpixelwidth, 'g,')
  #plt.plot(np.arange(zoomin,zoomax), bettercenters[zoomin:zoomax,1, i]*Xpixelwidth, 'r,')
  #plt.plot(pixeltransitions[i][:,0], pixeltransitions[i][:,2]*Xpixelwidth, 'k+')
  #plt.plot(x[zoomin:zoomax], slope/1000 * x[zoomin:zoomax] + offset, 'k')
  #plt.ylim(-0.5*Xpixelwidth, width*Xpixelwidth)
  #plt.title("%s X Center v Time - Zoomed, %f:%f microns"%(occname, lambdas[continua[i][0]], lambdas[continua[i][-1]]))
  #plt.xlabel("Frame Number")
  #plt.ylabel("Star's X Position, mrad from edge of frame")

  plt.savefig(outdir+"/BrightestPixelcenteringphotomcont%d.png"%i)
  plt.clf()
  plt.close()


  #####################
  # DIFF CENTER PLOTS #
  #####################

  # Plot differential centering
  print("Differential Centering Plots")
  for k in range(nconts):
    if i>k:
      plt.plot(brightestPixel[:,0,i] - brightestPixel[:,0,k], 'b.', label="Z Position")
      plt.plot(brightestPixel[:,1,i] - brightestPixel[:,1,k], 'g.', label="X Position")
      plt.legend(loc="upper left")
      plt.ylim(-5,5)
      plt.title("%s Center v Time %f:%f-%f:%f microns"%(occname, lambdas[continua[i][0]], lambdas[continua[i][-1]], lambdas[continua[k][0]], lambdas[continua[k][-1]]))
      plt.xlabel("Frame Number")
      plt.ylabel("Differential Centering Position")
      plt.savefig(outdir+"/differentialcentering%d-%d.png"%(i,k))
      plt.clf()
      plt.close()
      plt.plot(binbrightestPixels[:,0,i] - binbrightestPixels[:,0,k], 'b.', label="Z Binned")
      plt.plot(binbrightestPixels[:,1,i] - binbrightestPixels[:,1,k], 'g.', label="X Binned")
      plt.legend(loc="upper left")
      plt.ylim(-5,5)
      plt.title("%s Center v Time %f:%f-%f:%f microns"%(occname, lambdas[continua[i][0]], lambdas[continua[i][-1]], lambdas[continua[k][0]], lambdas[continua[k][-1]]))
      plt.xlabel("Frame Number")
      plt.ylabel("Differential Binned Centering Position")
      plt.savefig(outdir+"/binneddifferentialcentering%d-%d.png"%(i,k))
      plt.clf()
      plt.close()

  #####################
  # MOVIE             #
  #####################

  if movies:
    # Make all of the frames all-positive STRICTLY for plotting and stretching purposes
    # NOTE that this is the same shift for every frame to ensure that the animation stays faithful
    smoothframes         -= smoothframes.min()
    #gamma=1
    # Find minimum and maximum values for consistent brightestPixelorscaling
    framin = smoothframes.min()
    framax = smoothframes.max()
    # Make Image movie
    print("Creating Movie of Continuum Wavelengths, value range %d, %d"%(framin, framax))
    os.mkdir(outdir+'/framescont%d'%(i))
    for j in range(ncubs):
      plt.imshow(smoothframes[j]**gamma, interpolation='None', vmin=framin**gamma, vmax=framax**gamma, cmap='copper')
      plt.colorbar()
      plt.plot(brightestPixel[j][1][i],brightestPixel[j][0][i], 'r+')
      plt.xlim(-0.5,width-0.5)
      plt.ylim(-0.5,height-0.5)
      plt.title("%s Frame: %04d"%(occname,j))
      plt.savefig(outdir+'/framescont%d/frame%04d.png'%(i,j))
      plt.clf()
      plt.close()

  #####################
  # SPECTRUM PLOT     #
  #####################

# Make Spectrum plot
print("Making Spectrum Plots")
plt.imshow(normphotometry.transpose(), vmin=0, vmax=2, interpolation='none', extent=[0,ncubs, lambdas.max(), lambdas.min()], aspect='auto', cmap='copper')
plt.colorbar()
for q in range(nconts):
  plt.plot((0,ncubs), (lambdas[continua[q][ 0]],lambdas[continua[q][ 0]]), 'c')
  plt.plot((0,ncubs), (lambdas[continua[q][-1]],lambdas[continua[q][-1]]), 'y')
plt.title("Spectrum %s"%occname)
plt.ylabel("Wavelength, microns")
plt.xlabel("Cube Number")
plt.savefig(outdir+'/occultation.png')
plt.clf()
plt.close()
plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
plt.imshow(normphotometry[zoomin:zoomax].transpose(), vmin=0, vmax=2, interpolation='none', extent=[zoomin, zoomax, lambdas.max(), lambdas.min()], aspect='auto', cmap='copper')
plt.colorbar()
for q in range(nconts):
  plt.plot((zoomin, zoomax), (lambdas[continua[q][ 0]],lambdas[continua[q][ 0]]), 'c')
  plt.plot((zoomin, zoomax), (lambdas[continua[q][-1]],lambdas[continua[q][-1]]), 'y')
plt.title("Spectrum - Zoomed %s"%occname)
plt.ylabel("Wavelength, microns")
plt.xlabel("Cube Number")
plt.savefig(outdir+'/occultation-zoom.png')
plt.clf()
plt.close()

  #####################
  # SPECTRUM MOVIE    #
  #####################

if movies:
  # Make Spectrum movie
  print("Making Spectrum Movie")
  os.mkdir(outdir+'/spectra')
  for j in range(ncubs):
    plt.plot(lambdas, normphotometry[j])
    for q in range(nconts):
      plt.plot((lambdas[continua[q][ 0]],lambdas[continua[q][ 0]]), (0,2), 'c')
      plt.plot((lambdas[continua[q][-1]],lambdas[continua[q][-1]]), (0,2), 'y')
    plt.title("%s Frame: %04d"%(occname,j))
    plt.ylim(0,2)
    plt.xlim(lambdas.min(),lambdas.max())
    plt.savefig(outdir+"/spectra/spectra%04d.png"%(j))
    plt.clf()
    plt.close()

print("./analysis.py complete")
