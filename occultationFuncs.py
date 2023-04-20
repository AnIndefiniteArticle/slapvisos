#!/usr/bin/env python # Functions for analysis of Planetary Occultaions 
import numpy as np
import pysis as ps
import matplotlib.pyplot as plt
from scipy.io import readsav
from scipy import stats

def readVIMSimaging(cubdir, cubfiles, ncubs, nspec, height, width, visible):
  cubdata  = np.zeros((ncubs, nspec, height, width))
  print("Reading in data from %d cubes"%ncubs)
  if visible:
    nspec0 = ps.CubeFile(cubdir+cubfiles[0]).shape[0]
    nspec1 = ps.CubeFile(cubdir+cubfiles[1]).shape[0]
    for i in range(ncubs):
      cubdata[i,:nspec1]               = ps.CubeFile(cubdir+cubfiles[2*i+1]).data
      cubdata[i, nspec1:nspec0+nspec1] = ps.CubeFile(cubdir+cubfiles[2*i  ]).data
  else:
    for i in range(ncubs):
      cubdata[i] = ps.CubeFile(cubdir+cubfiles[i]).data
  return cubdata

def flatField(cubdata, flatfield, mode, outdir):
  shape       = cubdata.shape
  center      = (flatfield.shape[1]//2, flatfield.shape[2]//2)
  print(center," Flat Field Center")
  print(cubdata.min(),cubdata.max())
  croppedFlat = flatfield[:, center[0]:center[0]+shape[2], center[1]:center[1]+shape[3]]
  plt.imshow(croppedFlat.mean(axis=0), interpolation="none", cmap="copper", origin='lower')
  plt.colorbar()
  plt.savefig(outdir+"/croppedflat.png")
  plt.clf()
  cubmean = cubdata.mean(axis=(0))[100:120].mean(axis=0)
  plt.imshow(cubmean, interpolation="none", cmap="copper", vmin = cubmean.min(), vmax = cubmean.mean(), origin='lower')
  plt.title("Star Peak at %f"%cubmean.max())
  plt.colorbar()
  plt.savefig(outdir+"/summeddata.png")
  plt.clf()
  flatcor     = cubdata / croppedFlat.reshape(1,shape[1],shape[2],shape[3])
  flatcormean = flatcor.mean(axis=(0))[100:120].mean(axis=0)
  plt.imshow(flatcormean, interpolation="none", cmap="copper", vmin = flatcormean.min(), vmax = flatcormean.mean(), origin='lower')
  plt.title("Star Peak at %f"%flatcormean.max())
  plt.colorbar()
  plt.savefig(outdir+"/flatcorrecteddata.png")
  plt.clf()
  return flatcor

def spatialBackgroundSignal(cubdata, start, end):
  if end == -1:
    end = cubdata.shape[0]
  data  = cubdata.copy()
  nspec =    data.shape[1]
  print("Applying spatial background correction under the assumption that it is an extra background source from after-occultation data")
  flatfield  = np.mean(data[start:end], axis=0)
  #flatfield /= np.nanmedian(flatfield, axis=(1,2)).reshape((nspec,1,1))
  flatcor    = cubdata - flatfield
  return flatcor
 

def spatialBackgroundSensitivity(cubdata, start, end):
  if end == -1:
    end = cubdata.shape[0]
  data  = cubdata.copy()
  nspec =    data.shape[1]
  print("Applying spatial background correction under the assumption that it is sensitivity-driven from after-occultation data")
  #flatfield  = np.zeros(data.shape[1:])
  #for i in range(start, end):
  #  for j in range(nspec):
  #    data[i][j][np.where(data[i][j] >= data[i][j].mean() + 2*data[i][j].std())] = np.nan
  flatfield  = np.mean(data[start:end], axis=0)
  flatfield /= np.nanmedian(flatfield, axis=(1,2)).reshape((nspec,1,1))
  flatcor    = cubdata / flatfield
  return flatcor

def temporalBackground(data, starpixx, starpixy, aper, normclip, zoomin):
  ncubs, nspec, height, width = data.shape
  sky = data.copy()
  # discard in-aperture pixels to create sky frames
  sky[:,:,starpixy[0]:starpixy[1],starpixx[0]:starpixx[1]] = np.nan
  # take median of sky
  sky = np.nanmedian(sky, axis=(2,3))
  # subtract the median sky-value
  cordata = data - sky.reshape(ncubs, nspec, 1,1)
  # Sum flux within aperture
  photometry = (cordata * aper).sum(axis=(2,3))
  # normalize this spectra to the pre-occultation (stellar) values
  normphotometry  = photometry / np.nanmedian(photometry[normclip:zoomin], axis=0)
  return cordata, normphotometry

def transitionfinder(list, window, Xpositive = True, Zpositive = True):
  # for the entire list
  for i in range(len(list)):
    # calculate the mode within a window
    mode = stats.mode(list[np.max((i-window,0)):np.min((i+window,len(list)))])
    # If it's the first entry, or if both the new mode is new AND represents over half of the array, add a new transition point
    if i == 0:
      transitions = np.array([[i, mode[0][0][0], mode[0][0][1]]])
    elif (mode[0][0][0] != transitions[-1][1] or mode[0][0][1] != transitions[-1][2]) and np.any(mode[1] > window):
      transitions = np.append(transitions, np.array([[i, mode[0][0][0], mode[0][0][1]]]), axis=0)

  return transitions

def prfmetric(PRFfile, pixelSize=(0.25,0.5))
  """
  Calculate metric for PRF scan

  Compares PRF scan values one pixel width apart in each direction

  Parameters
  ----------
  PRFfile : string
      File location for PRF scan
  pixelSize : 2-tuple 
      Width of pixel to in each direction (as determined by mirror motion /
      look angle, not response / sensitivity)

  Returns
  -------
  Xscanmetrics : ndarray
      shape: (number of scans, number of points per scan, 4), with the 4
      representing [Xposition, Zposition, metric measured to left, metric
      measured to right]
  Zscanmetrics : ndarray
      shape: (number of scans, number of points per scan, 4), with the 4
      representing [Xposition, Zposition, metric measured up, metric measured
      down]
  """
  # Read in PRF file and allocate arrays
  PRFs     = readsav(PRFfile)
  XscanVal = PRFs['PRF1'] 
  XscanXs  = PRFs['XPOS1']
  XscanZs  = PRFs['ZPOS1']
  Xscans   = np.column_stack((XscanXs, XscanZx, XscanVal))
  ZscanVal = PRFs['PRF2'] 
  ZscanXs  = PRFs['XPOS2']
  ZscanZs  = PRFs['ZPOS2']
  Zscans   = np.column_stack((ZscanXs, ZscanZx, ZscanVal))

  # calculate index of position closest to one pixel in either direction
  for scans in (Xscans, Zscans):
    
  
def twopixcenters(data, transitions, PRFfile, Xwidth, Zwidth):
  frames   = data / data.max()
  centers  = np.zeros((len(frames),2))
  #metricreturns = np.zeros((len(frames), 142))
  
  step = 0
  for i in range(len(frames)):
    # increment steps with transitions
    if step < len(transitions)-1 and transitions[step+1][0] == i:
      step += 1

    # declare current pixel
    currentX = transitions[step][2]
    currentZ = transitions[step][1]

    # Determine whether to compare to transition before or after current one
    #if step == 0:
    #  centers[i,0] = currentZ
    #  centers[i,1] = currentX
    #  continue
    if step != 0 and (step == len(transitions)-1 or transitions[step+1][0] - i > i - transitions[step][0]):
      compareX = transitions[step-1][2]
      compareZ = transitions[step-1][1]
    elif step == 0:
      compareX = (transitions[step][2]+1) % len(frames[0])
      #print("Is this X? %d" % len(frames[0]))
      compareZ = (transitions[step][1]+1) % len(frames[0][0])
      #print("Is this Z? %d" % len(frames[0][0]))
    else:
      compareX = transitions[step+1][2]
      compareZ = transitions[step+1][1]

    if compareX > currentX:
      Left = False
    else:
      Left = True

    Ximagemetric = (frames[i][compareZ,compareX]) / frames[i][currentZ,currentX] #frames[i].max()
    # using the scan at index 7
    Xscansmetric = np.roll(XscanVal[:,11], (-1**(Left+1))*284) / XscanVal[:,11]
    
    if Left:
      Xmetriccompare = (Ximagemetric - Xscansmetric[142:284])**2
    else:
      Xmetriccompare = (Ximagemetric - Xscansmetric[284:-142])**2
    #metricreturns[i] = Xmetriccompare
    #print(Ximagemetric)
    #print(XscanXs[np.where(Xmetriccompare == Xmetriccompare.min())+142, 7])
    #print(np.where(Xmetriccompare == Xmetriccompare.min()))
    try:
      Zcorr = 0#(XscanZs[np.where(Xmetriccompare == Xmetriccompare.min())+142,7]/Zwidth)
    except:
      Zcorr = (XscanZs[np.where(Xmetriccompare == Xmetriccompare.min())[0][0]+(142*(2-Left)),7]/(2*Zwidth))
    try:
      Xcorr = (XscanXs[np.where(Xmetriccompare == Xmetriccompare.min())+(142*(2-Left)),7]/(2*Xwidth))
    except:
      Xcorr = (XscanXs[np.where(Xmetriccompare == Xmetriccompare.min())[0][0]+(142*(2-Left)),7]/(2*Xwidth))

    centers[i,0] = currentZ + Zcorr
    centers[i,1] = currentX + Xcorr

  return centers#, metricreturns
