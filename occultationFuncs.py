#!/usr/bin/env python # Functions for analysis of Planetary Occultaions 
import numpy as np
import pysis as ps
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
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

def prfmetric(PRFfile, pixelSize=(0.25,0.5), Plots=False, figsize=(15,15), dpi=300, fontsize=12, markersize=0.5, outdir='outputs/PRFscanplots/'):
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
  Plots : boolean
      Create diagnostic plots?
  figsize : 2-tuple
      width/height of diagnostic plots
  dpi : int
      dots-per-inch resolution of diagnostic plots
  fontsize : int
      size of plot fonts in pt

  Returns
  -------
  Xscanmetrics : ndarray
      shape: (number of scans, number of points per scan, 5), with the 5
      representing [Xposition, Zposition, nominal value, metric measured to left, metric
      measured to right]
  Zscanmetrics : ndarray
      shape: (number of scans, number of points per scan, 5), with the 5
      representing [Xposition, Zposition, nominal value, metric measured up, metric measured
      down]
  """
  # Read in PRF file and allocate arrays
  PRFs     = readsav(PRFfile)
  XscanVal = PRFs['PRF1'] 
  XscanXs  = PRFs['XPOS1']
  XscanZs  = PRFs['ZPOS1']
  Xscans   = np.stack((XscanXs, XscanZs, XscanVal), axis=2)
  ZscanVal = PRFs['PRF2'] 
  ZscanXs  = PRFs['XPOS2']
  ZscanZs  = PRFs['ZPOS2']
  Zscans   = np.stack((ZscanXs, ZscanZs, ZscanVal), axis=2)
  plt.rcParams.update({'font.size': fontsize})

  if Plots:
    # Plot positions of all scans over a rectangle representing the nominal pixel position
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.scatter(Xscans[:,:,0], Xscans[:,:,1], s=markersize, c=Xscans[:,:,2], cmap='copper', norm=mpc.LogNorm())
    plt.scatter(Zscans[:,:,0], Zscans[:,:,1], s=markersize, c=Zscans[:,:,2], cmap='copper', norm=mpc.LogNorm())
    plt.plot([-pixelSize[0]/2,-pixelSize[0]/2,pixelSize[0]/2,pixelSize[0]/2,-pixelSize[0]/2],
             [-pixelSize[1]/2,pixelSize[1]/2,pixelSize[1]/2,-pixelSize[1]/2,-pixelSize[1]/2], 'c:')
    plt.colorbar()
    plt.xlim((-1.1,1.1))
    plt.ylim((-1.1,1.1))
    plt.title("X and Z scan positions, with data gaps, overlaid on rectangle of nominal pixel size")
    plt.xlabel("X position of scans")
    plt.ylabel("Z position of scans")
    plt.savefig(outdir+"scanpositions.png")
    plt.close()

    # plot distribution of values for each X scan
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.plot(Xscans[:,:,1], Xscans[:,:,2], '.')
    plt.title("X scans distribution of values per scan")
    plt.ylabel("Normalized pixel response")
    plt.xlabel("Z position of X scans")
    plt.savefig(outdir+"xscanzvals.png")
    plt.close()

    # plot distribution of values for each Z scan
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.plot(Zscans[:,:,0], Zscans[:,:,2], '.')
    plt.title("Z scans distribution of values per scan")
    plt.ylabel("Normalized pixel response")
    plt.xlabel("X position of Z scans")
    plt.savefig(outdir+"zscanxvals.png")
    plt.close()

    # plot X scans vs X position
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.plot(Xscans[:,:,0], Xscans[:,:,2], '.')
    plt.title("X scans pixel response with X")
    plt.ylabel("Normalized pixel response")
    plt.xlabel("X position in X scans")
    plt.savefig(outdir+"xscanxvals.png")
    plt.close()

    # plot Z scans vs Z position
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.plot(Zscans[:,:,1], Zscans[:,:,2], '.')
    plt.title("Z scans pixel response with Z")
    plt.ylabel("Normalized pixel response")
    plt.xlabel("Z position in Z scans")
    plt.savefig(outdir+"zscanzvals.png")
    plt.close()

  # Allocate arrays to hold scan values shifted by one pixel left/right
  Lefts = np.zeros(Xscans[:,:,0].shape)
  Rights = np.zeros(Xscans[:,:,0].shape)
  # for each scan
  for j in range(len(Xscans[0])):
    # for each value in each scan
    for i in range(len(Xscans)):
      # calculate position one nominal pixel width away
      Left  = Xscans[i,j,0] - pixelSize[0]
      Right = Xscans[i,j,0] + pixelSize[0]
      # if out of bounds, set to nan
      if Left < Xscans[:,j,0].min():
        Lefts[i,j] = np.nan
      # else, find the closest position value
      else:
        k = np.argmin((Left - Xscans[:,j,0])**2)
        Lefts[i,j] = Xscans[k,j,2]/Xscans[i,j,2]
      # if out of bounds, set to nan
      if Right > Xscans[:,j,0].max():
        Rights[i,j] = np.nan
      # else, find the closest position value
      else:
        k = np.argmin((Right - Xscans[:,j,0])**2)
        Rights[i,j] = Xscans[k,j,2]/Xscans[i,j,2]
    if Plots:
      plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
      # plot X scan metrics vs X positions
      plt.plot(Xscans[:,j,0],  Lefts[:,j], 'r', label="Value one pixel-width left divided by value at this pixel position")
      plt.plot(Xscans[:,j,0], Rights[:,j], 'g', label="Value one pixel-width right divided by value at this pixel position")
      # add vertical lines at nominal pixel boundaries
      plt.axvline(-pixelSize[0]/2, linestyle='dashed')
      plt.axvline( pixelSize[0]/2, linestyle='dashed')
      # add horizontal line at unity (metric should be 1 at pixel boundary)
      plt.axhline(1,               linestyle='dashed')
      # titles and labels
      plt.title("Metrics along X scan number %d" %j)
      plt.xlabel("X position in brightest pixel")
      plt.ylabel("PRF (theoretical) pixel comparison metric value")
      plt.legend(loc=9)
      # bound in x to slightly outside of pixel
      plt.xlim(-pixelSize[0]/2-.1, pixelSize[0]/2+.1)
      # bound in y to reasonable range
      plt.ylim(0,1.2)
      plt.savefig(outdir+"PRFmetricXscan%d.png" %j)
      plt.close()

  # Allocate arrays to hold scan values shifted by one pixel to up/down
  Ups   = np.zeros(Zscans[:,:,0].shape)
  Downs = np.zeros(Zscans[:,:,0].shape)
  # for each scan
  for j in range(len(Zscans[0])):
    # for each value in each scan
    for i in range(len(Zscans)):
      # calculate position one nominal pixel height away
      Up   = Zscans[i,j,1] - pixelSize[1]
      Down = Zscans[i,j,1] + pixelSize[1]
      # if out of bounds, set to nan
      if Up < Zscans[:,j,1].min():
        Ups[i,j] = np.nan
      # else, find the closest position value
      else:
        k = np.argmin((Up - Zscans[:,j,1])**2)
        Ups[i,j] = Zscans[k,j,2]/Zscans[i,j,2]
      # if out of bounds, set to nan
      if Down > Zscans[:,j,1].max():
        Downs[i,j] = np.nan
      # else, find the closest position value
      else:
        k = np.argmin((Down - Zscans[:,j,1])**2)
        Downs[i,j] = Zscans[k,j,2]/Zscans[i,j,2]

    if Plots:
      plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
      # plot Z scan metrics vs Z positions
      plt.plot(Zscans[:,j,1],   Ups[:,j], 'r', label="Value one pixel-height up divided by value at this pixel position")
      plt.plot(Zscans[:,j,1], Downs[:,j], 'g', label="Value one pixel-height down divided by value at this pixel position")
      # add vertical lines at nominal pixel boundaries
      plt.axvline(-pixelSize[1]/2, linestyle='dashed')
      plt.axvline( pixelSize[1]/2, linestyle='dashed')
      # add horizontal line at unity (metric should be 1 at pixel boundary)
      plt.axhline(1,               linestyle='dashed')
      # titles and labels
      plt.title("Metrics along Z scan number %d" %j)
      plt.xlabel("Z position in brightest pixel")
      plt.ylabel("PRF (theoretical) pixel comparison metric value")
      plt.legend(loc=9)
      # bound in x to slightly outside pixel
      plt.xlim(-pixelSize[1]/2-.1, pixelSize[1]/2+.1)
      # bound in y to reasonable range
      plt.ylim(0,1.2)
      plt.savefig(outdir+"PRFmetricZscan%d.png" %j)
      plt.close()

  # overview plots of metric values
  if Plots:
    # Lefts
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.scatter(Xscans[:,:,0], Xscans[:,:,1], s=markersize, c=Lefts, cmap='copper', norm=mpc.LogNorm(vmax=1.1))
    plt.plot([-pixelSize[0]/2,-pixelSize[0]/2,pixelSize[0]/2,pixelSize[0]/2,-pixelSize[0]/2],
             [-pixelSize[1]/2,pixelSize[1]/2,pixelSize[1]/2,-pixelSize[1]/2,-pixelSize[1]/2], 'c:')
    plt.colorbar()
    plt.xlim((-1.1,1.1))
    plt.ylim((-1.1,1.1))
    plt.title("X scan metric comparing to pixel on left")
    plt.xlabel("X position of scans")
    plt.ylabel("Z position of scans")
    plt.savefig(outdir+"metricoverviewleft.png")
    plt.close()

    # Rights
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.scatter(Xscans[:,:,0], Xscans[:,:,1], s=markersize, c=Rights, cmap='copper', norm=mpc.LogNorm(vmax=1.1))
    plt.plot([-pixelSize[0]/2,-pixelSize[0]/2,pixelSize[0]/2,pixelSize[0]/2,-pixelSize[0]/2],
             [-pixelSize[1]/2,pixelSize[1]/2,pixelSize[1]/2,-pixelSize[1]/2,-pixelSize[1]/2], 'c:')
    plt.colorbar()
    plt.xlim((-1.1,1.1))
    plt.ylim((-1.1,1.1))
    plt.title("X scan metric comparing to pixel on right")
    plt.xlabel("X position of scans")
    plt.ylabel("Z position of scans")
    plt.savefig(outdir+"metricoverviewright.png")
    plt.close()

    # Ups
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.scatter(Zscans[:,:,0], Zscans[:,:,1], s=markersize, c=Ups, cmap='copper', norm=mpc.LogNorm(vmax=1.1))
    plt.plot([-pixelSize[0]/2,-pixelSize[0]/2,pixelSize[0]/2,pixelSize[0]/2,-pixelSize[0]/2],
             [-pixelSize[1]/2,pixelSize[1]/2,pixelSize[1]/2,-pixelSize[1]/2,-pixelSize[1]/2], 'c:')
    plt.colorbar()
    plt.xlim((-1.1,1.1))
    plt.ylim((-1.1,1.1))
    plt.title("Z scan metric comparing to pixel above")
    plt.xlabel("X position of scans")
    plt.ylabel("Z position of scans")
    plt.savefig(outdir+"metricoverviewup.png")
    plt.close()

    # Downs
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.scatter(Zscans[:,:,0], Zscans[:,:,1], s=markersize, c=Downs, cmap='copper', norm=mpc.LogNorm( vmax=1.1))
    plt.plot([-pixelSize[0]/2,-pixelSize[0]/2,pixelSize[0]/2,pixelSize[0]/2,-pixelSize[0]/2],
             [-pixelSize[1]/2,pixelSize[1]/2,pixelSize[1]/2,-pixelSize[1]/2,-pixelSize[1]/2], 'c:')
    plt.colorbar()
    plt.xlim((-1.1,1.1))
    plt.ylim((-1.1,1.1))
    plt.title("Z scan metric comparing to pixel below")
    plt.xlabel("X position of scans")
    plt.ylabel("Z position of scans")
    plt.savefig(outdir+"metricoverviewdown.png")
    plt.close()

    # Now, overall X metric (Lefts when Xpos < 0, else Rights)
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.scatter(Xscans[np.where(XscanXs>0)][:,0], Xscans[np.where(XscanXs>0)][:,1], s=markersize, c= Lefts[np.where(XscanXs>0)], cmap='copper', norm=mpc.LogNorm(vmax=1.1))
    plt.scatter(Xscans[np.where(XscanXs<0)][:,0], Xscans[np.where(XscanXs<0)][:,1], s=markersize, c=Rights[np.where(XscanXs<0)], cmap='copper', norm=mpc.LogNorm(vmax=1.1))
    plt.plot([-pixelSize[0]/2,-pixelSize[0]/2,pixelSize[0]/2,pixelSize[0]/2,-pixelSize[0]/2],
             [-pixelSize[1]/2,pixelSize[1]/2,pixelSize[1]/2,-pixelSize[1]/2,-pixelSize[1]/2], 'c:')
    plt.colorbar()
    plt.xlim((-1.1,1.1))
    plt.ylim((-1.1,1.1))
    plt.title("X scan metric comparing to nearest adjacent pixel in X")
    plt.xlabel("X position of scans")
    plt.ylabel("Z position of scans")
    plt.savefig(outdir+"metricoverviewX.png")
    plt.close()

    # and, finally, overall Z metric (similar)
    plt.figure(num=None, figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    plt.scatter(Zscans[np.where(ZscanZs>0)][:,0], Zscans[np.where(ZscanZs>0)][:,1], s=markersize, c=  Ups[np.where(ZscanZs>0)], cmap='copper', norm=mpc.LogNorm( vmax=1.1))
    plt.scatter(Zscans[np.where(ZscanZs<0)][:,0], Zscans[np.where(ZscanZs<0)][:,1], s=markersize, c=Downs[np.where(ZscanZs<0)], cmap='copper', norm=mpc.LogNorm( vmax=1.1))
    plt.plot([-pixelSize[0]/2,-pixelSize[0]/2,pixelSize[0]/2,pixelSize[0]/2,-pixelSize[0]/2],
             [-pixelSize[1]/2,pixelSize[1]/2,pixelSize[1]/2,-pixelSize[1]/2,-pixelSize[1]/2], 'c:')
    plt.colorbar()
    plt.xlim((-1.1,1.1))
    plt.ylim((-1.1,1.1))
    plt.title("Z scan metric comparing to nearest adjacent pixel in Z")
    plt.xlabel("X position of scans")
    plt.ylabel("Z position of scans")
    plt.savefig(outdir+"metricoverviewZ.png")
    plt.close()

  # stack up the relevant arrays in the form we want
  Xscanmetrics = np.stack((XscanXs, XscanZs, XscanVal,  Lefts, Rights), axis=2)
  Zscanmetrics = np.stack((ZscanXs, ZscanZs, ZscanVal,    Ups,  Downs), axis=2)
  # and return them
  return Xscanmetrics, Zscanmetrics
  
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
