#!/usr/bin/env python # Functions for analysis of Planetary Occultaions 
import numpy as np
import pysis as ps
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib.cm as cm
from scipy.io import readsav
from scipy import stats

def readVIMSimaging(cubdir, cubfiles, ncubs, nspec, height, width, visible):
  cubdata  = np.zeros((ncubs, nspec, height, width))
  cublabs  = [None]*ncubs
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
      cublabs[i] = ps.CubeFile(cubdir+cubfiles[i]).label
  return cubdata, cublabs

def getheaders(cubdir, cubfiles, ncubs):
  cublabs = [None]*ncubs
  for i in range(ncubs):
    if i%100 == 0:
      print(i, ncubs)
    cublabs[i] = ps.CubeFile(cubdir+cubfiles[i]).label
  return cublabs

def flatField(cubdata, flatfield, mode, outdir):
  shape       = cubdata.shape
  center      = (flatfield.shape[1]//2, flatfield.shape[2]//2)
  print(center," Flat Field Center")
  print(cubdata.min(),cubdata.max())
  croppedFlat = flatfield[:, center[0]:center[0]+shape[2], center[1]:center[1]+shape[3]]
  plt.imshow(croppedFlat.mean(axis=0), interpolation="none", cmap="viridis", origin='lower')
  plt.colorbar()
  plt.savefig(outdir+"/croppedflat.png")
  plt.clf()
  cubmean = cubdata.mean(axis=(0))[100:120].mean(axis=0)
  plt.imshow(cubmean, interpolation="none", cmap="viridis", vmin = cubmean.min(), vmax = cubmean.mean(), origin='lower')
  plt.title("Star Peak at %f"%cubmean.max())
  plt.colorbar()
  plt.savefig(outdir+"/summeddata.png")
  plt.clf()
  flatcor     = cubdata / croppedFlat.reshape(1,shape[1],shape[2],shape[3])
  flatcormean = flatcor.mean(axis=(0))[100:120].mean(axis=0)
  plt.imshow(flatcormean, interpolation="none", cmap="viridis", vmin = flatcormean.min(), vmax = flatcormean.mean(), origin='lower')
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

def prfmetric(PRFfile, pixelSize=(0.25,0.5), Plots=False, figsize=(10,10), dpi=300, fontsize=12, markersize=0.5, outdir='outputs/PRFscanplots/'):
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
  figsize : 2-tuple
      size of each subplot (width, height) in inches. A plot with four subplots
      will be 4x this area, as each subplot will be figsize.
  dpi : int
      dots per inch for plots
  Plots : boolean
      Create diagnostic plots?
  fontsize : int
      size of plot fonts in pt
  markersize : np.float64
      size of markers in scatterplot
  outdir : string
      Location of output directory for plots

  Returns
  -------
  Xscanmetrics : ndarray
      shape: (number of scans, number of points per scan, 5), with the 5 representing:
      [Xposition, Zposition, nominal value, metric measured to left, metric measured to right]
  Zscanmetrics : ndarray
      shape: (number of scans, number of points per scan, 6), with the 6 representing:
      [Zposition, Xposition, nominal value, metric measured up, metric measured down, metric measured both]]
  NOTE: For both of these, scan-parallel comes before scan-perpendicular! These two have rotated reference frames!
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
    nrows   = 2
    ncols   = 2
    fig,axs = plt.subplots(figsize=(ncols*figsize[0],nrows*figsize[1]), dpi=dpi, nrows=nrows, ncols=ncols)
    print("plotting the distribution of values for each X scan")
    axs[0,0].plot(Xscans[:,:,1], Xscans[:,:,2], '.')
    axs[0,0].set_title("X scans distribution of values per scan")
    axs[0,0].set_ylabel("Normalized pixel response")
    axs[0,0].set_xlabel("Z position of X scans")
    axs[0,0].set_yscale("log")

    print("plotting the distribution of values for each Z scan")
    axs[0,1].plot(Zscans[:,:,0], Zscans[:,:,2], '.')
    axs[0,1].set_title("Z scans distribution of values per scan")
    axs[0,1].set_ylabel("Normalized pixel response")
    axs[0,1].set_xlabel("X position of Z scans")
    axs[0,1].set_yscale("log")

    print("plotting X scans vs X position")
    axs[1,0].plot(Xscans[:,:,0], Xscans[:,:,2], '.')
    axs[1,0].set_title("X scans pixel response with X")
    axs[1,0].set_ylabel("Normalized pixel response")
    axs[1,0].set_xlabel("X position in X scans")
    axs[1,0].set_yscale("log")

    print("plotting Z scans vs Z position")
    axs[1,1].plot(Zscans[:,:,1], Zscans[:,:,2], '.')
    axs[1,1].set_title("Z scans pixel response with Z")
    axs[1,1].set_ylabel("Normalized pixel response")
    axs[1,1].set_xlabel("Z position in Z scans")
    axs[1,1].set_yscale("log")

    # saving
    fig.savefig(outdir+"scanlines.png")
    fig.clf()

    # allocating figures for scan-by-scan metric calibration
    nrowsx     = 7
    ncolsx     = 3
    nrowsz     = 4
    ncolsz     = 3
    xfigs,xaxs = plt.subplots(figsize=(ncolsx*figsize[0],nrowsx*figsize[1]), dpi=dpi, nrows=nrowsx, ncols=ncolsx)
    zfigs,zaxs = plt.subplots(figsize=(ncolsz*figsize[0],nrowsz*figsize[1]), dpi=dpi, nrows=nrowsz, ncols=ncolsz)
    zfig3,zax3 = plt.subplots(figsize=(ncolsz*figsize[0],nrowsz*figsize[1]), dpi=dpi, nrows=nrowsz, ncols=ncolsz)

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
        k = np.argmin(abs(Left - Xscans[:,j,0]))
        Lefts[i,j] = Xscans[k,j,2]/Xscans[i,j,2]
      # if out of bounds, set to nan
      if Right > Xscans[:,j,0].max():
        Rights[i,j] = np.nan
      # else, find the closest position value
      else:
        k = np.argmin(abs(Right - Xscans[:,j,0]))
        Rights[i,j] = Xscans[k,j,2]/Xscans[i,j,2]
    # For these scans, calibrate to correct up to 20 microradian spacecraft pointing errors
    # on the left side of the pixel, find 2 X positions closest to pixel boundary
    lftbnd = np.argmin(abs(Xscans[:,j,0] + pixelSize[0]/2))
    lftbnd = np.argmin(abs(Rights[lftbnd-10:lftbnd+10,j] - 1)) + (lftbnd-10)
    if Xscans[lftbnd,j,0] < -pixelSize[0]/2:
      lftbnd2 = np.argmax(Xscans[lftbnd-1:lftbnd+2,j,0]) + (lftbnd-1)
    else:
      lftbnd2 = np.argmin(Xscans[lftbnd-1:lftbnd+2,j,0]) + (lftbnd-1)
    # on a line between them, find where y=1
    lfty1 = Xscans[lftbnd,j,0] + (1-Rights[lftbnd,j])*(Xscans[lftbnd,j,0] - Xscans[lftbnd2,j,0])/ \
                                                      (Rights[lftbnd,j]   - Rights[lftbnd2,j])
    # repeat for right side
    rgtbnd = np.argmin(abs(Xscans[:,j,0] - pixelSize[0]/2))
    lftbnd = np.argmin(abs(Lefts[rgtbnd-10:rgtbnd+10,j] - 1)) + (rgtbnd-10)
    if Xscans[rgtbnd,j,0] < pixelSize[0]/2:
      rgtbnd2 = np.argmax(Xscans[rgtbnd-1:rgtbnd+2,j,0]) + (rgtbnd-1)
    else:
      rgtbnd2 = np.argmin(Xscans[rgtbnd-1:rgtbnd+2,j,0]) + (rgtbnd-1)
    # on a line between them, find where y=1
    rgty1 = Xscans[rgtbnd,j,0] + (1-Lefts[rgtbnd,j])*(Xscans[rgtbnd,j,0] - Xscans[rgtbnd2,j,0])/ \
                                                      (Lefts[rgtbnd,j]   -  Lefts[rgtbnd2,j])
    # split difference of left and right shifts, but plot both!
    lftshft = (-pixelSize[0]/2 - lfty1)
    rgtshft = ( pixelSize[0]/2 - rgty1)
    shift   = np.mean((lftshft,rgtshft))
    #print("scan %d, shift: %f, right: %f, left: %f" %(j,shift,lftshft,rgtshft))
    Xscans[:,j,0] += shift

    # set to nan if on wrong side of pixel
    k = np.argmin(abs(Xscans[:,j,0]))
    l = np.argmin(Xscans[:,j,0])
    m = np.argmax(Xscans[:,j,0])
    Lefts[ np.min((l,k+1)):np.max((l,k-1)),j]=np.nan
    Rights[np.min((k+1,m)):np.max((k-1,m)),j]=np.nan
    # Plot it all!
    if Plots:
      print("plotting X-scan %d"%j)
      # plot X scan metrics vs X positions
      # dots and labels
      #xaxs[int(j%nrowsx),j//nrowsx].plot(Xscans[np.min((k,m)):np.max((k,m)),j,0],  Lefts[np.min((k,m)):np.max((k,m)),j], 'r.') 
      #xaxs[int(j%nrowsx),j//nrowsx].plot(Xscans[np.min((l,k)):np.max((l,k)),j,0], Rights[np.min((l,k)):np.max((l,k)),j], 'g.')
      # lines connecting
      xaxs[int(j%nrowsx),j//nrowsx].plot(Xscans[np.min((k,m)):np.max((k,m)),j,0],  Lefts[np.min((k,m)):np.max((k,m)),j], 'r', label="Value of pixel with same PRF on the right divided by value at this pixel position") 
      xaxs[int(j%nrowsx),j//nrowsx].plot(Xscans[np.min((l,k)):np.max((l,k)),j,0], Rights[np.min((l,k)):np.max((l,k)),j], 'g', label="Value of pixel with same PRF on the  left divided by value at this pixel position")
      # shift markers
      xaxs[int(j%nrowsx),j//nrowsx].plot([-pixelSize[0]/2 - shift, pixelSize[0]/2 - shift], [1,1], 'mX', label="location of y=1 crossing before centering fix of %f mrad"%shift)
      # normalized flux of main pixel
      xaxs[int(j%nrowsx),j//nrowsx].plot(Xscans[:,j,0], Xscans[:,j,2], 'k.', label="Normalized flux of raw scan")
      # normalized flux of comparison pixels
      xaxs[int(j%nrowsx),j//nrowsx].plot(Xscans[:,j,0]+pixelSize[0], Xscans[:,j,2], 'm.', label="right comparison pixel flux", alpha=0.1)
      xaxs[int(j%nrowsx),j//nrowsx].plot(Xscans[:,j,0]-pixelSize[0], Xscans[:,j,2], 'y.', label= "left comparison pixel flux", alpha=0.1)
      # add vertical lines at nominal pixel boundaries
      xaxs[int(j%nrowsx),j//nrowsx].axvline(-pixelSize[0]/2, linestyle='dashed')
      xaxs[int(j%nrowsx),j//nrowsx].axvline( pixelSize[0]/2, linestyle='dashed')
      # add horizontal line at unity (metric should be 1 at pixel boundary)
      xaxs[int(j%nrowsx),j//nrowsx].axhline(1,               linestyle='dashed')
      # and another at 0.5 (each pixel should be at 0.5 at boundary if no flux lost)
      xaxs[int(j%nrowsx),j//nrowsx].axhline(0.5,             linestyle='dashed')
      # titles and labels
      xaxs[int(j%nrowsx),j//nrowsx].set_title("X scan %d" %j)
      xaxs[int(j%nrowsx),j//nrowsx].set_xlabel("X position from centered in brightest pixel (mrad)")
      xaxs[int(j%nrowsx),j//nrowsx].set_ylabel("Black: Flux, Ratio: (Flux one pixel width away, further from 0)/(Flux here)")
      xaxs[int(j%nrowsx),j//nrowsx].legend(loc=9)
      # bound in x to slightly outside of pixel
      #xaxs[int(j%nrowsx),j//nrowsx].set_xlim(-pixelSize[0]/2-.1, pixelSize[0]/2+.1)
      # bound in y to reasonable range
      xaxs[int(j%nrowsx),j//nrowsx].set_yscale("log")

  # Allocate arrays to hold scan values shifted by one pixel to up/down
  Ups   = np.zeros(Zscans[:,:,0].shape)
  Downs = np.zeros(Zscans[:,:,0].shape)
  Boths = np.zeros(Zscans[:,:,0].shape)
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
        k = np.argmin(abs(Up - Zscans[:,j,1]))
        Ups[i,j] = Zscans[k,j,2]/Zscans[i,j,2]
      # if out of bounds, set to nan
      if Down > Zscans[:,j,1].max():
        Downs[i,j] = np.nan
      # else, find the closest position value
      else:
        k = np.argmin(abs(Down - Zscans[:,j,1]))
        Downs[i,j] = Zscans[k,j,2]/Zscans[i,j,2]

    # Calculate 3-pixel metric
    Boths[:,j] = Ups[:,j] - Downs[:,j]

    # set to nan if on wrong side of pixel
    #k = np.argmin(abs(Xscans[:,j,0]))
    #l = np.argmin(Xscans[:,j,0])
    #m = np.argmax(Xscans[:,j,0])
    #Ups[  np.min((l,k)):np.max((l,k)),j]=np.nan
    #Downs[np.min((k,m)):np.max((k,m)),j]=np.nan

    if Plots:
      print("plotting Z-scan %d"%j)
      # plot Z scan metrics vs Z positions
      # dots and labels
      #zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[np.min((k,m)):np.max((k,m)),j,1],   Ups[np.min((k,m)):np.max((k,m)),j], 'r.')
      #zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[np.min((l,k)):np.max((l,k)),j,1], Downs[np.min((l,k)):np.max((l,k)),j], 'g.')
      # lines connecting
      zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[np.min((k,m)):np.max((k,m)),j,1],   Ups[np.min((k,m)):np.max((k,m)),j], 'r', label="Value of pixel with same PRF below divided by value at this pixel position")
      zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[np.min((l,k)):np.max((l,k)),j,1], Downs[np.min((l,k)):np.max((l,k)),j], 'g', label="Value of pixel with same PRF above divided by value at this pixel position")
      # normalized flux of main pixel
      zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[:,j,1], Zscans[:,j,2], 'k.', label="Normalized flux of raw scan")
      # normalized flux of comparison pixels
      zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[:,j,1]+pixelSize[1], Zscans[:,j,2], 'm.', label="down comparison pixel flux", alpha=0.1)
      zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[:,j,1]-pixelSize[1], Zscans[:,j,2], 'y.', label=  "up comparison pixel flux", alpha=0.1)
      # add vertical lines at nominal pixel boundaries
      zaxs[int(j%nrowsz),j//nrowsz].axvline(-pixelSize[1]/2, linestyle='dashed')
      zaxs[int(j%nrowsz),j//nrowsz].axvline( pixelSize[1]/2, linestyle='dashed')
      # add horizontal line at unity (metric should be 1 at pixel boundary)
      zaxs[int(j%nrowsz),j//nrowsz].axhline(1,               linestyle='dashed')
      # and another at 0.5 (each pixel should be at 0.5 at boundary if no flux lost)
      zaxs[int(j%nrowsz),j//nrowsz].axhline(0.5,             linestyle='dashed')
      # titles and labels
      zaxs[int(j%nrowsz),j//nrowsz].set_title("Z scan %d" %j)
      zaxs[int(j%nrowsz),j//nrowsz].set_xlabel("Z position from centered in brightest pixel (mrad)")
      zaxs[int(j%nrowsz),j//nrowsz].set_ylabel("Black: Flux, Ratio: (Flux one pixel width away, further from 0)/(Flux here)")
      zaxs[int(j%nrowsz),j//nrowsz].legend(loc=9)
      # bound in x to slightly outside pixel
      #zaxs[int(j%nrowsz),j//nrowsz].set_xlim(-pixelSize[1]/2-.1, pixelSize[1]/2+.1)
      # bound in y to reasonable range
      zaxs[int(j%nrowsz),j//nrowsz].set_yscale("log")

      # Repeat plot for 3-pixel metric
      zax3[int(j%nrowsz),j//nrowsz].plot(Zscans[:,j,1], abs(Boths[:,j]), 'xkcd:electric green', label='absolute value of 3-pixel metric')
      # normalized flux of main pixel
      zax3[int(j%nrowsz),j//nrowsz].plot(Zscans[:,j,1], Zscans[:,j,2], 'k.', label="Normalized flux of raw scan")
      # normalized flux of comparison pixels
      zax3[int(j%nrowsz),j//nrowsz].plot(Zscans[:,j,1]+pixelSize[1], Zscans[:,j,2], 'm.', label="down comparison pixel flux", alpha=0.1)
      zax3[int(j%nrowsz),j//nrowsz].plot(Zscans[:,j,1]-pixelSize[1], Zscans[:,j,2], 'y.', label=  "up comparison pixel flux", alpha=0.1)
      # add vertical lines at nominal pixel boundaries
      zax3[int(j%nrowsz),j//nrowsz].axvline(-pixelSize[1]/2, linestyle='dashed')
      zax3[int(j%nrowsz),j//nrowsz].axvline( pixelSize[1]/2, linestyle='dashed')
      # add horizontal line at unity (metric should be 1 at pixel boundary)
      zax3[int(j%nrowsz),j//nrowsz].axhline(1,               linestyle='dashed')
      # and another at 0.5 (each pixel should be at 0.5 at boundary if no flux lost)
      zax3[int(j%nrowsz),j//nrowsz].axhline(0.5,             linestyle='dashed')
      # titles and labels
      zax3[int(j%nrowsz),j//nrowsz].set_title("Z scan %d" %j)
      zax3[int(j%nrowsz),j//nrowsz].set_xlabel("Z position from centered in brightest pixel (mrad)")
      zax3[int(j%nrowsz),j//nrowsz].set_ylabel("normalized flux, or flux ratio difference")
      zax3[int(j%nrowsz),j//nrowsz].legend(loc=9)
      # log scale for y
      zax3[int(j%nrowsz),j//nrowsz].set_yscale("log")


  if Plots:
    print("saving metric scan plots")
    xfigs.savefig(outdir+"xscanmetrics.png")
    zfigs.savefig(outdir+"zscanmetrics.png")
    zfig3.savefig(outdir+"zscanmetric3.png")
    xfigs.clf()
    zfigs.clf()
    zfig3.clf()
    print("creating pixel overview plots")
    nrows   = 1
    ncols   = 3
    fig,axs = plt.subplots(figsize=(ncols*figsize[0] + 10,nrows*figsize[1]), dpi=dpi, nrows=nrows, ncols=ncols)

    # pixel response heatmaps on the scanline positions
    im=axs[0].scatter(Xscans[:,:,0], Xscans[:,:,1], s=markersize, c=Xscans[:,:,2], cmap='viridis', norm=mpc.LogNorm())
    axs[0].scatter(Zscans[:,:,0], Zscans[:,:,1], s=markersize, c=Zscans[:,:,2], cmap='viridis', norm=mpc.LogNorm())
    axs[0].plot([-pixelSize[0]/2,-pixelSize[0]/2,pixelSize[0]/2,pixelSize[0]/2,-pixelSize[0]/2],
             [-pixelSize[1]/2,pixelSize[1]/2,pixelSize[1]/2,-pixelSize[1]/2,-pixelSize[1]/2], 'c:')
    axs[0].set_xlim((-1.1,1.1))
    axs[0].set_ylim((-1.1,1.1))
    axs[0].set_title("X and Z scan positions, with data gaps, overlaid on rectangle of nominal pixel size")
    axs[0].set_xlabel("X position of scans")
    axs[0].set_ylabel("Z position of scans")
    axs[0].set_facecolor("k")#"xkcd:olive green")
    fig.colorbar(im, ax=axs[0])

    # Now, overall X metric (Lefts when Xpos < 0, else Rights)
    im=axs[1].scatter(Xscans[np.where(XscanXs>0)][:,0], Xscans[np.where(XscanXs>0)][:,1], s=markersize, c= Lefts[np.where(XscanXs>0)], cmap='viridis', norm=mpc.LogNorm(vmax=1.1))
    axs[1].scatter(Xscans[np.where(XscanXs<0)][:,0], Xscans[np.where(XscanXs<0)][:,1], s=markersize, c=Rights[np.where(XscanXs<0)], cmap='viridis', norm=mpc.LogNorm(vmax=1.1))
    axs[1].plot([-pixelSize[0]/2,-pixelSize[0]/2,pixelSize[0]/2,pixelSize[0]/2,-pixelSize[0]/2],
             [-pixelSize[1]/2,pixelSize[1]/2,pixelSize[1]/2,-pixelSize[1]/2,-pixelSize[1]/2], 'c:')
    axs[1].set_xlim((-1.1,1.1))
    axs[1].set_ylim((-1.1,1.1))
    axs[1].set_title("X scan metric comparing to nearest adjacent pixel in X")
    axs[1].set_xlabel("X position of scans")
    axs[1].set_ylabel("Z position of scans")
    axs[1].set_facecolor("k")#"xkcd:olive green")
    fig.colorbar(im, ax=axs[1])

    # and, finally, overall Z metric (similar)
    im=axs[2].scatter(Zscans[np.where(ZscanZs>0)][:,0], Zscans[np.where(ZscanZs>0)][:,1], s=markersize, c=  Ups[np.where(ZscanZs>0)], cmap='viridis', norm=mpc.LogNorm( vmax=1.1))
    axs[2].scatter(Zscans[np.where(ZscanZs<0)][:,0], Zscans[np.where(ZscanZs<0)][:,1], s=markersize, c=Downs[np.where(ZscanZs<0)], cmap='viridis', norm=mpc.LogNorm( vmax=1.1))
    axs[2].plot([-pixelSize[0]/2,-pixelSize[0]/2,pixelSize[0]/2,pixelSize[0]/2,-pixelSize[0]/2],
             [-pixelSize[1]/2,pixelSize[1]/2,pixelSize[1]/2,-pixelSize[1]/2,-pixelSize[1]/2], 'c:')
    axs[2].set_xlim((-1.1,1.1))
    axs[2].set_ylim((-1.1,1.1))
    axs[2].set_title("Z scan metric comparing to nearest adjacent pixel in Z")
    axs[2].set_xlabel("X position of scans")
    axs[2].set_ylabel("Z position of scans")
    axs[2].set_facecolor("k")#"xkcd:olive green")
    fig.colorbar(im, ax=axs[2])

    fig.savefig(outdir+'heatmaps.png')

  # stack up the relevant arrays in the form we want
  # NOTE THAT PARALLEL BEFORE PERPENDICULAR, NOT X BEFORE Z
  Xscanmetrics = np.stack((XscanXs, XscanZs, XscanVal,  Lefts, Rights), axis=2)
  Zscanmetrics = np.stack((ZscanZs, ZscanXs, ZscanVal,    Ups,  Downs, Boths), axis=2)
  # and return them
  return Xscanmetrics, Zscanmetrics

def argmax_lastNaxes(A, N):
    """
    Stolen from stackoverflow, with slight modification, here:
    https://stackoverflow.com/questions/30589211/numpy-argmax-over-multiple-axes-without-loop
    uses argmax to find the maximum location in each frame
    
    Parameters
    ----------
    A : ndarray
        array to find the maximum indices of
    N : int
        number of axes at the end of the array's shape to argmax over
    Returns
    ----------
    max_idx : ndarray
        N-dimensional array (where N is input parameter) of indices in A's last N dimensions where A is maximized
        e.g. if A's shape is (framenumber, columns, rows), and N=2, max_idx will be an array of the rows and columns where each frame is maximum
    """
    s = A.shape
    new_shp = s[:-N] + (np.prod(s[-N:]),)
    max_idx = A.reshape(new_shp).argmax(-1)
    return np.stack(np.unravel_index(max_idx, s[-N:]))

def rolling_average(frames, window, axis=0, mode='same'):
  """
  Simple rolling average implementation along a single axis

  Parameters
  ----------
  frames : ndarray
      array to take rolling average along a dimension of
  window : int
      number of frames to roll together (size of boxcar)
  axis   : int
      axis to take rolling average along
  mode   : str
      modes of np.convolve. 'same' returns data with the same dimensions.
      'full' and 'valid' are the other options. See np.convolve documentation
      for more information

  Returns
  ----------
  smoothedframes : ndarray
      same dimensions as frames, but boxcar smoothed along the provided axis
      using a boxcar of width window
  """
  #return np.lib.stride_tricks.sliding_window_view(frames, window, axis=axis).mean(axis=-1)
  return np.apply_along_axis(np.convolve, axis, frames, v=np.ones(window), mode=mode)/window

def rolling_std(timeseries, window, axis=0):
  std = np.zeros(timeseries.shape)
  for i in range(timeseries.shape[axis]):
    std[i] = np.std(timeseries.take(indices=range(np.max((0,i-window)),np.min((timeseries.shape[axis]-1,i+window))), axis=axis), axis=axis)
  return std
  

def transitionfinder(brightestPixel, window):
  """
  Finds when the mode brightest pixel changes and marks those frames as transition points
  
  Parameters
  ----------
  brightestPixel : array
      array of brightest pixel values
  window         : int
      window around each frame over which to take the mode

  Returns
  ----------
  transitions    : ndarray
      2d array with length of number of transitions, and three columns:
      frame number of transition, new brightest pixel, number of frames in window where that's the brightest pixel
  """
  # for each time step in a list of brightest pixels per timestep per wavelength
  for i in range(len(brightestPixel)):
    # calculate the mode within a window of +/- window indices from the current timestep, clipped by the ends of the array instead of wrapping
    mode = stats.mode(brightestPixel[np.max((i-window,0)):np.min((i+window,len(brightestPixel)))], keepdims=False)
    # If it's the first entry, or if both the new mode is new AND represents over half of the array, add a new transition point
    if i == 0:
      transitions = np.array([[i, mode[0], mode[1]]])
    # only add a new transition point if the new mode is adjacent to the previous one, and fills more than half of the array.
    elif (mode[0] == transitions[-1][1]+1 or mode[0] == transitions[-1][1]-1) and mode[1] > window:
      transitions = np.append(transitions, np.array([[i, mode[0], mode[1]]]), axis=0)

  return transitions

def bintoZscan(nframes, transitions, metrics):
  """
  bins rough star positions in one dimension to the scanline numbers in the perpendicular direction
  TODO: currently only bins to Z-scanlines from rough X position, merge with bintoXscan
  TODO: make it fit to closest Z-scanline, NOT just by which third of the pixel it falls in

  I'm hacking this loop to also calculate comparison pixels for X centering because it makes sense
  TODO: separate this out
  """
  # initialize scan array
  scan = np.zeros(nframes, dtype=int)
  # before first transition, always use middle scan?
  scan[:transitions[1][0]] = 5

  # initialize comparison array
  compare = np.ones(nframes, dtype=int)

  # start after the first transition
  step = 0
  # loop through frames after the first transition
  for i in range(nframes):
    # increment step if you pass a transition
    if i > transitions[step+1][0]:
      step += 1
    # if you run out of transitions
    if step == len(transitions)-1:
      # use scan 4 till the end
      scan[i:] = 4
      # and compare to the left
      compare[i:] = -1
      # and leave the loop
      break
    # frame at left, right, and center of pixel
    left   = transitions[step][0]
    right  = transitions[step+1][0]
    center = (left + right)/2
    # if closer to the left than the center, use 4
    if abs(i-left) < abs(i-center):
      scan[i] = 4
    # if closer to the center than the right, use 5
    elif abs(i-center) < abs(i-right):
      scan[i] = 5
    # use 6
    else:
      scan[i] = 6
    # if to the left of center, compare to the left
    # hack, or if you're on step 0 and less than half of the second step width's distance from the first transition
    if (i < center) or (step == 0 and i < (3*transitions[1][0] - transitions[2][0])/2 -10):
      compare[i] = -1
  return scan, compare

def comparison(frames, Xbrights, Zbrights):
  """
  This is an idea for an outline for this function that likely tries to do too much?
  Or maybe it's just what I need to figure out the final centering issues.
  """
  # allocate compares and metric
  compare = np.zeros((len(frames),2))
  imagemetric = np.zeros(len(frames))
  # loop through frames
  for i in range(len(frames)):
    # get pixel values on either side of brightest
    center  = frames[i, Zbrights[i], Xbrights[i]]
    left    = frames[i, Zbrights[i], Xbrights[i]-1]
    try:
      right = frames[i, Zbrights[i], Xbrights[i]+1]
    except:
      right = None
    # if either are brighter than main pixel
    #if left > center:
    #  # make that the brightest pixel 
    #  Xbrights[i] -= 1
    #  # i -= 1 and restart loop
    #  i           -= 1
    #  continue
    #if right and (right > center):
    #  # make that the brightest pixel 
    #  Xbrights[i] += 1
    #  # i -= 1; continue
    #  i           -= 1
    #  continue
    # set second-brightest pixel as comparison pixel
    if right and left < right:
      compare[i,0] =  1
      compare[i,1] = right
    else:
      compare[i,0] = -1
      compare[i,1] = left
    # calculate image metric
    imagemetric[i] = compare[i,1]/center
    # calculate other interesting frame diagnostics: mean, mode, background, total flux, etc
  # return the comparison pixels, image metric, etc
  return compare, imagemetric

def bintoXscan(subpixel, metrics):
  """
  selects closest X scan based on Z subpixel positions

  Parameters
  ----------
  subpixel : 1darray
      array of subpixel Z positions for each timestep
  metrics  : ndarray
      the X metrics from the prfmetric function

  Returns
  ----------
  scans    : 1darray
      array of nearest X scan for each timestep
  """
  # find average perpendicular position of each scan
  scanpos = metrics[:,:,1].mean(axis=0)

  # calculate distance of each subpixel position to each scan
  distance = abs(subpixel.reshape((len(subpixel),1)) - scanpos.reshape((1,len(scanpos))))

  # store which scan has the minimum distance
  scans = np.argmin(distance, axis=1)

  # return this array of scan numbers
  return scans

def findthestar(cubdata, specwin, Xmetrics, Zmetrics, window=10, pixelSize=(0.25,0.5), brightwindow=10, metriccutoff=0.01, sigclip=3):
  """
  Finds the location of the "star" (brightest pixel) in each frame of cubdata,
  spectrally monochromized, stretched by squaring, with a rolling average in
  frame number of size `smoothwin`

  Parameters
  ----------
  cubdata   : ndarray
      ndarray of shape (framenumber, spectral index, columns, rows)
  specwin   : tuple
      spectral window to sum over (min, max), in indices space, not wavelength space
      TODO: enable specifying in wavelength space
  Xmetrics  : ndarray
      output of prfmetric()
  Zmetrics  : ndarray
      output of prfmetric()
  window    : int
      number of frames to perform transition mode calculation over
  pixelSize : 2-tuple
      size of pixel in miliradians, default is VIMS HiRes mode
  brightwindow    : float
      brightest pixel window for removing dim frames with low snr
  metriccutoff    : float
      image metric cutoff to remove low SNR center fits
  sigclip : float
    number of sigma outliers in brightwindow to clip

  returns
  -----------
  right now, it returns a lot of intermediate steps for debug purposes
  """
  # remove last eight spectral channels (full of -8192), and average over the rest
  print("smoothing input data over spectral window")
  mono = cubdata[:,specwin[0]:specwin[1]].mean(axis=1)

  # subtract the minimum out (make everything positive) DEBUG ONLY
  #print("subtracting minimum to make everything positive")
  #smoothmono -= smoothmono.min()

  # take rolling average
  print("taking the rolling average")
  smoothmono = rolling_average(mono, window)

  # find brightest pixels in each frame
  print("finding the brightest pixels in each frame")
  maxcoords = argmax_lastNaxes(smoothmono, 2)

  # find transition frames in X
  print("finding the transition frames in X")
  Xtransitions = transitionfinder(maxcoords[1], window)

  # define brights from transitions
  Xbrights = np.zeros(len(mono), dtype=int)
  Xbrights[0] = Xtransitions[0,1]
  for i in range(len(Xtransitions)):
    try:
      Xbrights[Xtransitions[i,0]+1:Xtransitions[i+1,0]+1] = Xtransitions[i,1]
    except:
      Xbrights[Xtransitions[i,0]:] = Xtransitions[i,1]
  Zbrights = np.array([stats.mode(maxcoords[0], keepdims=True)[0][0]] * len(mono), dtype=int)

  # bin X-position along line to closest Z-scan
  # TODO remove Xcompares from being calculated here, once they are well-calculated elsewhere
  print("finding which Zscan to use for each frame")
  Zscans, Xcompares = bintoZscan(len(mono), Xtransitions, Zmetrics)

  # subtract from frames average value outside of 3 columns straddling brightest
  # Temporal background subtraction!
  # TODO Turn this into its own function to be improved upon
  corrmono = np.copy(mono)
  for i in range(len(mono)):
    background = mono[i].copy()
    background[:,Xbrights[i]] = np.nan
    background[:,Xbrights[i]-1] = np.nan
    try:
      background[:,Xbrights[i]+1] = np.nan
    except:
      pass
    corrmono[i] -= np.nanmean(background[:,1:]) # exclude 1st column

  # Priority TODO I can still see a clear spatial background that must also be corrected out
  # I will need to make a spatial background map in two parts, left side late, right side early
  # need average value out-of-transit for spatial correction
  # need standard deviation out-of-transit for noise-floor calculation for centering method
  # need to compare out-of-transit statistics before and after, for the pixels where this is possible, to look for signal from planet

  # generate comparisons and image metrics
  # TODO needs a revamped implementation
  # TODO make this calculate Z image metrics, too
  # TODO make later functions use these
  compares, imagemetric = comparison(mono, Xbrights, Zbrights)

  # Columns to do Z-centering on
  # TODO pass this the imagemetrics directly
  print("Z centering")
  columns = np.take_along_axis(smoothmono,np.array([[Xbrights]*corrmono.shape[1],]).transpose(),2)
  # 3-pixel center correction in Z
  Zcorr  = threepix(columns, Zbrights, Zscans, Zmetrics) #+ pixelSize[1]/2

  # bin Z corrections to closest X-scan
  print("finding which Xscans to use")
  #Xscans = 9*np.ones(len(Zscans), dtype=int)
  Xscans = bintoXscan(Zcorr, Xmetrics)

  # rows to do X-centering on
  print("X centering")
  rows = np.take_along_axis(corrmono,np.array([[Zbrights]]).transpose(),1)
  # 2-pixel center correction in X
  # TODO: find way to not hardcode [200:370] window limiting
  # TODO pass this the imagemetrics directly
  Xcorr, scanmetrics, imagemetrics, comparisons = twopix(rows, Xbrights, Xbrights + Xcompares, Xscans, Xmetrics[200:370], brightwindow, metriccutoff, sigclip) #[200:370] limits results to being on the pixel

  # combine centers with corrections
  Xcorr += pixelSize[0]/2
  Xcorr /= pixelSize[0]
  Zcorr += pixelSize[1]/2
  Zcorr /= pixelSize[1]

  # Do center-informed photometry!
  # TODO Priority
  # model peak brightness of PSF from absolute value of brightest considering
  # centering in both directions, and whether centering is trustworthy (is
  # compare above noise floor calculated by taking standard deviation during
  # the spatial background correction)

  # currently returns everything useful for bug testing
  return corrmono, maxcoords, Xbrights, Xcompares, Zbrights, Xtransitions, Zscans, Zcorr, Xscans, Xcorr, scanmetrics, imagemetrics, comparisons, columns, compares, imagemetric

def threepix(columns, brights, scans, metrics):
  """
  This function performs 3-pixel centering, or falls back to 2-pixel centering on edge-of-frame
  Parameters
  ----------
  columns : 2D array
    brightest column (could easily be row) that we will perform 3-pixel centering upon
    first dimension is time, second is space
  brights : 1D array
    index of brightest pixel in each column at each time step
  scans   : 1D array
    index of scan to use at each timestep
  metrics : 2D array
    one of the metric arrays output from the prfmetric function
    [:,:,0] is positions, [:,:,5] is metric

  Returns
  ----------
  corrections : 1D array
    subpixel corrections for each timestep
    values between 0 and 1, distance from center of pixel, because that's how the scans work
  """
  # select pixel values
  brightpix = np.take_along_axis(columns[:,:,0],( np.array([brights,])                       ).transpose(),1)
  leftpix   = np.take_along_axis(columns[:,:,0],((np.array([brights,]) - 1)% columns.shape[1]).transpose(),1)
  rightpix  = np.take_along_axis(columns[:,:,0],((np.array([brights,]) + 1)% columns.shape[1]).transpose(),1)
  # calculate imagemetric
  imagemetric = ( rightpix - leftpix ) / brightpix

  # select which scanmetric to use
  scanmetrics = metrics[:,scans,5].transpose()
  # compare imagemetric with scanmetric
  comparisons = (imagemetric - scanmetrics)**2

  # set correction to position with lowest chi-squared metric comparison
  corrections = metrics[np.nanargmin(comparisons, axis=1),scans,0]

  # TODO make this work
  # redo with twopix when at boundary
  #k = columns.shape[1]-1
  #zerobound = np.where(brights==0)
  #highbound = np.where(brights==k)
  #if len(zerobound) != 0:
  #  corrections[zerobound] = twopix(columns[zerobound], brights[zerobound], 1+brights[zerobound], scans, metrics)
  #if len(highbound) != 0:
  #  corrections[highbound] = twopix(columns[highbound], brights[highbound],-1+brights[highbound], scans, metrics)

  # return corrections
  return corrections

def twopix(rows, brights, compares, scans, metrics, brightwindow=10, metriccutoff=0.01, sigclip=3):
  """
  This function performs 2-pixel centering
  Parameters
  ----------
  rows    : 2D array
    brightest row (could easily be row) that we will perform 3-pixel centering upon
    first dimension is time, second is space
  brights : 1D array
    index of brightest pixel in each row at each time step
  compares : 1D array
    index of comparison pixel in each row at each timestep
  scans   : 1D array
    index of scan to use at each timestep
  metrics : 2D array
    one of the metric arrays output from the prfmetric function
    [:,:,0] is positions, [:,:,3] is negative, [:,:,4] is positive
  brightwindow  : float
    brightest pixel window for removing dim frames with low snr
  metriccutoff  : float
    image metric cutoff (too low for good snr)
  sigclip : float
    number of sigma outliers in brightwindow to clip

  Returns
  ----------
  corrections : 1D array
    subpixel corrections for each timestep
    integers are on pixel boundaries
    values between 0 and 1, distance between integer values
  """
  scans = np.array([9]*len(scans))
  # calculate imagemetric
  bripix = np.take_along_axis(rows[:,0,:],np.array([brights, ]).transpose(), 1)
  compix = np.take_along_axis(rows[:,0,:],np.array([compares,]).transpose(), 1)
  imagemetrics = compix/bripix
  #imagemetrics = abs(compix/bripix)**0.9

  # determine which side's metric to use for each timestep
  sides        = np.zeros(len(rows), dtype=int)
  sides[np.where(compares<brights)] = 4
  sides[np.where(compares>brights)] = 3
  # select relevant metrics per timestep
  scanmetrics  = metrics[:,scans,sides].transpose()
  scanpos      = metrics[:,scans,    0].transpose()

  # chi-squared comparison of imagemetric with relevant scanmetric
  comparisons  = (imagemetrics - scanmetrics)**2

  # set correction to positions with lowest chi-squared metric comparison
  corrections  = np.take_along_axis(scanpos, np.nanargmin(comparisons, axis=1).reshape((len(comparisons), 1)), 1)[:,0]

  # set to nan any subpixel corrections that don't have enough signal in the comparison pixel
  corrections[np.where(imagemetrics[:,0] < metriccutoff)] = np.nan
  #corrections[np.where(bripix[:,0] < brightwindow)] = np.nan
  corrections[np.where(abs(bripix[:,0] - rolling_average(bripix[:,0], brightwindow)) > sigclip*rolling_std(bripix, brightwindow))] = np.nan
  corrections[np.where(abs(compix[:,0] - rolling_average(compix[:,0], brightwindow)) > sigclip*rolling_std(compix, brightwindow))] = np.nan
  #corrections[np.where(abs(bripix[:,0] - rolling_average(bripix[:,0], brightwindow)) > sigclip*rolling_std(compix, brightwindow))] = np.nan

  # return corrections
  return corrections, scanmetrics, imagemetrics, comparisons
