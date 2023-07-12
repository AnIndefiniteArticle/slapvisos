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
    # Plot it all!
    if Plots:
      print("plotting X-scan %d"%j)
      # plot X scan metrics vs X positions
      k = np.argmin(abs(Xscans[:,j,0]))
      l = np.argmin(Xscans[:,j,0])
      m = np.argmax(Xscans[:,j,0])
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
      xaxs[int(j%nrowsx),j//nrowsx].plot(Xscans[:,j,0]+pixelSize[0], Xscans[:,j,2], 'r.', label="right comparison pixel flux", alpha=0.1)
      xaxs[int(j%nrowsx),j//nrowsx].plot(Xscans[:,j,0]-pixelSize[0], Xscans[:,j,2], 'g.', label= "left comparison pixel flux", alpha=0.1)
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

    if Plots:
      print("plotting Z-scan %d"%j)
      # plot Z scan metrics vs Z positions
      k = np.argmin(abs(Zscans[:,j,1]))
      l = np.argmin(Zscans[:,j,1])
      m = np.argmax(Zscans[:,j,1])
      # dots and labels
      #zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[np.min((k,m)):np.max((k,m)),j,1],   Ups[np.min((k,m)):np.max((k,m)),j], 'r.')
      #zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[np.min((l,k)):np.max((l,k)),j,1], Downs[np.min((l,k)):np.max((l,k)),j], 'g.')
      # lines connecting
      zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[np.min((k,m)):np.max((k,m)),j,1],   Ups[np.min((k,m)):np.max((k,m)),j], 'r', label="Value of pixel with same PRF below divided by value at this pixel position")
      zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[np.min((l,k)):np.max((l,k)),j,1], Downs[np.min((l,k)):np.max((l,k)),j], 'g', label="Value of pixel with same PRF above divided by value at this pixel position")
      # normalized flux of main pixel
      zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[:,j,1], Zscans[:,j,2], 'k.', label="Normalized flux of raw scan")
      # normalized flux of comparison pixels
      zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[:,j,1]+pixelSize[1], Zscans[:,j,2], 'r.', label="down comparison pixel flux", alpha=0.1)
      zaxs[int(j%nrowsz),j//nrowsz].plot(Zscans[:,j,1]-pixelSize[1], Zscans[:,j,2], 'g.', label=  "up comparison pixel flux", alpha=0.1)
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

  if Plots:
    print("saving metric scan plots")
    xfigs.savefig(outdir+"xscanmetrics.png")
    zfigs.savefig(outdir+"zscanmetrics.png")
    xfigs.clf()
    zfigs.clf()
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
  Xscanmetrics = np.stack((XscanXs, XscanZs, XscanVal,  Lefts, Rights), axis=2)
  Zscanmetrics = np.stack((ZscanXs, ZscanZs, ZscanVal,    Ups,  Downs), axis=2)
  # and return them
  return Xscanmetrics, Zscanmetrics

def transitionfinder(brightestPixel, window, Xpositive = True, Zpositive = True):
  # for each time step in a list of brightest pixels per timestep per wavelength
  for i in range(len(brightestPixel)):
    # calculate the mode within a window of +/- window indices from the current timestep, clipped by the ends of the array instead of wrapping
    mode = stats.mode(brightestPixel[np.max((i-window,0)):np.min((i+window,len(brightestPixel)))])
    # If it's the first entry, or if both the new mode is new AND represents over half of the array, add a new transition point
    if i == 0:
      transitions = np.array([[i, mode[0][0][0], mode[0][0][1]]])
    elif (mode[0][0][0] != transitions[-1][1] or mode[0][0][1] != transitions[-1][2]) and np.any(mode[1] > window):
      transitions = np.append(transitions, np.array([[i, mode[0][0][0], mode[0][0][1]]]), axis=0)

  return transitions

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
    #Xscansmetric = np.roll(XscanVal[:,11], (-1**(Left+1))*284) / XscanVal[:,11]
    
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
