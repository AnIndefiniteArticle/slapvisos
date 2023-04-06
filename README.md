Very Early Alpha

Code to analyze stellar occultations by Saturn taken by the Cassini
spacecraft's Visual and Infrared Mapping Spectrometer.

Contents:

imaginganalysis.py
 - does all the things

events directory
 - contains config files for various events
 - currently only some of them work with the current version of the code
 - need to copy (or symlink) the config file for the event you want to analyze
   to "config.py" in the same directory as imaginganalysis.py

Other notes:
 - need to have converted .QUB files from the PDS into .cub files locally using
   ISIS or similar.
 - code currently finds the center of the star to watch it refract behind
   Saturn's atmosphere, does photometry at certain wavelength bands, and
   generates spectra as the star disappears behind the planet.
 - Centering altorithm is currently center-of-light based, with plans to use
   Cassini PRF scans to make a more accurate centering algorithm
   - The star's PSF is smaller than a pixel, so centering algorithms need to be
     done carefully.
 - need to create directories for "figs" and the various movies before running

Current TODOs:
 - Implement PRF centering algorithm
 - Better documentation of entire code, including centering algorithm
 - Comparison of measured bending angle and opacity to theory
 - Better way to handle config files for multiple events
