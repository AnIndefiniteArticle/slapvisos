Beta!

Code to analyze stellar occultations by Saturn taken by the Cassini
spacecraft's Visual and Infrared Mapping Spectrometer.

Contents:

imaginganalysis.py
 - does all the things

events directory
 - contains config files for various events
 - currently events with visible data fail at flatfielding step (not good data anyway)
 - need to copy (or symlink) the config file for the event you want to analyze
   to "config.py" in the same directory as imaginganalysis.py

Other notes:
 - need to have converted .QUB files from the PDS into .cub files locally using
   ISIS or similar, and point to them in config file
 - code currently finds the center of the star to watch it refract behind
   Saturn's atmosphere, does photometry at certain wavelength bands, and
   generates spectra as the star disappears behind the planet.
 - Centering altorithm uses Cassini PRF scans in X. Z is coming soon.
   - The star's PSF is smaller than a pixel, so centering algorithms need to be
     done carefully.

Current TODOs:
 - Finish implementing Z-scan PRF centering algorithm
 - Comparison of measured bending angle and opacity to theory
 - Background subtraction with 1-D Saturn limb model
