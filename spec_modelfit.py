import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
import glob
from astropy import units as u
from astropy import constants as const
import time

def rebinspec(wv,fx,nwwv,*args,**kwargs):
    #Required arguments:
    #   - wv: old wavelength array
    #   - fx: flux to be rebinned
    #   - nwwv: new wavelength array
    #
    #Optional arguments:
    #   - var = var, input and output variance
    #   - ivar = ivar, input and output ivar
    
    var = kwargs.get('var',None)
    ivar = kwargs.get('ivar',None)

    if (var is not None) & (ivar is None):
        nwfx,nwvar = spec_interp(wv,fx,nwwv,var)

        return nwfx, nwvar
    elif (var is None) & (ivar is not None):
        var = 1./ivar
        nwfx,nwvar_1 = spec_interp(wv,fx,nwwv,var)
        nwvar_1[nwvar_1 == 0.0] = -10.0
        nwivar = 1.0/nwvar_1
        nwivar[nwivar < 0.0] = 0.0

        return nwfx, nwivar
    else:
        nwfx = spec_interp(wv,fx,nwwv)

        return nwfx
        



def rescaleModel(model, distance, scaling, rjup = True, highres=False):
    """
    Loads in model spectrum from FITS table. Columns labels: Column 1 is wavelength in microns. 
    Column 2 is flux in watts/m^2/um at *surface!* of object, i.e., at the top of the atmosphere 
    so you have to scale for radius and distance.
    
    This is written to accommodate the BT Settl models in particular, downsampling them because 
    they are way too high spectral resolution. A smarter way would be to
    have a function that could do *multiple* different models...
    """
    data = fits.getdata(model)
    wv_all = data[0][0]
    flam_all = data[0][1]
    wv = wv_all[(wv_all > 0.5) & (wv_all < 5.0)]
    flam = flam_all[(wv_all > 0.5) & (wv_all < 5.0)]
    
    if highres == True:
        # take every 33rd data point, since BT Settl is such high-res.
        wv = wv[::33]
        flam = flam[::33]
    
    if rjup == True:
        # set radius to 1 RJup
        radius = 6.9911e9 * u.cm
    if rjup == False:
        # set radius to 1 solar radius
        radius = (1*u.Rsun).to(u.cm)
    
    # scale radius to some factor x 1 RJup
    radius *= scaling
    
    # convert distance
    dist_pc = distance * u.pc
    dist_cm = dist_pc.to(u.cm)
    
    # angular radius of bd as viewed from earth
    theta = np.arcsin(radius/dist_cm)
    
    # calcluate received flux density -- is this right?!
    f = np.pi*flam*(theta**2)
    
    return wv, f



# test of rebinning to the correct spectral resolution:

def spec_interp(wv,fx,nwwv,*args):
    # Required arguments:
    #   - wv: old wavelength array
    #   - fx: flux to be rebinned
    #   - nwwv: new wavelength array
    #
    # Optional argument: variance

    # Optional var argument
    npix = len(wv)
    if len(args) == 0:
        var = np.ones(npix)
        nwvarFlag = False
    else:
        var = args[0]
        nwvarFlag = True

    nwpix = len(nwwv)

    #Calculate wavelength endpoints for each pixel
    wvl = (wv + np.roll(wv,1))/2.
    wvh = (wv + np.roll(wv,-1))/2.
    wvl[0] = wv[0] - (wv[1] - wv[0])/2.
    wvh[npix-1] = wv[npix-1] + (wv[npix-1]-wv[npix-2])/2.

    #Calculate endpoints of the final array
    bwv = np.zeros(nwpix+1)
    bwv[0:nwpix] = (nwwv+np.roll(nwwv,1))/2.
    bwv[0] = nwwv[0] - (nwwv[1] - nwwv[0])/2.
    bwv[nwpix] = nwwv[nwpix-1]+(nwwv[nwpix-1] - nwwv[nwpix - 1])/2.

    #Create tmp arrays for final array
    nwfx = np.zeros(nwpix)
    nwvar = np.zeros(nwpix)
    nwunitfx = np.zeros(nwpix)

    #Loop through the arrays
    for q in range(npix):

        #No overlap
        if (wvh[q] <= bwv[0]) | (wvl[q] >= bwv[nwpix]):
            continue

        #Find pixel that bw is within
        if wvl[q] <= bwv[0]:
            i1 = [0]
        else:
            i1 = np.argwhere((wvl[q] <= np.roll(bwv,-1)) & (wvl[q] > bwv))[0]

        if wvh[q] > bwv[nwpix]:
            i2 = [nwpix-1]
        else:
            i2 = np.argwhere((wvh[q] <= np.roll(bwv,-1)) & (wvh[q] > bwv))[0]

        j1 = i1[0]
        j2 = i2[0]

        #Now sum up
        for kk in range(j1,j2+1):
            #Rejected pixels do not get added in
            if var[q] > 0.:
                frac = ( np.min([wvh[q],bwv[kk+1]]) - np.max([wvl[q],bwv[kk]]) ) / (wvh[q]-wvl[q])
                nwfx[kk] = nwfx[kk]+frac*fx[q]
                nwunitfx[kk] = nwunitfx[kk]+frac*1.0

                #Variance
                if nwvarFlag:
                    if (var[q] <= 0.) | (nwvar[kk] == -1):
                       nwvar[kk] = -1
                    else:
                       nwvar[kk] = nwvar[kk]+frac*var[q]

    if nwvarFlag:
        fxOut = nwfx/nwunitfx
        varOut = nwvar*nwunitfx
        
        return fxOut,varOut
    else:
        fxOut = nwfx/nwunitfx
        return fxOut
   


