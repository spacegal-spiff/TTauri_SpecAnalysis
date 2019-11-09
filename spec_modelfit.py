

from __future__ import print_function, division, absolute_import


import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
import glob
from astropy import units as u
from astropy import constants as const
import time

def rebinspec(wv,fx,nwwv,*args,**kwargs):
    '''
    #Required arguments:
    #   - wv: old wavelength array
    #   - fx: flux to be rebinned
    #   - nwwv: new wavelength array
    #
    #Optional arguments:
    #   - var = var, input and output variance
    #   - ivar = ivar, input and output ivar
    
    todo
    ----------
    1. needs block for wavelengths out of the interpolation range
    
    '''
    
    # check the interpolation range
    if (np.nanmin(fx) > np.nanmin(nwwv)) | (np.nanmax(fx) > np.nanmax(nwwv)):
        print('rebinspec: new wavelength range probes outside of old wavelength range.')
    
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
   



def spectres(new_spec_wavs, old_spec_wavs, spec_fluxes, spec_errs=None):

    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.
    Parameters
    ----------
    new_spec_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.
    old_spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        old_spec_wavs, last dimension must correspond to the shape of
        old_spec_wavs. Extra dimensions before this may be used to
        include multiple spectra.
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.
    Returns
    -------
    res_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_spec_wavs, other dimensions are the same as
        spec_fluxes.
    resampled_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in
        res_fluxes. Only returned if spec_errs was specified.
    """

    # Arrays of left-hand sides and widths for the old and new bins
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_widths = np.zeros(old_spec_wavs.shape[0])
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_lhs[0] = old_spec_wavs[0]
    spec_lhs[0] -= (old_spec_wavs[1] - old_spec_wavs[0])/2
    spec_widths[-1] = (old_spec_wavs[-1] - old_spec_wavs[-2])
    spec_lhs[1:] = (old_spec_wavs[1:] + old_spec_wavs[:-1])/2
    spec_widths[:-1] = spec_lhs[1:] - spec_lhs[:-1]

    filter_lhs = np.zeros(new_spec_wavs.shape[0]+1)
    filter_widths = np.zeros(new_spec_wavs.shape[0])
    filter_lhs[0] = new_spec_wavs[0]
    filter_lhs[0] -= (new_spec_wavs[1] - new_spec_wavs[0])/2
    filter_widths[-1] = (new_spec_wavs[-1] - new_spec_wavs[-2])
    filter_lhs[-1] = new_spec_wavs[-1]
    filter_lhs[-1] += (new_spec_wavs[-1] - new_spec_wavs[-2])/2
    filter_lhs[1:-1] = (new_spec_wavs[1:] + new_spec_wavs[:-1])/2
    filter_widths[:-1] = filter_lhs[1:-1] - filter_lhs[:-2]

    if filter_lhs[0] < spec_lhs[0] or filter_lhs[-1] > spec_lhs[-1]:
        raise ValueError("spectres: The new wavelengths specified must fall"
                         "within the range of the old wavelength values.")

    # Generate output arrays to be populated
    res_fluxes = np.zeros(spec_fluxes[..., 0].shape + new_spec_wavs.shape)

    if spec_errs is not None:
        if spec_errs.shape != spec_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape"
                             "as spec_fluxes.")
        else:
            res_fluxerrs = np.copy(res_fluxes)

    start = 0
    stop = 0

    # Calculate new flux and uncertainty values, loop over new bins
    for j in range(new_spec_wavs.shape[0]):

        # Find first old bin which is partially covered by the new bin
        while spec_lhs[start+1] <= filter_lhs[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while spec_lhs[stop+1] < filter_lhs[j+1]:
            stop += 1

        # If new bin is fully within one old bin these are the same
        if stop == start:

            res_fluxes[..., j] = spec_fluxes[..., start]
            if spec_errs is not None:
                res_fluxerrs[..., j] = spec_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:

            start_factor = ((spec_lhs[start+1] - filter_lhs[j])
                            / (spec_lhs[start+1] - spec_lhs[start]))

            end_factor = ((filter_lhs[j+1] - spec_lhs[stop])
                          / (spec_lhs[stop+1] - spec_lhs[stop]))

            spec_widths[start] *= start_factor
            spec_widths[stop] *= end_factor

            # Populate res_fluxes spectrum and uncertainty arrays
            f_widths = spec_widths[start:stop+1]*spec_fluxes[..., start:stop+1]
            res_fluxes[..., j] = np.sum(f_widths, axis=-1)
            res_fluxes[..., j] /= np.sum(spec_widths[start:stop+1])

            if spec_errs is not None:
                e_wid = spec_widths[start:stop+1]*spec_errs[..., start:stop+1]

                res_fluxerrs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                res_fluxerrs[..., j] /= np.sum(spec_widths[start:stop+1])

            # Put back the old bin widths to their initial values for later use
            spec_widths[start] /= start_factor
            spec_widths[stop] /= end_factor

    # If errors were supplied return the res_fluxes spectrum and error arrays
    if spec_errs is not None:
        return res_fluxes, res_fluxerrs

    # Otherwise just return the res_fluxes spectrum array
    else:
        return res_fluxes