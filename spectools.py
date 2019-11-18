'''
a bunch of definitions to do the comparison...


'''
# standard imports
import numpy as np
import time
import glob

# astropy imports
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy import constants as const

# dust extinction (may need special import)
from dust_extinction.parameter_averages import F99

# plotting helpers
import matplotlib.pyplot as plt

def spec_difference(spec1,spec2,sclval=1.):
    """for spectra with the same wavelength, compute the total difference
    
    inputs
    ---------
    spec1      : first spectra, with 'wave' and 'flux' defined
    spec2      : second spectra, assumed to have same wavelengths as spec1
    sclval     : scaling value to apply to spec2 before computing the difference
    
    returns
    ---------
    difference quantity: fractional deviation
    
    
    todo
    ---------
    what to do with error handling?
    
    
    """
    if spec1['wave'][0] != spec2['wave'][0]:
        print('compare_spectra: wavelengths are not the same.')
        
    return np.sum(np.abs((spec1['flux'] - sclval*spec2['flux'])/spec1['flux']))

    

def minimize_offset(spec1,spec2,iterate=False):
    """compute a minimum residual offset value
    
    inputs
    ---------
    spec1      : first spectra, with 'wave' and 'flux' defined
    spec2      : second spectra, assumed to have same wavelengths as spec1
    
    
    returns
    ---------
    best-fit scale value for spec2 to minimize difference with spec1
    
    
    dependencies
    ---------
    spec_difference 
    
    
    """
    scalevals = np.linspace(-5.,5.,1000)
    residual = np.zeros(scalevals.size)

    for isclval,sclval in enumerate(scalevals):
        residual[isclval] = spec_difference(spec1,spec2,sclval=10.**sclval)
        
    # do it again for certainty?
    if iterate:
        scalevals = np.linspace(scalevals[residual.argmin()]-1.,scalevals[residual.argmin()]+1.,1000)
        residual = np.zeros(scalevals.size)

        for isclval,sclval in enumerate(scalevals):
            residual[isclval] = spec_difference(spec1,spec2,sclval=10.**sclval)
    
    return 10.**scalevals[residual.argmin()]
    



def compare_spectra(spec1,spec2,plotting=True,scale=True,title='',av=-1.):
    '''compare two spectra with one another

    inputs
    ---------
    spec1      : first spectra, with 'wave' and 'flux' defined
    spec2      : second spectra, assumed to have same wavelengths as spec1
    plotting   : boolean, True creates a diagnostic plot
    scale      : boolean, True if spec2 should be scaled to spec1
    title      : title for plot (if made)
    av         : extinction to apply to spec1
    
    
    returns
    ---------
    offset : best fit scaling value to make spec2 match spec1
    
    dependencies
    ---------
    minimize_offset
    spec_difference
    
    todo
    ---------
    
    '''
    # what cleaning is necessary to compare?
    
    if spec1['wave'][0] != spec2['wave'][0]:
        print('compare_spectra: wavelengths are not the same.')
        
    compspec = {}
    compspec['wave'] = spec1['wave']
        
    # apply extinction if desired
    if av>0.:
        ext = F99(Rv=3.1) # 3.1
        compspec['flux'] = spec1['flux']*ext.extinguish(compspec['wave']*u.um, Av=av)
        
    else:
        compspec['flux'] = spec1['flux']
        
        
    # if scale is True, try to minimize the residuals with an overall fit to the amplitude
    if scale:
        # find the scaling for spec2 which minimizes the difference with compspec (extincted spec1)
        offset = minimize_offset(compspec,spec2,iterate=True)
    else:
        offset = 1.
        
    total_residual = np.sum(np.abs((compspec['flux']-offset*spec2['flux'])/compspec['flux']))
    total_residual /= spec2['flux'].size


    if plotting:
        fig = plt.figure(figsize=(10,3))
        ax1 = fig.add_axes([0.2,0.5,0.7,0.47])
        ax2 = fig.add_axes([0.2,0.22,0.7,0.2])


        ax1.plot(compspec['wave'],compspec['flux'],color='black')
        ax1.plot(spec2['wave'],offset*spec2['flux'],color='red')

        ax2.plot(compspec['wave'],(compspec['flux']-offset*spec2['flux'])/compspec['flux'],color='black')
        ax2.plot([np.nanmin(spec1['wave']),np.nanmax(spec1['wave'])],[0.,0.],color='gray',lw=1.,linestyle='dashed')

        ax2.text(np.nanmax(spec1['wave']),0.,'e={}\nA={}\nAv={}'.format(np.round(total_residual,2),np.round(offset,3),np.round(av,2)),color='black',ha='left')

        ax1.set_xlim([np.nanmin(spec1['wave']),np.nanmax(spec1['wave'])])
        ax2.axis([np.nanmin(spec1['wave']),np.nanmax(spec1['wave']),-1.,1.])
        ax1.set_title(title,size=12)
        ax2.set_xlabel('Wavelength ($\mu$m)',size=12)
        ax2.set_ylabel('Residuals',size=12)
        ax1.set_ylabel('Flux',size=12)
        ax1.set_xticklabels(())

    return offset

    

def compare_spectra_extinction(spec1,spec2,plotting=True,scale=True,title='',extmax=2.,extnum=10):
    '''compare two spectra with one another, testing extinction range
    
    inputs
    ---------
    spec1      : first spectra, with 'wave' and 'flux' defined
    spec2      : second spectra, assumed to have same wavelengths as spec1
    plotting   : boolean, True creates a diagnostic plot
    scale      : boolean, True if spec2 should be scaled to spec1
    extmax     : maximum extinction value to test
    extnum     : number of extinction values to test
    
    
    returns
    ---------
    offset : best fit scaling value to make spec2 match spec1
    
    dependencies
    ---------
    minimize_offset
    spec_difference
    
    todo
    ---------    
    
    '''
    # what cleaning is necessary to compare?
    
    if spec1['wave'][0] != spec2['wave'][0]:
        print('compare_spectra: wavelengths are not the same.')
        
    # set up to apply extinction
    compspec = {}
    compspec['wave'] = spec1['wave']
    ext = F99(Rv=3.1) # 3.1
        
    
    extrange = np.linspace(0.,extmax,extnum)
    total_residual = np.zeros(extrange.size)

    fig = plt.figure(figsize=(10,3))
    ax1 = fig.add_axes([0.2,0.5,0.7,0.47])
    ax2 = fig.add_axes([0.2,0.22,0.7,0.2])


    for iav,av in enumerate(extrange):
        compspec['flux'] = spec1['flux']*ext.extinguish(compspec['wave']*u.um, Av=av)
        offset = minimize_offset(spec2,compspec,iterate=True)
        
        # I feel that this needs some scaling to account for the uncertainty in the comparison flux...
        total_residual[iav] = np.sum(np.abs(spec2['flux']-offset*compspec['flux'])/spec2['flux'])/spec2['flux'].size
    
        if iav==0:
            ax1.plot(spec2['wave'],offset*compspec['flux'],color='darkgray',lw=1.5) # with no extinction   
        else:
            ax1.plot(spec2['wave'],offset*compspec['flux'],color='gray',alpha=0.5,lw=0.5)
            

        if iav==0:
            ax2.plot(spec2['wave'],(spec2['flux']-offset*compspec['flux'])/spec2['flux'],color='darkgray',lw=1.5)
        else:
            ax2.plot(spec2['wave'],(spec2['flux']-offset*compspec['flux'])/spec2['flux'],color='gray',alpha=0.5,lw=0.5)
        
        ax2.plot([np.nanmin(spec1['wave']),np.nanmax(spec1['wave'])],[0.,0.],color='gray',lw=1.,linestyle='dashed')

    ax1.plot(spec1['wave'],spec2['flux'],color='red',lw=1.5)


    # plot the best residual
    best_residual = np.argmin(total_residual)
    compspec['flux'] = spec1['flux']*ext.extinguish(compspec['wave']*u.um, Av=extrange[best_residual])
    offset = minimize_offset(spec2,compspec,iterate=True)
    ax1.plot(compspec['wave'],offset*compspec['flux'],color='black',lw=1.0)
    ax2.plot(compspec['wave'],(spec2['flux']-offset*compspec['flux'])/spec2['flux'],color='black',lw=1.0)
    ax2.text(np.nanmax(spec1['wave']),0.,'e={}\nA={}\nAv={}'.format(np.round(np.nanmin(total_residual),2),np.round(offset,3),np.round(extrange[best_residual],2)),color='black',ha='left')
        
    ax1.set_xlim([np.nanmin(spec1['wave']),np.nanmax(spec1['wave'])])
    ax2.axis([np.nanmin(spec1['wave']),np.nanmax(spec1['wave']),-1.,1.])
    ax1.set_title(title,size=12)
    ax2.set_xlabel('Wavelength ($\mu$m)',size=12)
    ax2.set_ylabel('Residuals',size=12)
    ax1.set_ylabel('Flux',size=12)
    ax1.set_xticklabels(())


    

