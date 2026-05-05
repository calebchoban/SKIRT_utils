from astropy import units as u
from astropy import constants as const
from astropy.io import fits
from astropy.convolution import convolve_fft,Gaussian2DKernel
from astropy.nddata import block_reduce
from astropy.table import QTable
from scipy import integrate
from scipy.optimize import curve_fit
import numpy as np

try:   
    import stpsf
except:
    print("Need to install stpsf and download its corresponding files to use JWST and Roman PSFs. \n Generic Gaussian PSFs will be used otherwise. \n Follow instructions here \n https://stpsf.readthedocs.io/en/latest/installation.html#installation \n")


# Broadbands and corresponding pivot wavelengths taken from the SKIRT documentation
# https://skirt.ugent.be/skirt9/class_broad_band.html
# Useful for determining which broadbands are in a SKIRT FITS file.
filter_pivot_wavelengths = {
    "2MASS_2MASS_J":      1.2393  * u.micron,
    "2MASS_2MASS_H":      1.6494  * u.micron,
    "2MASS_2MASS_KS":     2.1638  * u.micron,
    "ALMA_ALMA_10":       349.89  * u.micron,
    "ALMA_ALMA_9":        456.2   * u.micron,
    "ALMA_ALMA_8":        689.59  * u.micron,
    "ALMA_ALMA_7":        937.98  * u.micron,
    "ALMA_ALMA_6":        1244.4  * u.micron,
    "ALMA_ALMA_5":        1616    * u.micron,
    "ALMA_ALMA_4":        2100.2  * u.micron,
    "ALMA_ALMA_3":        3043.4  * u.micron,
    "EUCLID_VIS_VIS":     0.71032 * u.micron,
    "EUCLID_NISP_Y":      1.0808  * u.micron,
    "EUCLID_NISP_J":      1.3644  * u.micron,
    "EUCLID_NISP_H":      1.7696  * u.micron,
    "GALEX_GALEX_FUV":    0.15351 * u.micron,
    "GALEX_GALEX_NUV":    0.23008 * u.micron,
    "GENERIC_JOHNSON_U":  0.35236 * u.micron,
    "GENERIC_JOHNSON_B":  0.44146 * u.micron,
    "GENERIC_JOHNSON_V":  0.55223 * u.micron,
    "GENERIC_JOHNSON_R":  0.68967 * u.micron,
    "GENERIC_JOHNSON_I":  0.87374 * u.micron,
    "GENERIC_JOHNSON_J":  1.2429  * u.micron,
    "GENERIC_JOHNSON_M":  5.0114  * u.micron,
    "HERSCHEL_PACS_70":   70.77   * u.micron,
    "HERSCHEL_PACS_100":  100.8   * u.micron,
    "HERSCHEL_PACS_160":  161.89  * u.micron,
    "HERSCHEL_SPIRE_250": 252.55  * u.micron,
    "HERSCHEL_SPIRE_350": 354.27  * u.micron,
    "HERSCHEL_SPIRE_500": 515.36  * u.micron,
    "IRAS_IRAS_12":       11.4    * u.micron,
    "IRAS_IRAS_25":       23.605  * u.micron,
    "IRAS_IRAS_60":       60.344  * u.micron,
    "IRAS_IRAS_100":      101.05  * u.micron,
    "JCMT_SCUBA2_450":    449.3   * u.micron,
    "JCMT_SCUBA2_850":    853.81  * u.micron,
    "PLANCK_HFI_857":     352.42  * u.micron,
    "PLANCK_HFI_545":     545.55  * u.micron,
    "PLANCK_HFI_353":     839.3   * u.micron,
    "PLANCK_HFI_217":     1367.6  * u.micron,
    "PLANCK_HFI_143":     2130.7  * u.micron,
    "PLANCK_HFI_100":     3001.1  * u.micron,
    "PLANCK_LFI_70":      4303    * u.micron,
    "PLANCK_LFI_44":      6845.9  * u.micron,
    "PLANCK_LFI_30":      10674   * u.micron,
    "RUBIN_LSST_U":       0.368   * u.micron,
    "RUBIN_LSST_G":       0.47823 * u.micron,
    "RUBIN_LSST_R":       0.62178 * u.micron,
    "RUBIN_LSST_I":       0.75323 * u.micron,
    "RUBIN_LSST_Z":       0.86851 * u.micron,
    "RUBIN_LSST_Y":       0.97301 * u.micron,
    "SLOAN_SDSS_U":       0.35565 * u.micron,
    "SLOAN_SDSS_G":       0.47024 * u.micron,
    "SLOAN_SDSS_R":       0.61755 * u.micron,
    "SLOAN_SDSS_I":       0.74899 * u.micron,
    "SLOAN_SDSS_Z":       0.89467 * u.micron,
    "SPITZER_IRAC_I1":    3.5508  * u.micron,
    "SPITZER_IRAC_I2":    4.496   * u.micron,
    "SPITZER_IRAC_I3":    5.7245  * u.micron,
    "SPITZER_IRAC_I4":    7.8842  * u.micron,
    "SPITZER_MIPS_24":    23.759  * u.micron,
    "SPITZER_MIPS_70":    71.987  * u.micron,
    "SPITZER_MIPS_160":   156.43  * u.micron,
    "SWIFT_UVOT_UVW2":    0.20551 * u.micron,
    "SWIFT_UVOT_UVM2":    0.22462 * u.micron,
    "SWIFT_UVOT_UVW1":    0.25804 * u.micron,
    "SWIFT_UVOT_U":       0.34628 * u.micron,
    "SWIFT_UVOT_B":       0.43496 * u.micron,
    "SWIFT_UVOT_V":       0.54254 * u.micron,
    "TNG_OIG_U":          0.37335 * u.micron,
    "TNG_OIG_B":          0.43975 * u.micron,
    "TNG_OIG_V":          0.53727 * u.micron,
    "TNG_OIG_R":          0.63917 * u.micron,
    "TNG_NICS_J":         1.2758  * u.micron,
    "TNG_NICS_H":         1.6265  * u.micron,
    "TNG_NICS_K":         2.2016  * u.micron,
    "UKIRT_UKIDSS_Z":     0.88263 * u.micron,
    "UKIRT_UKIDSS_Y":     1.0314  * u.micron,
    "UKIRT_UKIDSS_J":     1.2501  * u.micron,
    "UKIRT_UKIDSS_H":     1.6354  * u.micron,
    "UKIRT_UKIDSS_K":     2.2058  * u.micron,
    "WISE_WISE_W1":       3.3897  * u.micron,
    "WISE_WISE_W2":       4.6406  * u.micron,
    "WISE_WISE_W3":       12.568  * u.micron,
    "WISE_WISE_W4":       22.314  * u.micron,
    "JWST_NIRCAM_F070W":  0.7039  * u.micron,
    "JWST_NIRCAM_F090W":  0.9022  * u.micron,
    "JWST_NIRCAM_F115W":  1.154   * u.micron,
    "JWST_NIRCAM_F140M":  1.405   * u.micron,
    "JWST_NIRCAM_F150W":  1.501   * u.micron,
    "JWST_NIRCAM_F162M":  1.627   * u.micron,
    "JWST_NIRCAM_F164N":  1.645   * u.micron,
    "JWST_NIRCAM_F150W2": 1.659   * u.micron,
    "JWST_NIRCAM_F182M":  1.845   * u.micron,
    "JWST_NIRCAM_F187N":  1.874   * u.micron,
    "JWST_NIRCAM_F200W":  1.989   * u.micron,
    "JWST_NIRCAM_F210M":  2.095   * u.micron,
    "JWST_NIRCAM_F212N":  2.121   * u.micron,
    "JWST_NIRCAM_F250M":  2.503   * u.micron,
    "JWST_NIRCAM_F277W":  2.762   * u.micron,
    "JWST_NIRCAM_F300M":  2.989   * u.micron,
    "JWST_NIRCAM_F322W2": 3.232   * u.micron,
    "JWST_NIRCAM_F323N":  3.237   * u.micron,
    "JWST_NIRCAM_F335M":  3.362   * u.micron,
    "JWST_NIRCAM_F356W":  3.568   * u.micron,
    "JWST_NIRCAM_F360M":  3.624   * u.micron,
    "JWST_NIRCAM_F405N":  4.052   * u.micron,
    "JWST_NIRCAM_F410M":  4.082   * u.micron,
    "JWST_NIRCAM_F430M":  4.281   * u.micron,
    "JWST_NIRCAM_F444W":  4.404   * u.micron,
    "JWST_NIRCAM_F460M":  4.630   * u.micron,
    "JWST_NIRCAM_F466N":  4.654   * u.micron,
    "JWST_NIRCAM_F470N":  4.708   * u.micron,
    "JWST_NIRCAM_F480M":  4.818   * u.micron,
    "JWST_MIRI_F560W":    5.635   * u.micron,
    "JWST_MIRI_F770W":    7.639   * u.micron,
    "JWST_MIRI_F1000W":   9.953   * u.micron,
    "JWST_MIRI_F1130W":   11.31   * u.micron,
    "JWST_MIRI_F1280W":   12.81   * u.micron,
    "JWST_MIRI_F1500W":   15.06   * u.micron,
    "JWST_MIRI_F1800W":   17.98   * u.micron,
    "JWST_MIRI_F2100W":   20.80   * u.micron,
    "JWST_MIRI_F2550W":   25.36   * u.micron,
}

filter_names = np.array(list(filter_pivot_wavelengths.keys()))
filter_wavelengths = u.Quantity(list(filter_pivot_wavelengths.values()))


class IFU(object):
    '''
    This class is used to extract images from IFU data cubes.
    '''

    def __init__(self, 
                 dirc: str, 
                 inst_file: str, 
                 verbose: bool = True):
        '''
        Parameters
        ----------
        dirc : string
            Directory where the FITS file is located.
        inst_file : string
            Name of the FITS file containing the IFU data cube.
        verbose : bool, optional
            Whether to print information about the FITS file and the filters it contains. Default is True.
        '''
        self.dirc = dirc
        self.inst_file = inst_file
        self.verbose = verbose

        hdul = fits.open(self.dirc+self.inst_file)
        self.images = hdul[0].data * u.Unit(hdul[0].header['BUNIT'])
        self.pivot_wavelengths = (hdul[1].data['GRID_POINTS'] * u.Unit(hdul[0].header['CUNIT3'])).to('micron')
        self.pixel_res_angle = (hdul[0].header['CDELT1'] * u.Unit(hdul[0].header['CUNIT1'])).to('arcsec') # arcsec
        self.num_pixels = np.asarray([hdul[0].header['NAXIS1'],hdul[0].header['NAXIS2']])
        self.fov_angle = self.num_pixels * self.pixel_res_angle.to('arcsec')

        if 'DISTANGD' in hdul[0].header and 'DISTUNIT' in hdul[0].header:
            self.angular_distance = (hdul[0].header['DISTANGD'] * (u.Unit(hdul[0].header['DISTUNIT']))/u.Unit('rad')).to('kpc/rad') # kpc
            self.pixel_res_physical = self.angular_distance.to('kpc/rad') * self.pixel_res_angle.to('rad')
            self.fov_physical = self.num_pixels * self.pixel_res_physical.to('kpc')
        else:
            if verbose:
                print("Not given angular distance in FITS file. Cannot determine physical pixel resolution and FOV. Set to None.")
            self.angular_distance = None
            self.pixel_res_physical = None
            self.fov_physical = None
        
    
        # Determine what filters corresponds to the pivot wavelengths in the FITS file
        # Note you can have custom filters so this will let you know if now known filter matches.
        self.filters = ["N/A" for i in range(len(self.pivot_wavelengths))]
        self.unknown_filter = 0
        for i,wavelength in enumerate(self.pivot_wavelengths):
            idx = np.nanargmin(np.abs(wavelength - filter_wavelengths))
            if np.abs(wavelength - filter_wavelengths[idx])/filter_wavelengths[idx] > 0.001: # Needs to be small since commonly used filers can have slightly different pivot wavelengths for different telescopes
                self.unknown_filter += 1
            else:
                self.filters[i] = filter_names[idx]

        self.filters = np.array(self.filters)

        if self.verbose:
            print("FITS files info")
            print(hdul.info())
            print(hdul[0].header)
            print("Pixel brightness in units of", self.images.unit)
            print("Pixel resolution", self.pixel_res_angle, self.pixel_res_physical)
            print("Image FOV",self.fov_physical,self.fov_angle)
            print("There are %i photometric images"%len(self.pivot_wavelengths))
            print(f"{'Filter Name':<20} {'Pivot Wavelength':<15}")
            print("-" * 50)
            for i,wavelength in enumerate(self.pivot_wavelengths):
                print(f"{self.filters[i]:<20} {wavelength:<15.5f}")
            if self.unknown_filter:
                print("WARNING: %i filters in the FITS file do not match any known SKIRT filter. Check the SKIRT .ski file to see if they are custom filters."%self.unknown_filter)

        hdul.close()


    def get_filter_image(self, 
                         filter: str, 
                         psf: PSF|None = None, 
                         brightness_units: str|None=None, 
                         downsample_resolution: float|None=None, 
                         downsample_factor: int=1, 
                         sum_downsample: bool=False,):
        '''
        This function extracts the image data for the specified filter.

        Parameters
        ----------
        filter : string
            Name of filter to extract from FITS file.
        psf : Telescope_PSF, optional
            PSF you want to convolve the image with. Default is None.
        brightness_units : string, optional
            Surface brightness units wanted (wavelength, frequency, neutral). Default is None which uses units in FITS file.
        downsample_resolution : double, optional
            Resolution in arsec to downsample the image to. Default is None. Overrides downsample_factor if provided.
        downsample_factor : int, optional
            Integer factor you want to downsample the image by. Default is 1.
        sum_downsample : bool, optional
            Whether to sum the pixel values when downsampling instead of taking the average. Default is False since images are typically in surface brightnesses units which should be averaged since they are per unit area. This is mainly used when downsampling images with reliability statistics.
        Returns
        -------
        image : ndarray (N,N)
            NxN pixel image for the specified filter.
        '''

        has_filter = False
        for filt in self.filters:
            if filter in filt:
                filter = filt
                has_filter = True
                break

        if not has_filter:
            print("Filter %s not found in FITS file."%filter)
            return None

        idx = np.where(self.filters == filter)
        image = np.copy(self.images[idx][0])

        # Determine surface brightness units of images
        image_units = None
        if image.unit.is_equivalent(u.MJy / u.sr):
            image_units = 'frequency'
        elif image.unit.is_equivalent(u.erg / u.cm**2 / u.s / u.micron / u.sr):
            image_units = 'wavelength'
        elif image.unit.is_equivalent(u.erg / u.cm**2 / u.s / u.sr):
            image_units = 'neutral'
        else:
            print("WARNING: Unknown brightness units. Output image will have no units.")
            image_units = None
        

        if self.verbose:
            print("%s image brightness is per %s with units %s"%(filter,image_units, image.unit))

        # Convert to different units if requested
        if brightness_units is not None and image_units != brightness_units:
            wavelength = self.get_filter_wavelength(filter)
            if image_units != 'neutral':
                if brightness_units == 'wavelength':    
                    # Convert to F_lambda units
                    image = image.to(u.erg / u.cm**2 / u.s / u.angstrom / u.sr,
                    equivalencies=u.spectral_density(wavelength))
                elif brightness_units == 'frequency':
                    # Convert to F_nu units
                    image = image.to(u.erg / u.cm**2 / u.s / u.Hz/ u.sr,
                    equivalencies=u.spectral_density(wavelength))
                elif brightness_units == 'neutral':
                    # Convert to neutral (lambda F_lamda = nu F_nu) units
                    image = image.to(u.erg / u.cm**2 / u.s / u.micron / u.sr,
                    equivalencies=u.spectral_density(wavelength))*wavelength.to('micron')
            else:
                # Convert to F_lambda units
                if brightness_units == 'wavelength':
                    image = image / wavelength.to('micron')
                # Convert to F_nu units
                if brightness_units == 'frequency':
                    image = image / wavelength.to('Hz', equivalencies=u.spectral())

            if self.verbose:
                print("Image brightness units converted to %s."%image.unit)


        # Convolve with the telescope PSF. This is the effects of the telescope mirrors, 
        # reflectors, etc. before light hits the detectors. You always want to apply the PSF
        # first and then downsample.
        if psf is not None:
            if psf.filter not in filter:
                raise ValueError("PSF filter %s does not match image filter %s."%(psf.filter, filter))
            if np.abs(psf.pixel_scale - self.pixel_res_angle)/self.pixel_res_angle > 0.25:
                print("\t WARNING: PSF pixel scale %s does not match image pixel \n \t scale %s " \
                "within 25%%. Convolved image may be inaccurate."%(psf.pixel_scale, self.pixel_res_angle))
            if self.verbose:
                print("Convolving image with PSF for filter %s."%psf.filter)
                print("PSF has pixel scale %s and image has pixel scale %s."%(psf.pixel_scale, self.pixel_res_angle))
            image = convolve_fft(image, psf.psf_data)
        
        # Down sample instrument resolution to a lower resolution
        if downsample_resolution is not None or downsample_factor > 1:
            if downsample_resolution is not None:
                desired_resolution = downsample_resolution * u.arcsec
                downsample_factor = int(np.round(desired_resolution/self.pixel_res_angle))
            new_resolution = self.pixel_res_angle*downsample_factor
            if downsample_factor < 2:
                print("Desired resolution is either higher than the given image or >0.5 of given image so nothing to downsample.")
            else:
                print("Downsampling image from %s to %s arcsec."%(self.pixel_res_angle.to('arcsec'), new_resolution.to('arcsec')))

            if sum_downsample:
                reduce_func = np.sum # reducing resolution means new larger pixels have the sum of the smaller pixels
            else:
                reduce_func = np.mean # reducing resolution means new larger pixels have the average brigtness of the smaller pixels
            image = block_reduce(image, downsample_factor, func = reduce_func) 

        return image
    

    def cut_image_to_fov(self, image, fov):
        """
        Parameters
        ----------
        image : ndarray (N,N)
            NxN pixel image you want to cut to a specified fov.
        fov : list
            Set the x,y FOV of the image in kpc as a single value of specific x and y values.
        Returns
        -------
        cut_image : ndarray (M,L)
            MxL image cut out of the original image to the desired FOV

        """
        
        fov = np.asarray(fov)*u.kiloparsec
        if np.any(fov>self.fov_physical):
            print("Given FOV %f kpc is larger than instrument FOV %f kpc!"%(fov, self.fov_physical))
            return
        else:
            print(fov,self.fov_physical)
            cut_pix = (np.floor(self.num_pixels * fov / self.fov_physical).value/2).astype(np.int32)
            middle_pix = np.floor(self.num_pixels/2).astype(np.int32)
            cut_image = image[middle_pix[0]-cut_pix[0]:middle_pix[0]+cut_pix[0],middle_pix[0]-cut_pix[1]:middle_pix[1]+cut_pix[1]]
            return cut_image
        
    
    def get_filter_wavelength(self, filter):
        '''
        This function extracts the pivot wavelength for the specified filter.

        Parameters
        ----------
        filter : string
            Name of filter to extract from FITS file.

        Returns
        -------
        wavelength : float
            Pivot wavelength for the specified filter.
        '''
        if filter not in self.filters:
            print("Filter %s not found in FITS file. Returning None."%filter)
            return None

        idx = np.where(self.filters == filter)
        return self.pivot_wavelengths[idx]
    








class Telescope_PSF(object):
    '''
    This class is used to create a telescope PSF to be convolved with a given SKIRT image. The PSF can be a simple Gaussian with 
    user specified resoltuion for a generic telescope or a realistic PSF for JWST NIRCam and MIRI filters at each filters resolution 
    using stpsf. Since the SKIRT image is not guaranteed to be at the same resolution as the telescope, the get_psf function allows you to specify the resolution of the telescope and the relative resolution of the SKIRT instrument and the telescope to determine the FWHM of the Gaussian PSF. For JWST NIRCam and MIRI filters, stpsf provides PSFs at different resolutions so you can specify whether to use the instrument resolution or the telescope resolution to determine which PSF to use.
    '''

    def __init__(self,
                 IFU: IFU,
                 filter: str,  
                 verbose: bool = True):
    
        self.IFU = IFU
        self.filter = filter
        self.verbose = verbose
        self.psf = None

        self.telescope = 'GENERIC'
        if filter != 'GENERIC':
            if 'NIRCAM' in filter:
                self.telescope = 'NIRCAM'
            elif 'MIRI' in filter:
                self.telescope = 'MIRI'
            else:
                if self.verbose and filter:
                    print("WARNING: No telescope found for filter %s. Using GENERIC telescope."%filter)

        
    def get_psf(self,   
                telescope_res: float = 0.1,
                override_instrument_res: float = 0,
                ):
        '''
        This function creates a PSF for the specified telescope and filter. It can create a Gaussian PSF for a generic telescope or use stpsf to create a more realistic PSF for JWST NIRCam and MIRI filters.

        Parameters
        ----------
        telescope_res : float, optional
            Resolution of the telescope in arcsec you want to calculate the PSF for only used for a generic telescope. Default is 0.1 arcsec.
        override_instrument_res : float, optional
            If provided, this will override the SKIRT instrument resolution. Default is 0.
        
        Returns
        -------
        psf : ndarray (M,M)
            PSF array for the specified telescope and filter. For a generic telescope this will be a Gaussian. For JWST NIRCam and MIRI filters this a PSF provided by stpsf.
        '''
        
        telescope_res = telescope_res

        if self.telescope == 'GENERIC':
            self.get_gaussian_psf(telescope_res=telescope_res,
                                  override_instrument_res=override_instrument_res)
        elif self.telescope == 'NIRCAM':
            self.get_NIRCam_psf(override_instrument_res=override_instrument_res)
        elif self.telescope == 'MIRI':
            self.get_MIRI_psf(override_instrument_res=override_instrument_res)
        else:
            print("Something went wrong. Telescope %s not supported."%self.telescope)
            return None
        
        return self.psf


    def get_gaussian_psf(self,
                        telescope_res: float = 0.1,
                        override_instrument_res: float = 0,):
        '''
        Creates a Gaussian PSF for a generic telescope. The FWHM of the Gaussian is determined by the relative resolution of the SKIRT instrument and the telescope. If use_instrument_res is False then the instrument and telescope are assumed to have the same resolution and the FWHM of the Gaussian is 1 pixel.

        Parameters
        ----------
        telescope_res : float, optional
            Resolution of the telescope in arcsec you want to calculate the PSF for only used for a generic telescope. Default is 0.1 arcsec.
        override_instrument_res : float, optional
            If provided, this will override the instrument resolution. Default is 0.
        '''
        
        # Telescope resolution in arcsecond
        telescope_resolution = telescope_res * u.arcsec
        # SKIRT instrument can have any resolution. 
        # The relative resolution of the instrument and the actual telescope will determine the Gaussian sigma
        if self.IFU is not None and override_instrument_res == 0:
            instrument_res = self.IFU.pixel_res_angle.to('arcsec') # in arcsec
        else: instrument_res= override_instrument_res * u.arcsec # Override resolution of telescope
        # calculate the sigma in pixels.
        sigma = telescope_resolution/instrument_res

        # Oversampling factor of the PSF which is then downsampled to the telescope resolution. Default of 4 usually assumed most PSFs.
        oversample_factor = 4 
        gauss_psf = Gaussian2DKernel(sigma, factor = oversample_factor, mode='oversample')
        self.psf = PSF('GENERIC', gauss_psf.array, pixel_scale=instrument_res)

        if self.verbose:
            print(f"PSF model for filter GENERIC has pixel scale {telescope_res:.4g} arcsec / pixel and sigma {sigma:.4g} pixels.\n")


    def get_NIRCam_psf(self,
                       override_instrument_res: float = 0,):
        '''
        Creates a NIRCam PSF for the specified filter.
        Parameters
        ----------
        override_instrument_res : float, optional
            If provided, this will override the instrument resolution. Default is 0.

        '''

        # Only need filter number (i.e. F070W) and not the full name (i.e. JWST_NIRCAM_F070W)
        filter_name = self.filter.split('_')[-1]
        # Name of extension in the FITS file we want to use for the PSF.
        # DET_SAMP is the PSF sampled at the detector pixel scale (i.e. the true PSF of the telescope). 
        # OVERSAMP is the PSF oversampled by the oversample_factor (i.e. the PSF that is convolved with 
        # the image before downsampling to the telescope resolution).
        # DET_DIST and OVERDIST are the same but including additional "real world" effects such as 
        # geometric distortion of the instruments, and detector charge transfer and interpixel capacitance.
        extension_name= "DET_DIST" # select the most "realistic" PSF
        # Calculate npixels for the telescope given the npixels and resolution of the instrument given
        # Else we assume the instrument and telescope have the same resolution
        if self.IFU is not None and override_instrument_res == 0:
            instrument_res = self.IFU.pixel_res_angle.to('arcsec')
        else: instrument_res=override_instrument_res * u.arcsec # Override resolution of telescope
        # Have to run this once to get the pixel scale for the instrument
        nircam = stpsf.NIRCam()
        nircam.filter = filter_name
        psf_hdulist = nircam.calc_psf()
        telescope_res = psf_hdulist[extension_name].header["PIXELSCL"] * u.arcsec
        downsample_factor = int(np.round(telescope_res/instrument_res))
        if self.verbose:
            print(f"JWST NIRCam filter {filter_name} has resolution of {telescope_res} / pixel.")
            print(f"SKIRT instrument has resolution of {instrument_res} / pixel.")
            if downsample_factor >= 2:
                print(f"Filter PSF will be oversampled by {downsample_factor} to account for the higher resolution of the instrument.")
        if telescope_res/instrument_res < 0.75:
            print(f"WARNING: Instrument has lower resolution than telescope filter {filter_name}.")
            print("Convolving with this PSF will underpredict the effects of the PSF. Consider using a finer resolution instrument.")

        # Create NIRCam PSF model
        oversample_factor = 4 * np.max([1,downsample_factor]) # This sets the oversampling factor of the PSF which is then downsampled to the telescope detector resolution. Default of 4 is ok.
        nircam = stpsf.NIRCam()
        nircam.filter = filter_name
        psf_hdulist = nircam.calc_psf(
            oversample=oversample_factor,
            outfile=None,
        )

        if downsample_factor < 2:
            extension_name= "DET_DIST" # If the instrument resolution is close to the telescope resolution then use the PSF sampled at the detector pixel scale.
            psf = psf_hdulist[extension_name].data
            pixel_scale = psf_hdulist[extension_name].header["PIXELSCL"] * u.arcsec
        else:
            extension_name= "OVERDIST" # If the instrument resolution is finer than the telescope resolution then use the oversampled PSF and downsample it to the instrument resolution.
            psf = psf_hdulist[extension_name].data
            # Downsample oversampled PSF
            psf = block_reduce(psf, block_size=oversample_factor/downsample_factor, func=np.mean)
            pixel_scale = psf_hdulist[extension_name].header["PIXELSCL"] * oversample_factor/downsample_factor * u.arcsec

        psf /= np.sum(psf)  # normalize PSF

        if self.verbose:
            print(f"PSF model for filter {filter_name} has pixel scale {pixel_scale:.4g} / pixel.\n")
        
        self.psf = PSF(filter_name, psf, pixel_scale)


    def get_MIRI_psf(self,
                    override_instrument_res: bool = True,):
        '''
        Creates a MIRI PSF for the specified filter.
        Parameters
        ----------
        override_instrument_res : float, optional
            If provided, this will override the instrument resolution. Default is 0.

        '''

        # Only need filter number (i.e. F770W) and not the full name (i.e. JWST_MIRI_F770W)
        filter_name = self.filter.split('_')[-1]
        # Name of extension in the FITS file we want to use for the PSF.
        # DET_SAMP is the PSF sampled at the detector pixel scale (i.e. the true PSF of the telescope). 
        # OVERSAMP is the PSF oversampled by the oversample_factor (i.e. the PSF that is convolved with 
        # the image before downsampling to the telescope resolution).
        # DET_DIST and OVERDIST are the same but including additional "real world" effects such as 
        # geometric distortion of the instruments, and detector charge transfer and interpixel capacitance.
        extension_name= "DET_DIST" # select the most "realistic" PSF
        # Calculate npixels for the telescope given the npixels and resolution of the instrument given
        # Else we assume the instrument and telescope have the same resolution
        if self.IFU is not None and override_instrument_res == 0:
            instrument_res = self.IFU.pixel_res_angle.to('arcsec')
        else: instrument_res=override_instrument_res * u.arcsec # Override resolution of telescope
        # Have to run this once to get the pixel scale for the instrument
        miri = stpsf.MIRI()
        miri.filter = filter_name
        psf_hdulist = miri.calc_psf()
        telescope_res = psf_hdulist[extension_name].header["PIXELSCL"] * u.arcsec
        downsample_factor = int(np.round(telescope_res/instrument_res))
        if self.verbose:
            print(f"JWST MIRI filter {filter_name} has resolution of {telescope_res} / pixel.")
            print(f"SKIRT instrument has resolution of {instrument_res} / pixel.")
            if downsample_factor >= 2:
                print(f"Filter PSF will be oversampled by {downsample_factor} to account for the higher resolution of the instrument.")
        if telescope_res/instrument_res < 0.75:
            print(f"WARNING: Instrument has lower resolution than telescope filter {filter_name}.")
            print("Convolving with this PSF will underpredict the effects of the PSF. Consider using a finer resolution instrument.")

        # Create MIRI PSF model
        oversample_factor = 4 * np.max([1,downsample_factor]) # This sets the oversampling factor of the PSF which is then downsampled to the telescope detector resolution. Default of 4 is ok.
        miri = stpsf.MIRI()
        miri.filter = filter_name
        psf_hdulist = miri.calc_psf(
            oversample=oversample_factor,
            outfile=None,
        )

        if downsample_factor < 2:
            extension_name= "DET_DIST" # If the instrument resolution is close to the telescope resolution then use the PSF sampled at the detector pixel scale.
            psf = psf_hdulist[extension_name].data
            pixel_scale = psf_hdulist[extension_name].header["PIXELSCL"] * u.arcsec
        else:
            extension_name= "OVERDIST" # If the instrument resolution is finer than the telescope resolution then use the oversampled PSF and downsample it to the instrument resolution.
            psf = psf_hdulist[extension_name].data
            # Downsample oversampled PSF
            psf = block_reduce(psf, block_size=oversample_factor/downsample_factor, func=np.mean)
            pixel_scale = psf_hdulist[extension_name].header["PIXELSCL"] * oversample_factor/downsample_factor * u.arcsec

        psf /= np.sum(psf)  # normalize PSF

        if self.verbose:
            print(f"PSF model for filter {filter_name} has pixel scale {pixel_scale:.4g} / pixel.\n")
        
        self.psf = PSF(filter_name, psf, pixel_scale)


class PSF():
    '''
    PSF data structure to store the PSF array and its properties such as pixel scale, etc. This is used to convolve with the SKIRT images to create realistic images at the resolution of a given telescope.
    '''

    def __init__(self, filter: str, psf_data: np.ndarray, pixel_scale: u.Quantity):
        self.filter = filter
        self.psf_data = psf_data
        self.pixel_scale = pixel_scale

    def __str__(self):
        shape = self.psf_data.shape
        return (
            f"PSF(filter={self.filter}, "
            f"shape={shape}, "
            f"pixel_scale={self.pixel_scale})"
        )

    def __repr__(self):
        return self.__str__()


class SED:
    """
    A class to read and parse SKIRT SED (Spectral Energy Distribution) files.
    
    This class can handle SED files with the following format:
    - First line contains distance information: "# SED at inclination X deg, azimuth Y deg, distance Z Unit"
    - Subsequent comment lines describe columns: "# column N: description; variable_name (unit)"
    - Followed by numerical data in columns
    
    Example usage:
    -------------
    # Create SED reader object
    sed = SED('path/to/sed_file.txt')
    
    # Read the data 
    sed.read_data()
    
    # Access metadata
    print(f"Distance: {sed.distance}")
    print(f"Columns: {sed.columns}")
    
    # Access wavelength and flux data
    wavelength = sed.wavelength  # First column with units
    total_flux = sed.total_flux  # Second column with units
    
    # Access specific columns by index or name
    transparent_flux = sed.get_column(2)  # Third column (0-indexed)
    direct_primary = sed.get_column_by_name('direct primary flux')
    
    # Print summary
    print(sed)
    """
    
    def __init__(self, file_path):
        """
        Initialize the SED with the file path.

        Parameters:
        file_path (str): Path to the data file.
        """
        self.file_path = file_path
        self.data = None
        self.distance = None  # To store the distance as an astropy Quantity
        self.columns = []
        self.column_units = []

    def read_data(self):
        """
        Reads the metadata and numerical data from the file, and assigns units to the columns.
        """
        try:
            with open(self.file_path, "r") as file:
                for line in file:
                    if line.startswith("#"):
                        self._extract_metadata(line)

            # Read the numerical data
            data = np.loadtxt(self.file_path)

            # Create an astropy Table and assign units
            self.data = QTable(data, names=self.columns, units=self.column_units)

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def _extract_metadata(self, line):
        """
        Extracts metadata (distance, column names, and units) from a comment line.

        Parameters:
        line (str): The comment line containing metadata.
        """
        if "distance" in line.lower() or "redshift" in line.lower():
            if "distance" in line.lower():
                # Extract the distance with units from format: "distance 10 Mpc" or "luminosity distance 10 Mpc"
                parts = line.split("distance")
                if len(parts) > 1:
                    distance_part = parts[1].strip()
                    # Remove any trailing characters like commas
                    if "," in distance_part:
                        distance_part = distance_part.split(",")[0]
                    dist_parts = distance_part.split()
                    if len(dist_parts) >= 2:
                        value = float(dist_parts[0])
                        unit = dist_parts[1]
                        self.distance = value * u.Unit(unit)
            if "redshift" in line.lower():
                # Extract redshift and convert to distance using astropy cosmology
                parts = line.split("redshift")
                if len(parts) > 1:
                    redshift_part = parts[1].strip()
                    if "," in redshift_part:
                        redshift_part = redshift_part.split(",")[0]
                    self.redshift = float(redshift_part)
            else:
                self.redshift = 0 # Default to zero if no redshift information is provided
        elif "column" in line.lower():
            # Extract column names and their units from format: "column 1: wavelength; lambda (micron)"
            parts = line.split(":")
            if len(parts) >= 2:
                column_info = ":".join(parts[1:]).strip()  # Rejoin in case there are multiple colons
                if ";" in column_info:
                    description, variable_unit = column_info.split(";", 1)
                    description = description.strip()
                    variable_unit = variable_unit.strip()
                    
                    # Extract unit from format "variable_name (unit)"
                    if "(" in variable_unit and ")" in variable_unit:
                        start_paren = variable_unit.rfind("(")
                        end_paren = variable_unit.rfind(")")
                        unit_str = variable_unit[start_paren+1:end_paren]
                        variable_name = variable_unit[:start_paren].strip()
                    else:
                        # Fallback if no parentheses found
                        unit_str = "dimensionless"
                        variable_name = variable_unit
                    
                    self.columns.append(description)
                    try:
                        self.column_units.append(u.Unit(unit_str))
                    except ValueError:
                        # Handle unknown units by treating as dimensionless
                        print(f"Warning: Unknown unit '{unit_str}', treating as dimensionless")
                        self.column_units.append(u.dimensionless_unscaled)
    
    @property
    def wavelength(self):
        """Access the wavelength column."""
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        return self.data[self.columns[0]]  # First column is wavelength
    
    @property
    def rest_wavelength(self):
        """Access the rest wavelength column, if redshift information is available."""
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        return self.wavelength / (1 + self.redshift)
    
    @property
    def total_flux(self):
        """Access the total flux column."""
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        return self.data[self.columns[1]]  # Second column is total flux

    @property
    def transparent_flux(self):
        """Access the transparent flux column."""
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        return self.data[self.columns[2]]  # Third column is transparent flux

    def get_column(self, column_index, spectral_density_units='frequency', convert_to_luminosity=False, restframe=False):
        """
        Get a specific column by index.
        
        Parameters:
        column_index (int): Index of the column to retrieve (0-based)
        spectral_density_units (str): What spectral density units to convert to if the column is a flux column. Options are 'frequency','wavelength', or 'neutral'. Only applicable for flux columns.
        convert_to_luminosity (bool): Whether to convert the column from flux to luminosity units using the distance. Only applicable for flux columns.
        restframe (bool): Whether to return the column in the rest frame. Only applicable for wavelength columns.
        
        Returns:
        astropy.table.Column: The requested column with units
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        if column_index >= len(self.columns):
            raise IndexError(f"Column index {column_index} out of range. Available columns: 0-{len(self.columns)-1}")
        
        column_data = self.data[self.columns[column_index]]
        if 'flux' in self.columns[column_index].lower():
            column_data = self._convert_spectral_density_units(column_data, self.wavelength, convert_to_luminosity=convert_to_luminosity, spectral_density_units=spectral_density_units)
        if restframe and 'wavelength' in self.columns[column_index].lower():
            column_data = column_data / (1 + self.redshift)

        return column_data


    def get_column_by_name(self, column_name, spectral_density_units='frequency', convert_to_luminosity=False, restframe=False):
        """
        Get a column by its description name.
        
        Parameters:
        column_name (str): Name/description of the column
        spectral_density_units (str): What spectral density units to convert to if the column is a flux column. Options are 'frequency','wavelength', or 'neutral'. Only applicable for flux columns.
        convert_to_luminosity (bool): Whether to convert the column from flux to luminosity units using the distance. Only applicable for flux columns.
        restframe (bool): Whether to return the column in the rest frame. Only applicable for wavelength columns.

        Returns:
        astropy.table.Column: The requested column with units
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read_data() first.")
        for column in self.columns:
                if column_name.lower() in column.lower():
                    column_name = column  # Use the full column name if a match is found
                    break
        if column_name not in self.columns:
            raise KeyError(f"Column '{column_name}' not found. Available columns: {self.columns}")
        
        column_data = self.data[column_name]
        if 'flux' in column_name.lower():
            column_data=self._convert_spectral_density_units(column_data, self.wavelength, convert_to_luminosity=convert_to_luminosity, spectral_density_units=spectral_density_units)
        if restframe and 'wavelength' in column_name.lower():
            column_data = column_data / (1 + self.redshift)
        return column_data


    def _convert_spectral_density_units(self, column_data, wavelengths, convert_to_luminosity=False, spectral_density_units='frequency'):
        """
        Convert the units of a column to the specified spectral density units.
        
        Parameters:
        column_data (astropy.table.Column): The column to convert
        wavelengths (astropy.table.Column): The wavelength column to use for the conversion
        convert_to_luminosity (bool): Whether to convert the column from flux to luminosity units using the distance. Only applicable for flux columns.
        spectral_density_units (str): What spectral density units to convert to if the column is a flux column. Options are 'frequency','wavelength', or 'neutral'. Only applicable for flux columns.

        Returns:
        astropy.table.Column: The converted column with units
        """
        if convert_to_luminosity:
            if self.distance is None:
                raise ValueError("Distance information not found. Cannot convert to luminosity.")
            
            flux_to_luminosity_factor = 4 * np.pi * self.distance**2
            neutral_factor = 1
            if spectral_density_units == 'frequency':
                lum_units = u.L_sun / u.Hz
                flux_to_luminosity_factor *= 1/(1 + self.redshift) 
            elif spectral_density_units == 'wavelength':
                lum_units = u.L_sun / u.micron
                flux_to_luminosity_factor *= (1 + self.redshift) 
            elif spectral_density_units == 'neutral':
                lum_units = u.L_sun / u.micron
                neutral_factor *= wavelengths.to('micron')
            column_data = (column_data * flux_to_luminosity_factor).to(lum_units,equivalencies=u.spectral_density(wavelengths))*neutral_factor
        else:
            unit_conversion_factor = 1
            if spectral_density_units == 'frequency':
                flux_units = u.Jy
            elif spectral_density_units == 'wavelength':
                flux_units = u.erg / u.cm**2 / u.s / u.micron
            elif spectral_density_units == 'neutral':
                flux_units = u.erg / u.cm**2 / u.s / u.micron
                unit_conversion_factor *= wavelengths.to('micron')
            column_data = (column_data).to(flux_units,equivalencies=u.spectral_density(wavelengths))*unit_conversion_factor

        return column_data
    
    def __repr__(self):
        """String representation of the SED object."""
        if self.data is None:
            return f"SED(file_path='{self.file_path}', data_loaded=False)"
        
        info = f"SED(file_path='{self.file_path}'\n"
        info += f"  Distance: {self.distance}\n"
        info += f"  {len(self.data)} data points\n"
        info += f"  Columns: {len(self.columns)}\n"
        for i, (col, unit) in enumerate(zip(self.columns, self.column_units)):
            info += f"    {i+1}. {col} ({unit})\n"
        info += ")"
        return info

    def calculate_MUV(self, UV_wavelength=0.15*u.micron, no_dust=False):
        """
        Calculate the absolute ultraviolet magnitude (MUV) at a specific wavelength.

        Parameters:
        UV_wavelength (astropy.units.Quantity): The wavelength at which to calculate the UV magnitude (e.g., 0.15*u.micron).
        no_dust (bool): If True, will use the transparent flux instead of the total flux to calculate MUV, effectively giving the intrinsic UV magnitude without dust attenuation.

        Returns:
        astropy.units.Quantity: The ultraviolet magnitude at the specified wavelength.
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")

        # Get the L_nu at the specified UV wavelength
        wavelength = self.rest_wavelength
        if no_dust:
            total_lum_nu = self.get_column_by_name('transparent flux', 
                                                  spectral_density_units='frequency', 
                                                  convert_to_luminosity=True)
        else:
            total_lum_nu = self.get_column_by_name('total flux', 
                                                  spectral_density_units='frequency', 
                                                  convert_to_luminosity=True)
        mUV_0 = -32.36 # Used for AB magnitude system. Calculated as -2.5*log10(3631 Jy in Lsol/Hz at 10 pc)
        L_nu = np.interp(UV_wavelength.to(wavelength.unit).value, wavelength.value, total_lum_nu.value) * total_lum_nu.unit
        MUV = -2.5 * np.log10(L_nu.to(u.L_sun / u.Hz, equivalencies=u.spectral_density(wavelength)).value) + mUV_0

        return MUV


    def calculate_LIR(self, wavelength_range=(8*u.micron, 1000*u.micron)):
        """
        Calculate the total infrared luminosity (LIR) by integrating the SED over a specified wavelength range.

        Parameters:
        wavelength_range (tuple): A tuple specifying the lower and upper bounds of the wavelength range for integration, with astropy units (e.g., (8*u.micron, 1000*u.micron)).

        Returns:
        astropy.units.Quantity: The total infrared luminosity in units of solar luminosities (L_sun).
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")
        
        # Get wavelength and total flux columns

        wavelength = self.rest_wavelength
        total_flux = self.get_column_by_name('total flux', 
                                             spectral_density_units='wavelength', 
                                             convert_to_luminosity=True)  # Convert to L_lambda units
        
        # Create a mask for the specified wavelength range
        mask = (wavelength >= wavelength_range[0]) & (wavelength <= wavelength_range[1])
        
        # Integrate the flux over the specified wavelength range using trapezoidal rule
        LIR = integrate.trapezoid(total_flux[mask], x=wavelength[mask])
        
        return LIR.to(u.L_sun)
    
    def calculate_LUV(self, UV_wavelength=0.16*u.micron):
        """
        Calculate the ultraviolet luminosity (LUV) at a specific wavelength.

        Parameters:
        UV_wavelength (astropy.units.Quantity): The wavelength at which to calculate the UV luminosity (e.g., 0.16*u.micron).

        Returns:
        astropy.units.Quantity: The ultraviolet luminosity at the specified wavelength in units of solar luminosities (L_sun).
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")
        
        # Get wavelength and total flux columns
        wavelength = self.rest_wavelength
        total_flux = self.get_column_by_name('total flux', 
                                             spectral_density_units='wavelength', 
                                             convert_to_luminosity=True)  # Convert to L_lambda units
        
        # Interpolate the total flux at the specified UV wavelength
        LUV = UV_wavelength.to(wavelength.unit) * (np.interp(UV_wavelength.to(wavelength.unit).value, wavelength.value, total_flux.value) * total_flux.unit)
        
        return LUV.to(u.L_sun)


    def calculate_attenuation(self, only_direct_stellar=False):
        """
        Calculate the attenuation (A_lambda) across all wavelengths by comparing the total flux to the transparent flux.

        Parameters:
        only_direct_stellar (bool): Whether to consider only the direct stellar flux when calculating attenuation.
        If True, only the direct stellar flux will be used as the total. If False, the total flux 
        (direct+scattered stellar + dust emission) will be used. For very low Av systems scattering+emission 
        can lead to negative attenuation values.

        Returns:
        astropy.table.Column: The attenuation A_lambda in magnitudes for each wavelength.
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")
        
        # Get wavelength, total flux, and transparent flux columns
        # Get total flux (direct+scattered stellar + dust emission) or only direct stellar flux
        if not only_direct_stellar:
            total_flux = self.get_column_by_name('total flux', 
                                                spectral_density_units='wavelength', 
                                                convert_to_luminosity=False)  # Keep in flux units
        else:
            total_flux = self.get_column_by_name('direct primary flux', 
                                                spectral_density_units='wavelength', 
                                                convert_to_luminosity=False)  # Keep in flux units
            
        transparent_flux = self.get_column_by_name('transparent flux', 
                                                  spectral_density_units='wavelength', 
                                                  convert_to_luminosity=False)  # Keep in flux units
        
        # Calculate A_lambda using the formula: A_lambda = 2.5 * log10(transparent_flux / total_flux)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(total_flux == 0, np.nan, transparent_flux / total_flux)
            A_lambda = 2.5 * np.log10(ratio)
        
        return A_lambda

    def calculate_beta_UV(self, column_name='total'):
        """
        Calculate the UV spectral slope (beta) with a simple linear fit to the SED.

        Parameters:
        UV_wavelength_range (tuple): A tuple specifying the lower and upper bounds of the UV wavelength range for fitting, with astropy units (e.g., (0.13*u.micron, 0.3*u.micron)).

        Returns:
        float: The calculated UV spectral slope beta.
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")
        
        # Get wavelength and total flux columns
        wavelength = self.rest_wavelength
        total_flux = self.get_column_by_name(column_name, 
                                             spectral_density_units='wavelength', 
                                             convert_to_luminosity=False)  # Keep in flux units
        
        wavelength_1230A = 0.123*u.micron
        wavelength_3200A = 0.32*u.micron
        flux_1230A = np.interp(wavelength_1230A.to(wavelength.unit).value, wavelength.value, total_flux.value) * total_flux.unit
        flux_3200A = np.interp(wavelength_3200A.to(wavelength.unit).value, wavelength.value, total_flux.value) * total_flux.unit

        beta_UV = np.log10(flux_1230A/flux_3200A) / np.log10(wavelength_1230A/wavelength_3200A)

        return beta_UV



    def calculate_Td_peak(self):
        """
        Estimate the dust temperature from the peak wavelength of the far-infrared portion of the SED.
        
        Returns:
        astropy.units.Quantity: The estimated dust temperature in Kelvin.
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("SED data not loaded. Call sed.read_data() first.")
        
        # Get wavelength and total flux columns
        wavelength = self.rest_wavelength
        total_flux = self.get_column_by_name('total', 
                                             spectral_density_units='frequency', 
                                             convert_to_luminosity=False)  # Keep in flux units
        
        # Select the far-infrared portion of the SED (e.g., > 20 microns)
        FIR_mask = wavelength > 20*u.micron
        wavelength_FIR = wavelength[FIR_mask]
        flux_FIR = total_flux[FIR_mask]

        # Estimate T_dust from the peak wavelength of the FIR SED using Wien's displacement law
        peak_index = np.argmax(flux_FIR)
        peak_wavelength = wavelength_FIR[peak_index]
        T_dust_estimated = const.b_wien / peak_wavelength.to(u.m)

        return T_dust_estimated
