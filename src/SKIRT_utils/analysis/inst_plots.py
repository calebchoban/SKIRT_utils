import numpy as np
from astropy.visualization import make_lupton_rgb, make_rgb, LogStretch, LuptonAsinhStretch, ManualInterval

from crc_scripts.figure import Figure, Projection


def make_log_lupton_RGB_image(ifu, rgb_filters, psf=None, max_percentile=100, max_frac=1, min_percentile=0, min_frac=0, stretch=1000, label=None, output_name=None, **kwargs):
    """
    Generate a log-scaled RGB image using Lupton's RGB algorithm.

    Parameters
    ----------
    ifu : IFU
        IFU instrument object to extract filter images from.
    rgb_filters : list
        List of three filter names to use for the red, green, and blue channels.
    psf : Telescope_PSF or list, optional
        PSF(s) to convolve each channel with. If a list, each element is applied to the corresponding channel.
    max_percentile : int or list, optional
        Maximum percentile value(s) for scaling each channel. Overrides max_frac.
    max_frac : float or list, optional
        Fraction of maximum pixel brightness for max limit of scaling. Overridden by max_percentile.
    min_percentile : int or list, optional
        Minimum percentile value(s) for scaling each channel. Overrides min_frac.
    min_frac : float or list, optional
        Fraction of maximum pixel brightness for min limit of scaling. Overridden by min_percentile.
    stretch : int, optional
        Stretch factor for the log scaling.
    label : str, optional
        Label for the image.
    output_name : str, optional
        Output file name for the image. If None, the image will not be saved.
    """
    if psf is not None:
        if isinstance(psf, list):
            r_frame = ifu.get_filter_image(rgb_filters[0], psf=psf[0], **kwargs)
            g_frame = ifu.get_filter_image(rgb_filters[1], psf=psf[1], **kwargs)
            b_frame = ifu.get_filter_image(rgb_filters[2], psf=psf[2], **kwargs)
    else:
        r_frame = ifu.get_filter_image(rgb_filters[0], psf=psf, **kwargs)
        g_frame = ifu.get_filter_image(rgb_filters[1], psf=psf, **kwargs)
        b_frame = ifu.get_filter_image(rgb_filters[2], psf=psf, **kwargs)

    if max_percentile is not None:
        if not isinstance(max_percentile, list):
            max_percentile = [max_percentile]*3
        maximums = [np.percentile(r_frame[r_frame>0].value, max_percentile[0]), np.percentile(g_frame[g_frame>0].value, max_percentile[1]), np.percentile(b_frame[b_frame>0].value, max_percentile[2])]
    else:
        if not isinstance(max_frac, list):
            max_frac = [max_frac]*3
        maximums = [np.max(r_frame.value)*max_frac[0], np.max(g_frame.value)*max_frac[1], np.max(b_frame.value)*max_frac[2]]
    if min_percentile is not None:
        if not isinstance(min_percentile, list):
            min_percentile = [min_percentile]*3
        minimums = [np.percentile(r_frame[r_frame>0].value, min_percentile[0]), np.percentile(g_frame[g_frame>0].value, min_percentile[1]), np.percentile(b_frame[b_frame>0].value, min_percentile[2])]
    else:
        if not isinstance(min_frac, list):
            min_frac = [min_frac]*3
        minimums = [np.max(r_frame.value)*min_frac[0], np.max(g_frame.value)*min_frac[1], np.max(b_frame.value)*min_frac[2]]

    intervals = [ManualInterval(vmin=minimums[0], vmax=maximums[0]), ManualInterval(vmin=minimums[1], vmax=maximums[1]), ManualInterval(vmin=minimums[2], vmax=maximums[2])]

    if ifu.verbose:
        print("RGB maximum values", maximums)
        print("RGB minimum values", minimums)

    RGB_image = make_lupton_rgb(r_frame, g_frame, b_frame, interval=intervals,
               stretch_object=LogStretch(a=stretch))

    img = Projection(1)
    img.set_image_axis(0)
    img.plot_image(0, RGB_image, fov_kpc=ifu.fov_physical.value, fov_arcsec=ifu.fov_angle.value, label=label)

    if output_name is not None:
        img.save(output_name)


def make_lupton_RGB_image(ifu, rgb_filters, psf=None, stretch=0.5, Q=8, min_percentile=0, max_percentile=100, label=None, output_name=None, **kwargs):
    """
    Generate an RGB image using Lupton's RGB algorithm which keeps the true color of each pixel.

    Parameters
    ----------
    ifu : IFU
        IFU instrument object to extract filter images from.
    rgb_filters : list
        List of three filter names to use for the red, green, and blue channels.
    psf : Telescope_PSF or list, optional
        PSF(s) to convolve each channel with. If a list, each element is applied to the corresponding channel.
    stretch : float, optional
        Stretch factor for arcsinh. Sets where you want the color to be linear.
    Q : float, optional
        Q parameter for arcsinh. Sets how bright the brightest pixels are.
    min_percentile : int or list, optional
        Minimum percentile value(s) for scaling.
    max_percentile : int or list, optional
        Maximum percentile value(s) for scaling.
    label : str, optional
        Label for the image.
    output_name : str, optional
        Output file name for the image. If None, the image will not be saved.
    """
    if psf is not None:
        if isinstance(psf, list):
            r_frame = ifu.get_filter_image(rgb_filters[0], psf=psf[0], **kwargs)
            g_frame = ifu.get_filter_image(rgb_filters[1], psf=psf[1], **kwargs)
            b_frame = ifu.get_filter_image(rgb_filters[2], psf=psf[2], **kwargs)
    else:
        r_frame = ifu.get_filter_image(rgb_filters[0], psf=psf, **kwargs)
        g_frame = ifu.get_filter_image(rgb_filters[1], psf=psf, **kwargs)
        b_frame = ifu.get_filter_image(rgb_filters[2], psf=psf, **kwargs)

    if not isinstance(max_percentile, list):
        max_percentile = [max_percentile]*3
    maximums = [np.percentile(r_frame[r_frame>0].value, max_percentile[0]), np.percentile(g_frame[g_frame>0].value, max_percentile[1]), np.percentile(b_frame[b_frame>0].value, max_percentile[2])]
    if not isinstance(min_percentile, list):
        min_percentile = [min_percentile]*3
    minimums = [np.percentile(r_frame[r_frame>0].value, min_percentile[0]), np.percentile(g_frame[g_frame>0].value, min_percentile[1]), np.percentile(b_frame[b_frame>0].value, min_percentile[2])]

    intervals = [ManualInterval(vmin=minimums[0], vmax=maximums[0]), ManualInterval(vmin=minimums[1], vmax=maximums[1]), ManualInterval(vmin=minimums[2], vmax=maximums[2])]

    if ifu.verbose:
        print("RGB maximum values", maximums)
        print("RGB minimum values", minimums)

    RGB_image = make_lupton_rgb(
        r_frame, g_frame, b_frame,
        stretch=stretch,
        Q=Q,
        interval=intervals,
    )

    img = Projection(1)
    img.set_image_axis(0)
    img.plot_image(0, RGB_image, fov_kpc=ifu.fov_physical.value, fov_arcsec=ifu.fov_angle.value, label=label)

    if output_name is not None:
        img.save(output_name)


def make_lupton_RGB_image_grid(ifu, rgb_filters, psf=None, Q_lims=[1, 10], stretch_lims=[0.1, 1], bins=4, output_name=None, **kwargs):
    """
    Create a grid of RGB images across Q and stretch parameter ranges. Useful for
    determining what Q and stretch values to use.

    Parameters
    ----------
    ifu : IFU
        IFU instrument object to extract filter images from.
    rgb_filters : list
        List of three filter names to use for the red, green, and blue channels.
    psf : Telescope_PSF or list, optional
        PSF(s) to convolve each channel with. If a list, each element is applied to the corresponding channel.
    Q_lims : list, optional
        The range [min, max] of the Q parameter, binned in linear space.
    stretch_lims : list, optional
        The range [min, max] of the stretch parameter, binned in log space.
    bins : int, optional
        Number of bins between Q and stretch limits.
    output_name : str, optional
        Name of output file for the image. If None, the image will not be saved.
    """
    if psf is not None:
        if isinstance(psf, list):
            r_frame = ifu.get_filter_image(rgb_filters[0], psf=psf[0], **kwargs)
            g_frame = ifu.get_filter_image(rgb_filters[1], psf=psf[1], **kwargs)
            b_frame = ifu.get_filter_image(rgb_filters[2], psf=psf[2], **kwargs)
    else:
        r_frame = ifu.get_filter_image(rgb_filters[0], psf=psf, **kwargs)
        g_frame = ifu.get_filter_image(rgb_filters[1], psf=psf, **kwargs)
        b_frame = ifu.get_filter_image(rgb_filters[2], psf=psf, **kwargs)

    stretchs = np.logspace(np.log10(stretch_lims[0]), np.log10(stretch_lims[1]), bins)
    Qs = np.linspace(Q_lims[0], Q_lims[1], bins)

    N = bins * bins
    nrows = bins
    img = Projection(N, nrows=nrows)
    for i in range(N):
        img.set_image_axis(i)

    for i, Q in enumerate(Qs):
        for j, stretch in enumerate(stretchs):
            RGB_image = make_lupton_rgb(
                r_frame, g_frame, b_frame,
                stretch=stretch,
                Q=Q,
            )
            label = 'Q=%1.1f, stretch=%1.1e' % (Q, stretch)
            img.plot_image(i*bins+j, RGB_image, label=label)

    if output_name is not None:
        img.save(output_name)


def make_RGB_image(ifu, rgb_filters, psf=None, stretch=0.5, Q=8, min_percentile=0, max_percentile=100, label=None, output_name=None, trim_monocolor=True, scale_bar='both', **kwargs):
    """
    Generate an RGB image. Unlike the Lupton scheme, this does not preserve true pixel
    colors above the maximum brightness, but can emphasize certain colors.

    Parameters
    ----------
    ifu : IFU
        IFU instrument object to extract filter images from.
    rgb_filters : list
        List of three filter names to use for the red, green, and blue channels.
    psf : Telescope_PSF or list, optional
        PSF(s) to convolve each channel with. If a list, each element is applied to the corresponding channel.
    stretch : float, optional
        Stretch factor for arcsinh.
    Q : float, optional
        Q parameter for arcsinh.
    min_percentile : int or list, optional
        Minimum percentile value(s) for scaling.
    max_percentile : int or list, optional
        Maximum percentile value(s) for scaling.
    label : str, optional
        Label for the image.
    output_name : str, optional
        Output file name for the image. If None, the image will not be saved.
    trim_monocolor : bool, optional
        If True, pixels that are monochromatic (only one non-zero channel) are set to zero.
    scale_bar : str, optional
        Set to 'physical', 'angle', or 'both' for a kpc, arcminute, or both scale bars.
    """
    if psf is not None:
        if isinstance(psf, list):
            r_frame = ifu.get_filter_image(rgb_filters[0], psf=psf[0], **kwargs)
            g_frame = ifu.get_filter_image(rgb_filters[1], psf=psf[1], **kwargs)
            b_frame = ifu.get_filter_image(rgb_filters[2], psf=psf[2], **kwargs)
    else:
        r_frame = ifu.get_filter_image(rgb_filters[0], psf=psf, **kwargs)
        g_frame = ifu.get_filter_image(rgb_filters[1], psf=psf, **kwargs)
        b_frame = ifu.get_filter_image(rgb_filters[2], psf=psf, **kwargs)

    monochromatic_mask = (
        ((r_frame > 0) & (g_frame == 0) & (b_frame == 0)) |
        ((r_frame == 0) & (g_frame > 0) & (b_frame == 0)) |
        ((r_frame == 0) & (g_frame == 0) & (b_frame > 0))
    )

    if trim_monocolor:
        r_frame[monochromatic_mask] = 0
        g_frame[monochromatic_mask] = 0
        b_frame[monochromatic_mask] = 0

    if not isinstance(max_percentile, list):
        max_percentile = [max_percentile]*3
    maximums = [np.percentile(r_frame[r_frame>0].value, max_percentile[0]), np.percentile(g_frame[g_frame>0].value, max_percentile[1]), np.percentile(b_frame[b_frame>0].value, max_percentile[2])]
    if not isinstance(min_percentile, list):
        min_percentile = [min_percentile]*3
    minimums = [np.percentile(r_frame[r_frame>0].value, min_percentile[0]), np.percentile(g_frame[g_frame>0].value, min_percentile[1]), np.percentile(b_frame[b_frame>0].value, min_percentile[2])]

    intervals = [ManualInterval(vmin=minimums[0], vmax=maximums[0]), ManualInterval(vmin=minimums[1], vmax=maximums[1]), ManualInterval(vmin=minimums[2], vmax=maximums[2])]

    if ifu.verbose:
        print("RGB maximum values", maximums)
        print("RGB minimum values", minimums)

    stretch_obj = LuptonAsinhStretch(stretch=stretch, Q=Q)

    RGB_image = make_rgb(
        r_frame, g_frame, b_frame,
        stretch=stretch_obj,
        interval=intervals,
    )

    img = Projection(1)
    img.set_image_axis(0)

    fov_kpc = None; fov_arcsec = None
    if scale_bar == 'both':
        fov_kpc = ifu.fov_physical.value; fov_arcsec = ifu.fov_angle.value
    elif scale_bar == 'physical':
        fov_kpc = ifu.fov_physical.value
    elif scale_bar == 'angle':
        fov_arcsec = ifu.fov_angle.value
    img.plot_image(0, RGB_image, fov_kpc=fov_kpc, fov_arcsec=fov_arcsec, label=label)

    if output_name is not None:
        img.save(output_name)


def make_rgb_hist(ifu, rgb_filters, psf=None, output_name=None, **kwargs):
    """
    Create a histogram of pixel brightness for each channel in an RGB image.

    Parameters
    ----------
    ifu : IFU
        IFU instrument object to extract filter images from.
    rgb_filters : list
        List of three filter names to use for the red, green, and blue channels.
    psf : Telescope_PSF or list, optional
        PSF(s) to convolve each channel with. If a list, each element is applied to the corresponding channel.
    output_name : str, optional
        Name of output file for the histogram. If None, the histogram will not be saved.
    """
    if psf is not None:
        if isinstance(psf, list):
            r_frame = ifu.get_filter_image(rgb_filters[0], psf=psf[0], **kwargs)
            g_frame = ifu.get_filter_image(rgb_filters[1], psf=psf[1], **kwargs)
            b_frame = ifu.get_filter_image(rgb_filters[2], psf=psf[2], **kwargs)
    else:
        r_frame = ifu.get_filter_image(rgb_filters[0], psf=psf, **kwargs)
        g_frame = ifu.get_filter_image(rgb_filters[1], psf=psf, **kwargs)
        b_frame = ifu.get_filter_image(rgb_filters[2], psf=psf, **kwargs)
    rgb_image = np.array([r_frame, g_frame, b_frame])

    fig = Figure(1)
    labels = ['r', 'g', 'b']

    x_lim = [np.min(rgb_image[rgb_image>0]), np.max(rgb_image)]
    fig.set_axis(0, 'Brightness', 'CDF', y_lim=[0.001, 1], y_label='CDF', y_scale='linear', x_lim=x_lim, x_label=f'Brightness ({r_frame.unit:latex})', x_scale='log')
    for i, image in enumerate(rgb_image):
        data = image.flatten()
        data = data[data>x_lim[0]]
        fig.plot_1Dhistogram(0, data, bin_lims=None, bin_nums=100, scale='log', label=labels[i], density=True, color=labels[i], cumulative=True)
    fig.set_all_legends()
    if output_name is not None:
        fig.save(output_name)
