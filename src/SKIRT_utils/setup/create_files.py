
import os
import numpy as np
from string import Formatter
from astropy.cosmology import LambdaCDM
import astropy.units as u


# This is a workaround to handle missing keys in the dictionary and avoid
# KeyError exceptions. It returns the key wrapped in curly braces if the key
# is missing.
class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

def format_ski_file(
    snapnum: int,
    default_pixel_ang_scale: float|None = None,
    default_pixel_phy_scale: float|None = None,
    skirt_sim_mode: str='dust_emission',
    dust_emission_type: str='equilibrium',
    medium_grid: str='octtree',
    output_path: str = '.',
    cosmological: bool = True,
    cosmo_params: str = 'PLANCK',
    distance: float = 10.0,
    FIRE_ver: int = 0,
    num_packets: float = 1e7,
    num_wavelengths: int = 200,
    min_wavelength: float = 0.09,
    max_wavelength: float = 2000.0,
    min_source_wavelength: float|None = None,
    max_source_wavelength: float|None = None,
    min_rad_field_wavelength: float|None = None,
    max_rad_field_wavelength: float|None = None,
    num_rad_field_wavelengths: int|None = None,
    num_dust_emission_wavelengths: int|None = None,
    half_fov_kpc: float = 18.0,
    inst_half_fov_kpc: float|None = None,
    max_temperature: float = 0.0,
    mass_fraction: float = 1.0,
    max_dust_frac: float = 1e-7,
    min_tree_level: int = 3,
    max_tree_level: int = 20,
    set_template_file: str = '',
    snapshot_scale_factors: str = '',
    instrument_pixel_ang_scales: dict = {},
    instrument_pixel_phy_scales: dict = {},
    add_template_placeholders: dict = {},
):
    """
    Prepare SKIRT input files for a given snapshot of a FIRE simulation.

    Parameters
    ----------
    snapnum : int
        Snapshot number to process. Used for determining redshift and scale factors for cosmological simulations.
    default_pixel_ang_scale : float, optional
        Pixel scale in arcseconds per pixel for the default instrument images.Must use either default_pixel_ang_scale or default_pixel_phy_scale, not both.
    default_pixel_phy_scale : float, optional
        Pixel scale in kpc per pixel for the default instrument images. Must use either default_pixel_ang_scale or default_pixel_phy_scale, not both.
    skirt_sim_mode : str, optional
        Simulation mode for SKIRT (e.g., "emission" or "extinction" or "monochromatic").
    dust_emission_type : str, optional
        Dust emission mode (e.g., "equilibrium" or "stochastic").
    medium_grid : str, optional
        Dust medium grid used in SKIRT (e.g., "octtree" or "voronoi"). voronoi is only recommended for   low resolution instruments or SEDs.
    output_path : str, optional
        Path to the output file for the formatted ski file.
    cosmological : bool, optional
        Set whether you want SKIRT simulation to be cosmological. 
    cosmo_params : str, optional
        Cosmological parameters used by the FIRE simulation. Options are PLANCK or AGORA.
    distance : float, optional
        Distance to the simulation object in Mpc for rest-frame instruments. Note for non-cosmological runs all instruments are in the rest frame.
    FIRE_ver : int, optional
        Version of FIRE simulation (e.g., 2, 3) used to determine redshift of snapshot number.
    num_packets : float, optional
        Number of photon packets for SKIRT simulation.
    num_wavelengths : int, optional
        Number of wavelengths for SKIRT simulation. Used for source wavelength grid, radiation field grid, and SED instruments by default, but these can be set independently if desired.
    min_wavelength : float, optional
        Minimum wavelength in microns for SKIRT simulation. Used by source wavelength grid, radiation field grid, and SED instruments by default, but these can be set independently if desired.
    max_wavelength : float, optional
        Maximum wavelength in microns for SKIRT simulation. Used by source wavelength grid, radiation field grid, and SED instruments by default, but these can be set independently if desired.
    min_source_wavelength : float, optional
        Minimum wavelength in microns for the source wavelength grid. Default is set to min_wavelength for consistency with the global grid, but can be set to a different value if desired.
    max_source_wavelength : float, optional
        Maximum wavelength in microns for the source wavelength grid. Default is set to max_wavelength for consistency with the global grid, but can be set to a different value if desired.
    num_dust_emission_wavelengths : int, optional
        Number of wavelengths for the dust emission grid. Default is set to 100. Note wavelengths are subbinned by 2x in the MIR to capture PAH features.
    half_fov_kpc : float, optional
        Half field of view in kpc used for spatial grid and instruments if inst_half_fov_kpc not given. Make sure this matches the box size of the data extracted from the simulation since SKIRT will use all sources provided to it but only dust cells contained in the spatial grid.
    inst_half_fov_kpc : float, optional
        Half field of view in kpc used for instruments. If not given, will default to half_fov_kpc.
    max_temperature : float, optional
        Maximum temperature for the medium elements in the simulation. Unless
        using a post-processed dust prescription (e.g. with a constant dust-to-metals
        ratio), this should always be set to 0.0 K.
    mass_fraction : float, optional
        Fraction of the total medium mass to include in the simulation. This
        should be set to 1.0 unless using a custom dust prescription (e.g. with a
        constant dust-to-metals ratio).
    max_dust_frac : float, optional
        When using medium_grid="octtree", this sets the maximum fraction of the
        total dust mass allowed in any grid cell. Smaller values lead to finer
        gridding and higher memory usage.
    min_tree_level : int, optional
        When using medium_grid="octtree", this sets the minimum level of the octree grid (i.e. the maximum grid size). If you want finer detail for diffuse regions set this to a higher value.
    max_tree_level : int, optional
        When using medium_grid="octtree", this sets the maximum level of the octree grid (i.e. the minimum grid size). If you find many cells in the maximum level for a given max_dust_frac, you should increase this value.
    set_template_file : str, optional
        Path to a ski template file to use. Will override the auto-selected template.
    snapshot_scale_factors : str, optional
        Path to text file containing scale factors for each snapshot of the simulation.
        If not given, will default to FIRE-2/3 values depending on FIRE_ver.
    instrument_pixel_ang_scales : dict, optional
        Dictionary mapping observer-frame instrument name prefixes to their pixel scales in arcseconds
        per pixel. For each entry with key ``Name`` and value ``scale``, the placeholders
        ``{NameNumPixelsX}`` and ``{NameNumPixelsY}`` are computed and added to the template.
        Example: ``{"JWST": 0.031, "ALMA": 0.1}`` populates ``{JWSTNumPixelsX/Y}`` and
        ``{ALMANumPixelsX/Y}``.
    instrument_pixel_phy_scales : dict, optional
        Dictionary mapping observer-frame instrument name prefixes to their pixel scales in kpc
        per pixel. For each entry with key ``Name`` and value ``scale``, the placeholders
        ``{NameNumPixelsX}`` and ``{NameNumPixelsY}`` are computed and added to the template.
        Example: ``{"JWST": 0.5, "ALMA": 1.0}`` populates ``{JWSTNumPixelsX/Y}`` and
        ``{ALMANumPixelsX/Y}``.
    add_template_placeholders : dict
        Dictionary of additional placeholder keys and corresponding values to add to the template beyond the default templates. Ensure keys should match the placeholder names in the template file.
    """

    # Make output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # 1. Determine the template file we need (or the one provided by the user) and read it in
    
    # Determines name of template file which matches requested SKIRT mode and medium grid
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_dirc = os.path.join(current_dir, "ski_templates/")
    # user defined template
    if set_template_file != '':
        # Check if template files exists at given path or in template directory
        if os.path.exists(set_template_file):
            ski_template_filepath = set_template_file
        elif os.path.exists(os.path.join(template_dirc, set_template_file)):
            ski_template_filepath = os.path.join(template_dirc, set_template_file)
        else:
            raise ValueError(f"set_template_file given as '{set_template_file}' but no file found at this path or in the template directory.")
    # auto-select template based on SKIRT mode and medium grid. note this there are only a handful of templates
    else:
        template_dirc = os.path.join(template_dirc, "default/")
        template_files = [file for file in os.listdir(template_dirc) if os.path.isfile(os.path.join(template_dirc, file))]
        ski_template_filepath=False
        for file in template_files:
            if medium_grid not in file: continue
            if skirt_sim_mode=='emission' and 'emission' not in file: continue
            elif skirt_sim_mode=='extinction' and 'extinction' not in file: continue
            elif skirt_sim_mode=='monochromatic' and 'monochromatic' not in file: continue
            ski_template_filepath = os.path.join(template_dirc, file)
        if not ski_template_filepath:
            raise ValueError("No template matches requested SKIRT mode")
        print(f"Using template file {ski_template_filepath} for SKIRT input formatting...")

    # Read the SKIRT input file template
    with open(ski_template_filepath, "r") as f:
        ski_template = f.read()

    # Check that the selected template is compatible with the cosmological setting
    if 'LocalUniverseCosmology' in ski_template and cosmological:
        raise ValueError("The selected template is for a local universe cosmology, but cosmological=True. Please select a different template or set cosmological=False.")
    elif 'LocalUniverseCosmology' not in ski_template and not cosmological:
        raise ValueError("The selected template is for a cosmological simulation, but cosmological=False. Please select a different template or set cosmological=True.")

    dust_emission_type = dust_emission_type.capitalize() # format to match expected template formatting
    if dust_emission_type not in ["Equilibrium", "Stochastic"]:
        raise ValueError(f"Invalid dust_emission_type: '{dust_emission_type}'. Must be 'equilibrium' or 'stochastic'.")

    # Determine the redshift of the simulation based on the snapshot number and FIRE version
    if cosmological:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if snapshot_scale_factors:
            scale_factors = np.loadtxt(snapshot_scale_factors)
        if FIRE_ver == 0: raise ValueError("FIRE_ver must be set to 2 or 3 for cosmological simulations if snapshot_scale_factors not given.")
        elif FIRE_ver <=2: scale_factors = np.loadtxt(os.path.join(current_dir, "ski_templates/FIRE2_snapshot_scale-factors.txt"))
        else: scale_factors = np.loadtxt(os.path.join(current_dir, "ski_templates/FIRE3_snapshot_scale-factors.txt"))
        z = 1.0 / scale_factors[snapnum] - 1
    else:
        z = 0

    # For high redshift CMB heating of dust grains should be accounted for
    if z>3: include_CMB_heating = "true"
    else: include_CMB_heating = "false"

    # For non-zero redshift snapshots, present-day observers are symbolized by an observer distance of zero. 
    # However, at z!=0 SKIRT will throw a fatal error. 
    # To get around this for the z=0 snapshot, override its redshift to a small value.
    # More info at https://skirt.ugent.be/root/_user_redshift.html
    if cosmological and (z==0 or (FIRE_ver==2 and snapnum == 600) or (FIRE_ver==3 and snapnum == 500)):
        z = 0.003 # corresponds to a luminosity distance of ~13 Mpc
        print(f"WARNING: Overriding redshift for snapnum {snapnum} to {z = } so observer instruments are at non-zero distance from object.")

    # Define a cosmology object for a given FIRE-2 object
    if cosmo_params not in ['PLANCK', 'AGORA']:
        print(f"Unknown cosmological parameters: '{cosmo_params}'. ")
        print("Defaulting to PLANCK cosmological constants")
        cosmo_params = 'PLANCK'
    if cosmo_params == 'AGORA':
        cosmo = LambdaCDM(
            name="AGORA",
            H0=70.2,
            Om0=0.272,
            Ode0=0.728,
            Ob0=0.0455,
            meta={
                "n_s": 0.961,
                "sigma8": 0.807,
                "reference": "Wetzel et al. 2023, Table 1",
            },
        )
    else:
        cosmo = LambdaCDM(
            name="PLANCK",
            H0=68.0,
            Om0=0.31,
            Ode0=0.69,
            Ob0=0.048,
            meta={
                "n_s": 0.97,
                "sigma8": 0.82,
                "reference": "Wetzel et al. 2023, Table 1",
            },
        )

    # Set arbitrary distance to the object in Mpc
    d_to_source = distance  # only used for rest-frame instruments which have flat cosmology, all other instruments have distance set by redshift

    # Calculate the spatial scale in the simulation (in physical kpc) that
    # corresponds to 1 pixel for rest frame and observer frame instruments
    if default_pixel_ang_scale is not None:
        default_rest_kpc_per_pixel = default_pixel_ang_scale * u.arcsec.to(u.radian) * d_to_source * 1e3 # convert from Mpc to kpc
        if cosmological:
            default_obs_kpc_per_pixel = (
                cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value * default_pixel_ang_scale
            )
        else:
            default_obs_kpc_per_pixel = default_rest_kpc_per_pixel
        
    elif default_pixel_phy_scale is not None:
        default_obs_kpc_per_pixel = default_pixel_phy_scale
        default_rest_kpc_per_pixel = default_pixel_phy_scale
    else:
        raise ValueError("Must specify either default_pixel_ang_scale or default_pixel_phy_scale.")

    

    # Define limits for relevant wavelength ranges used in SKIRT

    # 1. Define *rest-frame* wavelength range for sources (stars)
    min_wavelength_source = min_source_wavelength if min_source_wavelength is not None else min_wavelength
    max_wavelength_source = max_source_wavelength if max_source_wavelength is not None else max_wavelength

    # 2. Define wavelength limits for dust emission and radiation field
    #    grids; these are set to the same values reported in Camps & Baes (2020)
    #    used for stochastic grain heating. Includes increased binning across
    #    MIR for PAH emission lines.
    num_wavelengths_dust_base = num_dust_emission_wavelengths if num_dust_emission_wavelengths is not None else num_wavelengths
    # The wavelength range of dust emission needs to capture the entire emission spectrum 
    # in order the produce the correct luminosities per the SKIRT docs. 
    min_wavelength_dust_base = 0.2 # microns
    max_wavelength_dust_base = 1000 # microns

    # Need finer wavelength binning for PAH features in MIR
    num_wavelengths_dust_sub = 2 * num_wavelengths_dust_base

    min_PAH_wavelength = 3.0 # microns; approximate start of PAH features
    max_PAH_wavelength = 25.0 # microns; approximate end of PAH features
    if min_wavelength_dust_base > min_PAH_wavelength or max_wavelength_dust_base < max_PAH_wavelength:
        print("WARNING: The specified min or max dust emission wavelengths excludes the MIR regime where PAH features reside (3.0 - 25.0 microns).")
    min_wavelength_dust_sub, max_wavelength_dust_sub = np.max([3.0, min_wavelength_dust_base]), np.min([25.0, max_wavelength_dust_base])

    # This is the wavelength grid SKIRT stores radiation field info for each cell. 
    # Can't be too large since this uses a lot of memory (each cell stores the rad field at each wavelength).
    num_wavelengths_rad = num_rad_field_wavelengths if num_rad_field_wavelengths is not None else num_wavelengths
    min_rad_field_wavelength = min_wavelength if min_rad_field_wavelength is None else min_rad_field_wavelength
    max_rad_field_wavelength = max_wavelength if max_rad_field_wavelength is None else max_rad_field_wavelength
    min_wavelength_rad, max_wavelength_rad = min_rad_field_wavelength, max_rad_field_wavelength

    # 3. Define wavelength limits for *observer-frame* SED instruments
    min_wavelength_obs, max_wavelength_obs = (
        min_wavelength * (1 + z),
        max_wavelength * (1 + z),
    )

    # 4. Determine if metallicity should be imported
    # For simulations without live dust you want SKIRT to infer the dust mass 
    # from the gas mass and metallicity. Otherwise you can give SKIRT the exact
    # dust mass.
    # If mass_fraction < 1, dust mass is inferred from M_dust = M_gas * Z * mass_fraction
    # If mass_fraction = 1, dust mass is given directly by the simulation.
    import_metallicity = "true" if mass_fraction < 1.0 else "false"

    # -------

    # either set instrument fov to the same as the spatial grid (half_fov_kpc) or use the user-provided value (inst_half_fov_kpc)
    if inst_half_fov_kpc is None: 
        inst_half_fov_kpc = half_fov_kpc
    grid_half_fov_pc = half_fov_kpc * 1e3
    inst_fov_pc = 2 * inst_half_fov_kpc * 1e3

    default_obs_npixels = int(np.ceil(2 * inst_half_fov_kpc / default_obs_kpc_per_pixel))
    default_rest_npixels = int(np.ceil(2 * inst_half_fov_kpc / default_rest_kpc_per_pixel))

    # Deal with extra user defined instruments in the provided template
    all_npixels = default_obs_npixels
    extra_npixels = {}
    extra_kpc_per_pixel = {}
    if instrument_pixel_ang_scales or instrument_pixel_phy_scales:
        for name, pixel_ang_scale in instrument_pixel_ang_scales.items():
            if cosmological:
                kpc_per_pixel = cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value * pixel_ang_scale
            else:
                kpc_per_pixel = pixel_ang_scale * u.arcsec.to(u.radian) * d_to_source * 1e3 # convert from Mpc to kpc
            extra_kpc_per_pixel[name] = kpc_per_pixel
            extra_npixels[name] = int(np.ceil(2 * inst_half_fov_kpc / kpc_per_pixel))
        for name, pixel_phy_scale in instrument_pixel_phy_scales.items():
            kpc_per_pixel = pixel_phy_scale
            extra_kpc_per_pixel[name] = kpc_per_pixel
            extra_npixels[name] = int(np.ceil(2 * inst_half_fov_kpc / kpc_per_pixel))
        all_npixels += list(extra_npixels.values())

    # Catch instruments with too many pixels, since these use a lot of memory and arent needed for most use cases.
    if np.max(all_npixels) >= 1e4:
        raise ValueError(
            "Number of pixels must be less than 1e4. "
            + "The current redshift is too small or the pixel scale is too fine."
        )

    if cosmological:
        print("SKIRT simulation is cosmological with the following parameters:")
        print(f"\t Redshift: {z}")
        print(f"\t Reduced Hubble Constant: {cosmo.h}")
        print(f"\t Matter Density Fraction: {cosmo.Om0}")
    else:
        print(f"SKIRT simulation is non-cosmological. All instruments are in the rest frame and distance to source is set to {d_to_source} Mpc.")

    print(
        f"SKIRT simulation default instrument setup:\n" +
        f"\t rest frame to source at {d_to_source} Mpc: {default_rest_npixels} x {default_rest_npixels} pixels ({inst_fov_pc:g} pc x {inst_fov_pc:g} pc) " +
        f"with a physical scale of {default_rest_kpc_per_pixel:.3f} kpc / pixel."
    )
    if cosmological:
        print(
            f"\t observer frame: {default_obs_npixels} x {default_obs_npixels} pixels ({inst_fov_pc:g} pc x {inst_fov_pc:g} pc) "
            + f"with a physical scale of {default_obs_kpc_per_pixel:.3f} kpc / pixel.\n"
        )
    if instrument_pixel_ang_scales or instrument_pixel_phy_scales:
        for name, npix in extra_npixels.items():
            print(
                f"SKIRT simulation extra instrument {name} setup:\n" +
                f"\t {npix} x {npix} pixels ({inst_fov_pc:g} pc x {inst_fov_pc:g} pc) "
                + f"with a physical scale of {extra_kpc_per_pixel[name]:.3f} kpc / pixel.\n"
            )


    # Format the SKIRT input file with the appropriate parameters
    with open(output_path, "w") as f:
        num_packets = "{:g}".format(num_packets)
        dict_placeholders = SafeDict(
                numPackets=f"{num_packets:s}",
                redshift=f"{z:f}",
                reducedHubbleConstant=f"{cosmo.h:f}",
                matterDensityFraction=f"{cosmo.Om0:f}",
                minWavelengthSource=f"{min_wavelength_source:.4f}",
                maxWavelengthSource=f"{max_wavelength_source:.4f}",
                minWavelengthRad=f"{min_wavelength_rad:.4f}",
                maxWavelengthRad=f"{max_wavelength_rad:.4f}",
                numWavelengthsRad=f"{num_wavelengths_rad:d}",
                massFraction=f"{mass_fraction:.4f}",
                importMetallicity=f"{import_metallicity:s}",
                importTemperature=f"{'true' if max_temperature > 0 else 'false'}",
                maxTemperature=f"{max_temperature:.4f}",
                minX=f"-{grid_half_fov_pc:.4f}",
                maxX=f"{grid_half_fov_pc:.4f}",
                minY=f"-{grid_half_fov_pc:.4f}",
                maxY=f"{grid_half_fov_pc:.4f}",
                minZ=f"-{grid_half_fov_pc:.4f}",
                maxZ=f"{grid_half_fov_pc:.4f}",
                dustEmissionType=f"{dust_emission_type:s}",
                includeHeatingByCMB=f"{include_CMB_heating:s}",
                distance=f"{d_to_source:.4f}",
                minWavelengthRest=f"{min_wavelength:.4f}",
                maxWavelengthRest=f"{max_wavelength:.4f}",
                minWavelengthObs=f"{min_wavelength_obs:.4f}",
                maxWavelengthObs=f"{max_wavelength_obs:.4f}",
                numWavelengths=f"{num_wavelengths:d}",
                fieldOfViewX=f"{inst_fov_pc:.4f}",
                fieldOfViewY=f"{inst_fov_pc:.4f}",
                DefaultObsNumPixelsX=f"{default_obs_npixels:d}",
                DefaultObsNumPixelsY=f"{default_obs_npixels:d}",
                DefaultRestNumPixelsX=f"{default_rest_npixels:d}",
                DefaultRestNumPixelsY=f"{default_rest_npixels:d}",
            )
        # Add extra placeholders for instruments with user-specified pixel scales
        for name, npix in extra_npixels.items():
            dict_placeholders = dict_placeholders | SafeDict(**{
                f"{name}ObsNumPixelsX": f"{npix:d}",
                f"{name}ObsNumPixelsY": f"{npix:d}",
            })
        # Add placeholders for dust emission wavelengths
        if skirt_sim_mode == "dust_emission":
            dict_placeholders = dict_placeholders | SafeDict(
                    minWavelengthBaseGridDust=f"{min_wavelength_dust_base:.4f}",
                    maxWavelengthBaseGridDust=f"{max_wavelength_dust_base:.4f}",
                    numWavelengthsBaseGridDust=f"{num_wavelengths_dust_base:d}",
                    minWavelengthSubGridDust=f"{min_wavelength_dust_sub:.4f}",
                    maxWavelengthSubGridDust=f"{max_wavelength_dust_sub:.4f}",
                    numWavelengthsSubGridDust=f"{num_wavelengths_dust_sub:d}",
                )
        # Add extra placeholders from octtree medium grid
        if medium_grid == "octtree":
            max_dust_frac = "{:g}".format(max_dust_frac) # convert to string
            dict_placeholders = dict_placeholders | SafeDict(
                    minTreeLevel=f"{min_tree_level:d}",
                    maxTreeLevel=f"{max_tree_level:d}",
                    maxDustFraction=f"{max_dust_frac:s}",
                )
        # Add any additional placeholders specified by the user
        for key,value in add_template_placeholders.items():
            dict_placeholders[key] = value
             
        # Find all placeholders in template and look for any that are missing in the dictionary
        formatter = Formatter()
        placeholders = [field_name for _, field_name, _, _ in formatter.parse(ski_template) if field_name]
        diff = list(set(placeholders) - set(dict_placeholders.keys()))
        if len(diff) > 0:
            print(f"ERROR: The following placeholders are in the template but missing from the default dictionary: {diff}")
            print("For additional instruments add the desired pixel scales to the instrument_pixel_ang_scales or instrument_pixel_phy_scales arguments and the corresponding placeholders will be automatically added to the dictionary.")
            print("For everything else, define desired values for the placeholders in the add_template_placeholders argument.")
            return
        # diff = list(set(dict_placeholders.keys()) - set(placeholders))
        # if len(diff) > 0:
        #     print(f"WARNING: The following placeholders are in the default dictionary but missing from the template: {diff}")

        template = ski_template.format_map(dict_placeholders)
        f.write(template)

    print(f"Formatted SKIRT input file written to {output_path}")


def format_job_script(
    snapnum: int,
    simulation_object: str,
    output_path: str,
    nodes: int = 1,
    tasks_per_node: int = 1,
    cpus_per_task: int = 4,
    memory: str = "100GB",
    walltime: str = "01:00:00",
    email: str = "",
    computer="BigRed200",
    ):
    """
    Formats a SLURM job script for running SKIRT.
    
    Parameters
    ----------
    snapnum : int
        Snapshot number to process.
    simulation_object : str
        Name of the simulation object (e.g., "m12i_res7100").
    output_path : str
        Path to the output directory for the job script.
    nodes : int, optional
            Number of nodes to use for the job.
    tasks_per_node : int, optional
        Number of tasks to run on each node.
    cpus_per_task : int, optional
        Number of CPU cores to allocate for each task.
    memory : str, optional
        Amount of RAM memory to allocate for the job (e.g., "100GB").
    walltime : str, optional
        Maximum wall time for the job (e.g., "01:00:00").
    email : str, optional
        Email address to receive job notifications (leave empty for no notifications).
    computer : str, optional
        Name of the computer cluster used to determine job template.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    job_template_filepath = os.path.join(current_dir, 'job_templates')
    if computer == "BigRed200":
        job_template_filepath += '/bigred200.sh'
    else:
        raise ValueError(f"Unknown computer: '{computer}'. No job template available.")

    # Read the SKIRT input file template
    with open(job_template_filepath, "r") as f:
        job_template = f.read()

    # Format the job script with the appropriate parameters
    # Make output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    job_name = 'skirt_'+simulation_object+'_snap_'+str(snapnum)

    with open(output_path, "w") as f:
        template = job_template.format_map(
            SafeDict(
                job_name=f"{job_name:s}",
                nodes=f"{nodes:d}",
                tasks_per_node=f"{tasks_per_node:d}",
                cpus_per_task=f"{cpus_per_task:d}",
                memory=f"{memory:s}",
                walltime=f"{walltime:s}",
                email=f"{email:s}"
            )
        )

        f.write(template)

    print(f"Formatted job script written to {output_path}")
