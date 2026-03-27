
import os
import numpy as np
from string import Formatter
from astropy.cosmology import LambdaCDM
import astropy.units as u
import xml.etree.ElementTree as ET

def format_ski_file(
    snapnum: int,
    simulation_object: str,
    default_pixel_scale: float,
    JWST_pixel_scale: float,
    ALMA_pixel_scale: float,
    highres_pixel_scale: float,
    skirt_sim_mode: str,
    dust_emission_type: str,
    medium_grid: str,
    output_path: str,
    cosmological: bool = True,
    FIRE_ver: int = 0,
    num_packets: float = 1e7,
    num_wavelengths: int = 200,
    min_wavelength: float = 0.09,
    max_wavelength: float = 2000.0,
    half_fov_kpc: float = 18.0,
    max_temperature: float = 0.0,
    mass_fraction: float = 1.0,
    max_dust_frac: float = 1e-7,
    set_template_file: str ='',
    snapshot_scale_factors: str = '',
    **kwargs
):
    """
    Prepare SKIRT input files for a given snapshot of a FIRE simulation.

    Parameters
    ----------
    snapnum : int
        Snapshot number to process.
    simulation_object : str
        Name of the simulation object (e.g., "m12i_res7100").
    default_pixel_scale : float
        Default pixel scale in arcseconds per pixel for SKIRT images.
    JWST_pixel_scale : float
        Pixel scale in arcseconds per pixel for JWST images.
    ALMA_pixel_scale : float
        Pixel scale in arcseconds per pixel for ALMA images.
    highres_pixel_scale : float
        Pixel scale in arcseconds per pixel for hgh res images.
    skirt_sim_mode : str
        Simulation mode for SKIRT (e.g., "dust_emission" or "extinction_only").
    dust_emission_type : str
        Dust emission mode (e.g., "equilibrium" or "stochastic")
    medium_grid : str
        Dust medium grid used in SKIRT (e.g., "octtree" or "voronoi")
    output_path : str
        Path to the output directory for SKIRT input files.
    cosmological : bool, optional
        Set whether simulation is cosmological.
    FIRE_ver : int, optional
        Version of FIRE simulation (e.g., 2,3) used for determine redshift of snapshot number.
    num_packets : float, optional
        Number of packets for SKIRT simulation (default is "1e7").
    num_wavelengths : int, optional
        Number of wavelengths for SKIRT simulation (default is 200).
    min_wavelength : float, optional
        Minimum wavelength for SKIRT simulation (default is 0.09). 
    max_wavelength : float, optional
        Maximum wavelength for SKIRT simulation (default is 2000.0).
    half_fov_kpc : float, optional
        Half field of view in kpc (default is 18.0).
    max_temperature : float
        Maximum temperature for the medium elements in the simulation. Unless
        using a custom dust prescription (e.g. with a constant dust-to-metals
        ratio), this should always be set to 0.0 K.
    mass_fraction : float
        Fraction of the total medium mass to include in the simulation. This
        should be set to 1.0 unless using a custom dust prescription (e.g. with a
        constant dust-to-metals ratio).
    max_dust_frac : float
        When using medium_grid="octtree", this sets the maximum fraction of the
        total dust mass allowed in any grid cell. Smaller values lead to finer 
        griding.
    set_template_file : str
        Name of ski template you want to use. Will override the auto-selected template.
    snapshot_scale_factors : string
        Path to text file containing scale factors for each snapshot of the simulation.
        If not given, will default to FIRE-2/3 values depending on FIRE_ver.
    """

    if cosmological:
        if snapshot_scale_factors:
            scale_factors = np.loadtxt(snapshot_scale_factors)
        if FIRE_ver == 0: raise ValueError("FIRE_ver must be set to 2 or 3 for cosmological simulations if snapshot_scale_factors not given.")
        elif FIRE_ver <=2: scale_factors = np.loadtxt("FIRE2_snapshot_scale-factors.txt")
        else: scale_factors = np.loadtxt("FIRE3_snapshot_scale-factors.txt")
        z = 1.0 / scale_factors[snapnum] - 1
    else:
        z = 0

    # For high redshift CMB heating of dust grains should be accounted for
    if z>3: include_CMB_heating = "true"
    else: include_CMB_heating = "false"

    # For non-zero redshift snapshots, present-day observers are symbolized by an observer distance of zero. 
    # However, at z!=0 SKIRT with throw a fatal error. 
    # To get around this for the z=0 snapshot, override its redshift to a small value.
    # More info at https://skirt.ugent.be/root/_user_redshift.html
    # Important note for high-z, CMB heating of dust is off by default in this example.
    # To turn it on set includeHeatingByCMB="true" in DustEmissionOptions
    if z==0 or (FIRE_ver==2 and snapnum == 600) or (FIRE_ver==3 and snapnum == 500):
        z = 0.003 # corresponds to a luminosity distance of ~13 Mpc
        print(f"WARNING: Overriding redshift for snapnum {snapnum} to {z = } so observer instruments are at non-zero distance from object.")

    # Define a cosmology object for a given FIRE-2 object
    if simulation_object in ["m12i_res7100", "m10q_res7100"]:
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
        if simulation_object not in ["m11d_res7100", "m11i_res7100", "m11e_res7100", "HiZ"]:
            print(f"Unknown simulation object: '{simulation_object}'. ")
            print("Defaulting to PLANCK cosmological constants")
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


    # Calculate the spatial scale in the simulation (in physical kpc) that
    # corresponds to 1 pixel
    JWST_kpc_per_pixel = (
        cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value * JWST_pixel_scale
    )
    ALMA_kpc_per_pixel = (
        cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value * ALMA_pixel_scale
    )
    default_kpc_per_pixel = (
        cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value * default_pixel_scale
    )

    # Set arbitrary distance to the object in Mpc
    d_to_source = 10.0  # only used for rest-frame instruments all other instruments have distance set by redshift

    # Define limits for relevant wavelength ranges used in SKIRT

    # 1. Define *rest-frame* wavelength range for sources (stars)
    max_wavelength_source = 120.0  # microns;
    min_wavelength_source = min_wavelength

    # 2. Define wavelength limits for dust emission and radiation field
    #    grids; these are set to the same values reported in Camps & Baes (2020)
    #    used for stochastic grain heating. Includes increased binning across
    #    MIR for PAH emission lines.
    num_wavelengths_dust_base = 100
    min_wavelength_dust_base, max_wavelength_dust_base = 0.2, 2e3

    num_wavelengths_dust_sub = 200
    min_wavelength_dust_sub, max_wavelength_dust_sub = 3.0, 25.0

    num_wavelengths_rad = 50
    min_wavelength_rad, max_wavelength_rad = 0.02, 90.0

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

    # Determines name of template file which matches requested SKIRT mode and medium grid
    if set_template_file != '':
        ski_template_filepath = set_template_file
    else:
        template_dirc = "./ski_templates/default/"
        template_files = [file for file in os.listdir(template_dirc) if os.path.isfile(os.path.join(template_dirc, file))]
        ski_template_filepath=False
        for file in template_files:
            if all(modes in file for modes in [skirt_sim_mode,medium_grid]):
                ski_template_filepath = os.path.join(template_dirc, file)
        if not ski_template_filepath:
            raise ValueError("No template matches requested SKIRT mode")

    # Read the SKIRT input file template
    with open(ski_template_filepath, "r") as f:
        ski_template = f.read()


    inst_names = get_instrument_names(ski_template_filepath)

    JWST_npixels = int(np.ceil(2 * half_fov_kpc / JWST_kpc_per_pixel))
    ALMA_npixels = int(np.ceil(2 * half_fov_kpc / ALMA_kpc_per_pixel))
    default_npixels = int(np.ceil(2 * half_fov_kpc / default_kpc_per_pixel))

    if np.max([JWST_npixels,ALMA_npixels,default_npixels]) >= 1e4:
        raise ValueError(
            "Number of pixels must be less than 1e4. "
            + "The current redshift is too small or the pixel scale is too fine."
        )

    half_fov_pc = half_fov_kpc * 1e3
    fov_pc = 2 * half_fov_pc

    print(
        f"SKIRT simulation will use JWST {JWST_npixels} x {JWST_npixels} pixels ({fov_pc:g} pc x {fov_pc:g} pc) "
        + f"with a physical scale of {JWST_kpc_per_pixel:.2f} kpc / pixel."
    )
    print(
        f"SKIRT simulation will use ALMA {ALMA_npixels} x {ALMA_npixels} pixels ({fov_pc:g} pc x {fov_pc:g} pc) "
        + f"with a physical scale of {ALMA_kpc_per_pixel:.2f} kpc / pixel."
    )
    print(
        f"SKIRT simulation will use default {default_npixels} x {default_npixels} pixels ({fov_pc:g} pc x {fov_pc:g} pc) "
        + f"with a physical scale of {default_kpc_per_pixel:.2f} kpc / pixel."
    )

    # Format the SKIRT input file with the appropriate parameters

    # This is a workaround to handle missing keys in the dictionary and avoid
    # KeyError exceptions. It returns the key wrapped in curly braces if the key
    # is missing.
    class SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"


    # Make output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

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
                maxTemperature=f"{max_temperature:.4f}",
                minX=f"-{half_fov_pc:.4f}",
                maxX=f"{half_fov_pc:.4f}",
                minY=f"-{half_fov_pc:.4f}",
                maxY=f"{half_fov_pc:.4f}",
                minZ=f"-{half_fov_pc:.4f}",
                maxZ=f"{half_fov_pc:.4f}",
                dustEmissionType=f"{dust_emission_type:s}",
                includeHeatingByCMB=f"{include_CMB_heating:s}",
                distance=f"{d_to_source:.4f}",
                minWavelengthRest=f"{min_wavelength:.4f}",
                maxWavelengthRest=f"{max_wavelength:.4f}",
                numWavelengthsRest=f"{num_wavelengths:d}",
                minWavelengthObs=f"{min_wavelength_obs:.4f}",
                maxWavelengthObs=f"{max_wavelength_obs:.4f}",
                numWavelengthsObs=f"{num_wavelengths:d}",
                fieldOfViewX=f"{fov_pc:.4f}",
                fieldOfViewY=f"{fov_pc:.4f}",
                JWSTNumPixelsX=f"{JWST_npixels:d}",
                JWSTNumPixelsY=f"{JWST_npixels:d}",
                ALMANumPixelsX=f"{ALMA_npixels:d}",
                ALMANumPixelsY=f"{ALMA_npixels:d}",
                DefaultNumPixelsX=f"{default_npixels:d}",
                DefaultNumPixelsY=f"{default_npixels:d}",
                DefaultNumPixelsZ=f"{default_npixels:d}",  # for probes
            )
        if skirt_sim_mode == "dust_emission":
            dict_placeholders = dict_placeholders | SafeDict(
                    minWavelengthBaseGridDust=f"{min_wavelength_dust_base:.4f}",
                    maxWavelengthBaseGridDust=f"{max_wavelength_dust_base:.4f}",
                    numWavelengthsBaseGridDust=f"{num_wavelengths_dust_base:d}",
                    minWavelengthSubGridDust=f"{min_wavelength_dust_sub:.4f}",
                    maxWavelengthSubGridDust=f"{max_wavelength_dust_sub:.4f}",
                    numWavelengthsSubGridDust=f"{num_wavelengths_dust_sub:d}",
                )

        if medium_grid == "octtree":
            # min and max level of octtree allowed
            min_tree_level = 3
            max_tree_level = 20
            max_dust_frac = "{:g}".format(max_dust_frac) # convert to string
            dict_placeholders = dict_placeholders | SafeDict(
                    minTreeLevel=f"{min_tree_level:d}",
                    maxTreeLevel=f"{max_tree_level:d}",
                    maxDustFraction=f"{max_dust_frac:s}",
                )     

        # Find all placeholders in template and print them out to make sure all are being filled
        formatter = Formatter()
        placeholders = [field_name for _, field_name, _, _ in formatter.parse(ski_template) if field_name]
        print(placeholders)

        template = ski_template.format_map(dict_placeholders)
        f.write(template)

    print(f"Formatted SKIRT input file written to {output_path}")



def get_instrument_names(ski_file: str):
    """
    Extract instrument names from a SKIRT .ski parameter file, optionally filtered by type.
    
    Parameters
    ----------
    ski_file : str
        Path to the SKIRT .ski XML parameter file.
    
    Returns
    -------
    instrument_names : List[str]
        List of instrument names matching the specified criteria.
    
    """
    tree = ET.parse(ski_file)
    root = tree.getroot()
    
    instrument_names = []
    
    # Find the instrumentSystem element
    instrument_system = root.find('.//InstrumentSystem')
    if instrument_system is None:
        return instrument_names
    
    # Iterate over all instruments
    for instrument in instrument_system:
        # Filter by type if specified
        if instrument.tag not in ["FrameInstrument", "FullInstrument"]:
            continue
        
        # Get the instrumentName attribute
        name = instrument.get('instrumentName')
        if name is not None:
            instrument_names.append(name)
    
    return instrument_names



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


    job_template_filepath = './job_templates/'
    if computer == "BigRed200":
        job_template_filepath += 'bigred200.sh'
    else:
        raise ValueError(f"Unknown computer: '{computer}'. No job template available.")

    # Read the SKIRT input file template
    with open(job_template_filepath, "r") as f:
        job_template = f.read()

    # Format the job script with the appropriate parameters

    # This is a workaround to handle missing keys in the dictionary and avoid
    # KeyError exceptions. It returns the key wrapped in curly braces if the key
    # is missing.
    class SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"


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
