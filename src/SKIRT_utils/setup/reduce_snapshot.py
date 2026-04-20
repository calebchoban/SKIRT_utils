
import os
import numpy as np
import importlib.util
from crc_scripts.utils.stellar_hsml_utils import get_particle_hsml
from crc_scripts.io.gizmo import load_halo
from crc_scripts.io.particle import Particle

def create_SKIRT_particle_files(snap_dir, 
                                snap_num, 
                                output_dir, 
                                import_dust=False, 
                                import_species=False, 
                                import_sizes=False,
                                import_dust_hsml=True, 
                                use_halo_file=False, 
                                star_file_name='stars.dat', 
                                dust_file_name='dust.dat', 
                                max_box_coords=None,
                                max_box_virial=False,
                                set_principal_vectors=None):
    '''
    This function extracts and reduces the star particle and dust cell data from the specified simulation snapshot 
    into SKIRT input file. Multiple options for dust are available depending on your simulation. Simulations
    with no dust model can still be used with SKIRT by assuming a constant dust-to-metals ratio and using the 
    gas metallicity to determine the dust mass. Simulations with a dust model can have the dust mass 
    extracted (with grain composition and size distribution set by SKIRT), the mass of each dust species 
    (only size distribution set by SKIRT), or the grain size distributions for each species, depending on
    what is tracked in the simulation.

    Parameters
    ----------
    snap_dir : string
        Name of snapshot directory
    snap_num : int
        Snapshot number
    output_dir : string
        Name of directory where SKIRT outputs will be written to
    import_dust_hsml : boolean
        Sets whether gas/dust smoothing length will be extracted from simulation. This is used by 
        SKIRT when building an octotree grid for the dust medium. Set to False if using the 
        Voronoi grid since each particle position is used to make a Voronoi tesselation.
    import_dust : boolean
        Sets whether dust mass will be extracted from simulation. Otherwise a constant D/Z 
        ratio is assumed by SKIRT.
    import_species : boolean
        Requires import_dust to be True. Sets whether dust species masses will be extracted from the simulation.
        Otherwise a set dust population (MW, LMC, SMC) will be assumed by SKIRT.
    import_sizes : boolean
        Requires import_dust to be True. Sets whether dust grain size distribution will be extracted from the simulation. Otherwise a set dust size distribution for each dust species (Weingartner MW/SMC, Zubko) will be used by SKIRT.
    use_halo_file : string, optional
        Use halo files (i.e. AHF) to initially center halo. Only really needed for high-z or subhalos.
    star_file_name : string
        Name of file for star particle data.
    dust_file_name : string
        Name of file for dust particle data. Note for import_sizes option, this will be used as a prefix and the species name will be added to the file name (i.e. dust_sil.dat, dust_carb.dat, dust_pah.dat).
    max_box_coords : list of floats, optional
        List of maximum x, y, z coordinates in kpc of star and gas/dust particles to be exported. 
        If None, all particles are included. If provided, the list should be in the form 
        [max_x, max_y, max_z].
    max_box_virial : boolean, optional
        If True, the maximum box coordinates will be set to the virial radius of the halo. This option overrides max_box_coords if both are provided.
    set_principal_vectors : 2d array, optional
        Set the principal vectors used to orient the galaxy. This is a 3x3 array where each row is a vector. The first row is the vector for the x-axis, the second row is the vector for the y-axis, and the third row is the vector for the z-axis. If None, the principal vectors will be determined from the halo. Use this if you want to use the same orientation for multiple snapshots or if you want to set a specific orientation. 
    '''

    # Create output directory for reduced data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # This loads the galactic halo from the snapshot
    mode='AHF' if use_halo_file else None
    halo = load_halo(snap_dir, snap_num, mode=mode)
    # This orientates the halo so that the galactic disk is face-on
    print("Orientating halo")
    halo.set_orientation()
    if set_principal_vectors is not None:
        print("Overriding principal vectors with provided vectors...")
        halo.principal_axes_vectors = set_principal_vectors

    if not halo.sp.Flag_DustSpecies and import_dust:
        raise ValueError("import_dust set to True but this snapshot does not have a dust model. Set import_dust to False.")
    if not halo.sp.Flag_GrainSizeBins and import_sizes:
        raise ValueError("import_sizes set to True but this snapshot does not have grain size bins. Set import_sizes to False.")
    # Load data for star particles (ptype = 4)
    # For idealized sims, the initial stars are stored as ptype = 2 and 3 and need to be appended
    if not halo.sp.cosmological: append_dummies=True
    else: append_dummies=False
    p4 = halo.loadpart(4, append_dummies=append_dummies)

    
    # Compute the star softening lengths. This takes some time.
    # Do this before mask so softening length are correct for all particles.
    coords = p4.get_property('position')
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    print("Calculating star particle smoothing lengths...")
    h = get_particle_hsml(x, y, z)

    if max_box_virial:
        if not use_halo_file:
            raise ValueError("max_box_virial set to True but use_halo_file is not set to True. Set use_halo_file to True to use this option.")
        max_box_coords = [halo.rvir, halo.rvir, halo.rvir]
        print(f"Setting max box coordinates to virial radius of the halo: {halo.rvir} kpc")

    if max_box_coords is not None:
        coords = p4.get_property('position')
        x, y, z = coords[:,0], coords[:,1], coords[:,2]
        mask = (np.abs(x) <= max_box_coords[0]) & (np.abs(y) <= max_box_coords[1]) & (np.abs(z) <= max_box_coords[2])
        print(f"Applying mask to star particles with max box coordinates {max_box_coords} kpc")
        p4.mask(mask)
        h = h[mask]
        box_string = f"All particles in a x {max_box_coords[0]:.3f} kpc by y {max_box_coords[1]:.3f} kpc by z {max_box_coords[2]:.3f} kpc box around the galactic center are included."
    else:   
        box_string="All particles in the snapshot are included."

    # x,y,x coordinates
    coords = p4.get_property('position')
    x, y, z = coords[:,0], coords[:,1], coords[:,2]

    spam_spec = importlib.util.find_spec("gizmo_analysis")
    has_gizmo_analysis = spam_spec is not None

    if has_gizmo_analysis:
        # mass at formation, metallicity, and age
        m, Z, t = p4.get_property('M_form'), p4.get_property('Z_all')[:,0], 1e9*p4.get_property('age')
    else:
        print("WARNING: gizmo_analysis not installed. This module is needed to determine the zero-age stellar mass which SKIRT expects for star particles.")
        print("This will instead use the current star mass, which can underestimate the mass by <=30%.")
        print("You can pip install gizmo_analysis @ https://git@bitbucket.org/awetzel/gizmo_analysis.git")
        # current mass , metallicity, and age
        m, Z, t = p4.get_property('M'), p4.get_property('Z_all')[:,0], 1e9*p4.get_property('age')
    
    f = open(output_dir+"/"+star_file_name, 'w')
    # Write header for star file
    snap_info = f"Snapshot number: {snap_num}, redshift: {halo.redshift:.3f}, from snapshot directory: {snap_dir}"
    dust_info = ""
    if import_dust:
        if not import_species:
            dust_info = "Imported total dust mass information."
        elif import_species and not import_sizes:
            dust_info = "Imported dust species information."
        elif import_species and import_sizes:
            dust_info = "Imported dust species and grain size distribution information."
    else:
        dust_info = "No dust information imported. A constant dust-to-metals ratio will be assumed by SKIRT."
        
    header =    '# Star particle data\n' + \
                '# ' + snap_info + '\n' + \
                '# ' + box_string + '\n' + \
                '# Column 1: position x (pc)\n' + \
                '# Column 2: position y (pc)\n' + \
                '# Column 3: position z (pc)\n' + \
                '# Column 4: smoothing length (pc)\n' + \
                '# Column 5: mass (Msun)\n' + \
                '# Column 6: metallicity (1)\n' + \
                '# Column 7: age (yr)\n'
    f.write(header)
    # Step through each star particle and write its data
    for i in range(p4.npart):
        line = "%.2f %.2f %.2f %.2f %.3e %.3e %.3e\n" %(1e3*x[i],1e3*y[i],1e3*z[i],1e3*h[i],m[i],Z[i],t[i])
        f.write(line)
    f.close()

    print(f"Star data written to {star_file_name}...")

    # Load gas particle data (ptype = 0)
    p0 = halo.loadpart(0)

    if max_box_coords is not None:
        coords = p0.get_property('position')
        x, y, z = coords[:,0], coords[:,1], coords[:,2]
        mask = (np.abs(x) <= max_box_coords[0]) & (np.abs(y) <= max_box_coords[1]) & (np.abs(z) <= max_box_coords[2])
        print(f"Applying mask to gas particles with max box coordinates {max_box_coords} kpc")
        p0.mask(mask)

    # x,y,x coordinates
    coords = p0.get_property('position')
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    # If the snapshots include dust amounts, give those to SKIRT and set D/Z to 1
    # Else just assume a constant D/Z everywhere.
    h, T = p0.get_property('size'), p0.get_property('temperature')
    if import_dust and not import_species:
        # total dust mass
        m_dust = p0.get_property('M_dust')
    elif import_dust and import_species and not import_sizes:
        # silicate dust mass, carbonaceous dust mass, neutral PAH dust mass
        # Set the fraction of the carbonaceous dust mass that is in neutral PAHs since they are not explicitly tracked.
        pah_frac = 0.1 
        m_sil, m_carb, m_pah = p0.get_property('M_sil+'), (1-pah_frac)*p0.get_property('M_carb'), pah_frac*p0.get_property('M_carb')
    elif import_dust and import_species and import_sizes:
        # silicate, carbonaceous, and PAH dust masses and dust size distributions
        # Need to separate PAHs from general carbonaceous dust since they are not explicitly tracked.

        # Global grain bin info
        num_dust_bins = halo.sp.Flag_GrainSizeBins
        grain_bin_edges = halo.sp.Grain_Bin_Edges
        # Dust species info
        snap_species = halo.sp.dust_species
        spec_indices = halo.sp.dust_species_indices
        species_bin_mass = p0.get_property('grain_bin_mass') / 1.989E+33 # convert from grams to Msun

        # silicates
        sil_bin_mass = species_bin_mass[:,spec_indices[snap_species.index('silicates')]]
        if 'iron' in snap_species:
            sil_bin_mass += species_bin_mass[:,spec_indices[snap_species.index('iron')]]
        m_sil = np.sum(sil_bin_mass, axis=1)
        sil_bin_weight = sil_bin_mass/m_sil[:,np.newaxis] * num_dust_bins # mass weights for each silicate grain size bin

        # carbonaceous dust and PAHs
        carb_bin_mass = species_bin_mass[:,spec_indices[snap_species.index('carbonaceous')]]

        max_PAH_size = 1.3e-3 # microns, this is the size below which grains are considered PAHs. For the fiducial Nbin=16, this corresponds to the 2 smallest grain size bins
        pah_bin_cut = np.where(grain_bin_edges <= max_PAH_size)[0][-1] # index of the last grain size bin that is considered a PAH
        num_PAH_bins = pah_bin_cut + 1
        print(f"Separating PAHs from carbonaceous dust using a max PAH size of {max_PAH_size} microns which corresponds to the {num_PAH_bins} smallest grain size bins.")
        pah_bin_mass = np.copy(carb_bin_mass[:,:pah_bin_cut+1])
        pah_mass = np.sum(pah_bin_mass, axis=1)
        PAH_bin_weight = pah_bin_mass/pah_mass[:,np.newaxis] * num_PAH_bins # mass weights for each PAH grain size bin
        carb_bin_mass[:,:pah_bin_cut+1] = 0 # set the carbonaceous dust mass in the PAH bins to 0 since that mass is counted as PAH mass
        carb_mass = np.sum(carb_bin_mass, axis=1)
        carb_bin_weight = carb_bin_mass/carb_mass[:,np.newaxis] * (num_dust_bins) # mass weights for each carbonaceous grain size bin
    else:
        # gas mass, metallicity
        m_gas, Z = p0.get_property('M'), p0.get_property('Z_all')[:,0]


    # Need to save multiple files for grain size distributions
    if import_dust and import_sizes:
        dust_file_names = [dust_file_name.split('.dat')[0]+'_silicate.dat', dust_file_name.split('.dat')[0]+'_graphite.dat', dust_file_name.split('.dat')[0]+'_neutralPAH.dat']
        spec_bins = [num_dust_bins,num_dust_bins,num_PAH_bins]
        spec_masses = [m_sil, carb_mass, pah_mass]
        spec_weights = [sil_bin_weight, carb_bin_weight, PAH_bin_weight]
    else:
        dust_file_names = [dust_file_name]

    for k,file_name in enumerate(dust_file_names):
        f = open(output_dir+"/"+file_name, 'w')
        # Make header for gas/dust with data columns matching specified properties
        # Position is always needed, smoothing length used for octtree grid, 
        # and temperature used as a cut off for dust mass. 
        # Then other columns depend on what dust properties are being imported. 
        #   1. If not importing dust, then import gas mass and metallicity to allow 
        #      SKIRT to calculate dust mass using an assumed D/Z ratio. 
        #   2. If importing dust but not species, then import only total dust mass. 
        #   3. If importing species but not sizes, then import the mass of each dust species. 
        #   4. If importing species and sizes, then import the mass in each grain size 
        #      bin for each species.
        
        header =   '# Dust cell data\n' + \
                '# ' + snap_info + '\n' + \
                '# ' + box_string + '\n' + \
                '# ' + dust_info + '\n' + \
                '# Column 1: position x (pc)\n' + \
                '# Column 2: position y (pc)\n' + \
                '# Column 3: position z (pc)\n'
        column_num = 4
        if import_dust_hsml:
            header += '# Column %i: smoothing length (pc)\n'%column_num
            column_num+=1
        if not import_dust:
            header += '# Column %i: mass (Msun)\n'%column_num + \
                    '# Column %i: metallicity (1)\n'%(column_num+1) + \
                    '# Column %i: temperature (K)\n'%(column_num+2)  
            column_num+=3         
        elif import_dust and not import_species:
            header += '# Column %i: dust mass (Msun)\n'%column_num
        elif import_dust and import_species and not import_sizes:
            header += '# Column %i: silicate mass (Msun)\n'%column_num + \
                    '# Column %i: graphite mass (Msun)\n'%(column_num+1) + \
                    '# Column %i: neutral PAH mass (Msun)\n'%(column_num+2)
            column_num+=3
        elif import_dust and import_species and import_sizes:
            num_bins = spec_bins[k]
            header += '# Column %i: dust mass (Msun)\n'%column_num
            column_num +=1
            for j in range(num_bins):
                header += '# Column %i: bin %i weight (1)\n'%(column_num,j+1)
                column_num +=1
            

        f.write(header)

        for i in range(p0.npart):
            line = "%.2f %.2f %.2f "%(1e3*x[i],1e3*y[i],1e3*z[i])
            if import_dust_hsml:
                line += "%.3e "%(1e3*h[i])
            if not import_dust:
                line += "%.3e %.3e %.3e\n" %(m_gas[i],Z[i],T[i])
            elif import_dust and not import_species:
                line += "%.3e\n" %(m_dust[i]) 
            elif import_dust and import_species and not import_sizes:
                line += "%.3e %.3e %.3e\n" %(m_sil[i], m_carb[i], m_pah[i])
            elif import_dust and import_species and import_sizes:
                num_bins = spec_bins[k]
                spec_m = spec_masses[k]
                spec_weight = spec_weights[k]
                line += "%.3e"%(spec_m[i])
                for j in range(num_bins):
                    line += " %.3e"%(spec_weight[i,j])
                line += "\n"
            f.write(line)
        f.close()

        print(f"Gas/Dust data written to {file_name}...")