import numpy as np
import h5py

def add_vcor_field(radar, field_i, field_o, data, std_name=None,
                   long_name=None, replace=False, description=None):
    """
   Add a field to the object with metadata from a existing field 
   (Py-ART) adding the possibility of defining "standard name" and 
   "long_name" attributes.

    Parameters
    ----------
    radar : Radar
        Py-ART radar structure
    field_i : str
        Reference field name
    field_o : str
        Added field name
    data : array
        Added field data
    std_name : str
        Standard name of added field
    long_name : str
        Long name of added field
    replace : bool
        True to replace the existing field
    description: str
        Description for metadata
        
    """

    radar.add_field_like(field_i, field_o, data, 
                         replace_existing=replace)
    if long_name is not None:
        radar.fields[field_o]['long_name'] = long_name
    if std_name is not None:
        radar.fields[field_o]['standard_name'] = std_name
    if description is not None:
        radar.fields[field_o]['dualprf_correction'] = description
        
        
def instrument_parameters_odimh5(radar, odim_file):
    """
    Builds the dictionary 'instrument_parameters' in the radar instance, 
    using the parameter metadata in the input odim5 file.

    Parameters
    ----------
    radar : Radar
        Py-ART radar structure
    odim_file : str
        Complete path and filename of input file

    Returns
    -------
    radar : Radar
        Py-ART radar structure with added 'instrument_parameters'.
    """

    ny, prt, prt_mode, prt_ratio, prf_flag = _get_prf_pars_odimh5(odim_file, nrays=radar.nrays, 
                                                                  nsweeps=radar.nsweeps, sw_start_end=radar.get_start_end)

    # Create dictionaries
    mode_dict = {'comments': 'Pulsing mode Options are: "fixed", "staggered", "dual". Assumed "fixed" if missing.',
                 'meta_group': 'instrument_parameters',
                 'long_name': 'Pulsing mode',
                 'units': 'unitless',
                 'data': prt_mode}
    prt_dict = {'units': 'seconds',
                'comments': 'Pulse repetition time. For staggered prt, also see prt_ratio.',
                'meta_group': 'instrument_parameters',
                'long_name': 'Pulse repetition time',
                'data': prt}
    ratio_dict = {'units': 'unitless',
                 'meta_group': 'instrument_parameters',
                 'long_name': 'Pulse repetition frequency ratio',
                 'data': prt_ratio}
    ny_dict = {'units': 'meters_per_second',
               'comments': 'Unambiguous velocity',
               'meta_group': 'instrument_parameters',
               'long_name': 'Nyquist velocity',
               'data': ny}
    flag_dict = {'units': 'unitless',
                 'comments': 'PRF used to collect ray. 0 for high PRF, 1 for low PRF.',
                 'meta_group': 'instrument_parameters',
                 'long_name': 'PRF flag',
                 'data': prf_flag}

    # add metadata in radar object:
    radar.instrument_parameters = {'nyquist_velocity':ny_dict, 'prt':prt_dict, 
                                   'prt_ratio':ratio_dict, 'prt_mode':mode_dict, 
                                   'prf_flag':flag_dict}
    
    return radar


def _get_prf_pars_odimh5(odim_file, nrays, nsweeps, sw_start_end):
    """
    Credit: Joshua Soderholm (joshua-wx)
    
    Retrieves PRF scanning parameters from odim5 file, shaped for 
    building the 'instrument_parameters' dictionary: 
    nyquist velocity, PRF, dual PRF factor and PRF flags for each ray
    (if batch mode dual-PRF).

    Parameters
    ----------
    radar : Radar
        Py-ART radar structure

    Returns
    -------
    ny_array : numpy array (float)
        Nyquist velocity for each ray.
    prt_array : numpy array (float)
        PRT for each ray.
    prt_mode_array: numpy array (str)
        PRT mode for each sweep, 'dual' or 'fixed'.
    prt_ratio_array: numpy array (float)
        PRT ratio for each ray.
    prf_flag_array: numpy array (bool int)
        Ray PRF flag, high (0) or low (1) PRF.    
    """

    ny_array = np.zeros(nrays)
    prt_array = np.zeros(nrays)
    prt_mode_array = np.repeat(b'fixed', nsweeps)
    prt_ratio_array = np.ones(nrays)
    prf_flag_array = np.zeros(nrays)

    with h5py.File(odim_file, 'r') as hfile:
    
        for sw in range(0, nsweeps):
    
            # extract PRF/NI data from odimh5 file
            d_name = 'dataset' + str(sw+1)
            d_how = hfile[d_name]['how'].attrs
            try:
                ny = d_how['NI']
            except Exception as e:
                ny = 0
                print(f'Failed to read NI for sweep {sw} in {odim_file}', e)
                
            try:
                prf_h = d_how['highprf']
            except Exception as e:
                prf_h = 1000
                print(f'Failed to read highprf for sweep {sw} in {odim_file}', e)
                
            try:
                prf_ratio = d_how['rapic_UNFOLDING']
            except Exception as e:
                prf_ratio = None
                print(f'Failed to read rapic_UNFOLDING for sweep {sw} in {odim_file}', e)
                
            try:
                prf_type = d_how['rapic_HIPRF']
            except Exception as e:
                prf_type = None #single PRF
                
                #print(f'rapic_HIPRF missing for sweep {sw} in {odim_file}, assuming single PRF', e)
                
            # extract rays for current sweep
            ray_s, ray_e = sw_start_end(sw) # start and end rays of sweep
            ray_e += 1

            # Assign values
            prt_array[ray_s:ray_e] = 1/prf_h
            ny_array[ray_s:ray_e] = ny
    
            if prf_ratio != b'None' and prf_type != None:
        
                prt_mode_array[sw] = b'dual'
        
                fact_h = float(prf_ratio.decode('ascii')[0])
                fact_l = float(prf_ratio.decode('ascii')[2])
        
                prt_ratio_array[ray_s:ray_e] = fact_l/fact_h
                flag_sw = prf_flag_array[ray_s:ray_e]
        
                if prf_type==b'EVENS':
                    flag_sw[::2] = 1 # with 1 as the first index, 1=1, 2=0, 3=1
                elif prf_type==b'ODDS': #odds=0, evens=1
                    flag_sw[1::2] = 1 #with 1 as the first index, 1=0, 2=1, 3=0
                elif prf_type==b'None':
                    prt_mode_array[sw] = b'fixed'
                else:
                    print('error, unknown flag type', prf_type)
                #reinsert flag sweep into prf_flag_array
                prf_flag_array[ray_s:ray_e] = flag_sw
            else:
                prt_mode_array[sw] = b'fixed'

    return ny_array, prt_array, prt_mode_array, prt_ratio_array, prf_flag_array.astype(int)

def get_prf_pars(radar, sw):
    """
    Retrieves PRF scanning parameters from radar object: 
    nyquist velocity, PRF, dual PRF factor and PRF flags for each ray
    (if batch mode dual-PRF).

    Parameters
    ----------
    radar : Radar
        Py-ART radar structure

    Returns
    -------
    v_ny : float
        Nyquist velocity.
    prf_h : float
        PRF, high if dual mode.
    prf_fact: int or None
        Dual-PRF factor (for batch and stagger modes).
    prf_flag : array (1D) or None
        Ray flag: high (0) or low (1) PRF.
    """

    pars = radar.instrument_parameters

    sweep_start = radar.get_start(sw)
    sweep_slice = radar.get_slice(sw)
    v_nyq = pars['nyquist_velocity']['data'][sweep_start]
    prf_h = round(1 / pars['prt']['data'][sweep_start], 0)
    prt_mode = pars['prt_mode']['data'][sw]
    prf_fact = None
    prf_flag = None
    
    if prt_mode != b'fixed':
        prt_rat = pars['prt_ratio']['data'][sweep_start]
        if prt_rat != 1.0:
            prf_fact = int(round(1 / (prt_rat - 1), 0))
    if prt_mode == b'dual':
        prf_flag = pars['prf_flag']['data'][sweep_slice].astype(int)

    return v_nyq, prf_h, prf_fact, prf_flag

def prf_factor_array(radar, sw):
    """
    Returns an array with the dual-PRF factor for each gate.
    Raises error if dual-PRF factor info is not available in
     the radar object.

    Parameters
    ----------
    radar : Radar
        Py-ART radar structure
 
    Returns
    -------
    prf_fac_arr : numpy array
        Data with dual-PRF factor for each gate
    """

    v_ny, prf_h, prf_fact, prf_flag = get_prf_pars(radar, sw)
    dim = (radar.nrays, radar.ngates)
    if prf_fact is None:
        print('ERROR: dual-PRF factor is missing.\nIs this dual-PRF data?')
        return None

    if prf_flag is None:
        flag_vec = np.zeros(dim[0])
        print('WARNING: prf_flag is missing.')

    else:
        flag_vec = prf_flag

    flag_arr = np.transpose(np.tile(flag_vec.astype(int), (dim[1], 1)))    
    prf_fac_arr = flag_arr + prf_fact

    return prf_fac_arr