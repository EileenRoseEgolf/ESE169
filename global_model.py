import numpy as np
import pandas as pd
from model_structure import Rate

global_atmospheres = ['AfricaMidEast_atm','Asia_atm','Europe_atm',
                      'NorthAmer_atm','Oceania_atm','SouthAmer_atm','USSR_atm']
global_lands = ['AfricaMidEast_land','Asia_land','Europe_land',
                      'NorthAmer_land','Oceania_land','SouthAmer_land','USSR_land']

global_oceans = ['arcs','arcd',
                'npas','npad',
                'nats','natd',
                'socs','socd',
                'pacs','pacd',
                'sats','satd',
                'inds','indd',
                'meds','medd']

global_compartments = global_atmospheres + global_lands + global_oceans

def ocn2basin(ocn, lat):
    if ocn == 'ATL':
        if lat>20:
            return 'NAT'
        else:
            return 'SAT'
    elif ocn == 'PAC':
        if lat>20:
            return 'NPA'
        else:
            return 'PAC'
    else:
        return ocn

def get_region(CN, lon, lat):
    region = CN2Region[CN]
    if region == 'Europe':
        if (lon > 29) or (lon < -175):
            region = 'USSR'
    return region
    
def choose_TSS(lon,lat):
    i = np.argmin(np.abs(lons-lon))
    j = np.argmin(np.abs(lats-lat))
    imin = max(i-1,0)
    jmin = max(j-1,0)
    jmax = min(j+2,180)
    imax = min(i+2,360)
    return np.max(TSS[jmin:jmax,imin:imax]) # find the highest TSS in the window around i,j


def get_pb_aq_flux(region, vol):
    # vol in km3/yr
    return vol*RegionCw[region]*ug_Mg*km3_m3

def get_pb_part_flux(region, tss):
    # TSS in Mg/yr
    return tss*RegionCp[region]

def get_carbon_areaflux(pp, fexp, depth):
    return 0.1 * pp**1.77 * depth**fexp

def get_Pb_flux(cflux, surface_area, Pb_C_ratio):
    return cflux * surface_area * Pb_C_ratio * g2Mg


#-----------------------------------------------------------------------
# ATMOSPHERE RATES
#----------------------------------------------------------------------
atmosphere_rates = []

k_dep_lookup = {}

mass = np.loadtxt('atm/Pb_mass.txt') # kg/column
deptable = np.loadtxt('atm/Pb_TotDep_1990.csv',delimiter=',')
dep = np.zeros((180,360))
for row in deptable:
    i,j = int(row[0]-1), int(row[1]-1)
    dep[j,i] = row[4] # kg/cell/yr

for mask in ['Asia','Europe','NorthAmer','Oceania','SouthAmer','USSR']:
    m = np.flipud(np.loadtxt(f'atm/masks/{mask}.txt'))
    totdep = dep*m
    land_total_deposition = np.sum(totdep)
    land_total_mass = np.sum(m*mass)
    k_dep = land_total_deposition / land_total_mass
    atmosphere_rates.append(Rate(f'{mask} deposition', k_dep, f'{mask}_atm', f'{mask}_land'))
    k_dep_lookup[mask] = k_dep
    
m = np.flipud(np.loadtxt(f'atm/masks/Africa.txt'))
totdep = dep*m
m = np.flipud(np.loadtxt(f'atm/masks/MidEast.txt'))
totdep = totdep + dep*m
land_total_deposition = np.sum(totdep)
land_total_mass = np.sum(m*mass)
k_dep = land_total_deposition / land_total_mass

atmosphere_rates.append(Rate(f'AfricaMidEast deposition', k_dep, f'AfricaMidEast_atm',
                                 f'AfricaMidEast_land'))
k_dep_lookup['AfricaMidEast'] = k_dep


# DEPOSITION TO OCEANS
atm = 'NorthAmer'
atmosphere_rates.append(Rate(f'{atm} to nats deposition', 0.042*k_dep_lookup[atm],
                             f'{atm}_atm', f'nats'))
atmosphere_rates.append(Rate(f'{atm} to sats deposition', 0.017*k_dep_lookup[atm],
                             f'{atm}_atm', f'sats'))
atmosphere_rates.append(Rate(f'{atm} to npas deposition', 0.017*k_dep_lookup[atm],
                             f'{atm}_atm', f'npas'))
atm = 'SouthAmer'
atmosphere_rates.append(Rate(f'{atm} to pacs deposition', 0.018*k_dep_lookup[atm],
                             f'{atm}_atm', f'pacs'))
atmosphere_rates.append(Rate(f'{atm} to sats deposition', 0.006*k_dep_lookup[atm],
                             f'{atm}_atm', f'sats'))
atm = 'Europe'
atmosphere_rates.append(Rate(f'{atm} to nats deposition', 0.20*k_dep_lookup[atm],
                             f'{atm}_atm', f'nats'))
atmosphere_rates.append(Rate(f'{atm} to meds deposition', 0.066*k_dep_lookup[atm],
                             f'{atm}_atm', f'meds'))
atm = 'Asia'
atmosphere_rates.append(Rate(f'{atm} to npas deposition', 0.057*k_dep_lookup[atm],
                             f'{atm}_atm', f'npas'))
atmosphere_rates.append(Rate(f'{atm} to pacs deposition', 0.057*k_dep_lookup[atm],
                             f'{atm}_atm', f'pacs'))
atmosphere_rates.append(Rate(f'{atm} to inds deposition', 0.029*k_dep_lookup[atm],
                             f'{atm}_atm', f'inds'))
atm = 'Oceania'
atmosphere_rates.append(Rate(f'{atm} to socs deposition', 0.020*k_dep_lookup[atm],
                             f'{atm}_atm', f'socs'))
atmosphere_rates.append(Rate(f'{atm} to pacs deposition', 0.020*k_dep_lookup[atm],
                             f'{atm}_atm', f'pacs'))
atmosphere_rates.append(Rate(f'{atm} to inds deposition', 0.001*k_dep_lookup[atm],
                             f'{atm}_atm', f'inds'))

#-----------------------------------------------------------------------
# TERRESTRIAL RATES
#----------------------------------------------------------------------
terrestrial_rates = []
base_burial = 150/8e6 # Maximum from Rausch and Pacyna
for land in global_lands:
    terrestrial_rates.append(Rate(f'{land} sequestration', base_burial, land, None))


#-----------------------------------------------------------------------
# RIVER RATES
#----------------------------------------------------------------------

# regional river aqueous Pb concentrations (ug/m3)
RegionCw = {'AfricaMidEast':9.13e4, 'NorthAmer':11900, 'SouthAmer':50,
            'Oceania':3e3, 'Europe':178, 'Asia':95.3, 'USSR':3.6e3, None:0}
# regional river particle Pb concentrations (g/g)
RegionCp = {'AfricaMidEast':2.8e-5, 'NorthAmer':4.6e-5, 'SouthAmer':3.4e-5,
            'Oceania':6.69e-6, 'Europe':3.6e-9, 'Asia':2.75e-5, 'USSR':1.95e-4, None:0}



TSS = np.flipud(np.loadtxt('rivers/TSS_2x25.txt'))*1e6 # convert to Mg/yr
lons = np.loadtxt('rivers/lon_2x25.txt')
lats = np.flip(np.loadtxt('rivers/lat_2x25.txt'))


ug_Mg = 1e-12
km3_m3 = 1e9

# river continent codes to our regions
CN2Region = {'AM':'AfricaMidEast','AF':'AfricaMidEast','AS':'Asia','AU':'Oceania',
            'CA':'NorthAmer', 'CB':'NorthAmer','EU':'Europe', 'Ho':None, 'NA':'NorthAmer',
             'PO':'Oceania', 'SA':'SouthAmer'}


df = pd.read_csv('rivers/coastal-stns-byVol-updated-oct2007.txt',delim_whitespace=True,
                encoding="ISO-8859-1")
df = df.fillna('NA')
df['basin'] = df.apply(lambda x: ocn2basin(x['OCN'], x['latm']), axis=1)
df['region'] = df.apply(lambda x: get_region(x['CN'], x['lonm'], x['latm']), axis=1)
df['TSS (Mg/yr)'] = df.apply(lambda x: choose_TSS( x['lonm'], x['latm']), axis=1)
df['Pb aq (Mg/yr)'] = df.apply(lambda x: get_pb_aq_flux(x['region'], x['Vol(km3/yr)']), axis=1)
df['Pb part (Mg/yr)'] = df.apply(lambda x: get_pb_part_flux(x['region'], x['TSS (Mg/yr)']), axis=1)

terr_reservoir = 8e9/7. # Mg
subset = (df['region']=='Asia') & (df['basin'] == 'PAC')
kriv_aq_PAC_as = df['Pb aq (Mg/yr)'][subset].sum() / terr_reservoir
kriv_p_PAC_as = df['Pb part (Mg/yr)'][subset].sum() / terr_reservoir

subset = (df['region']=='Asia') & (df['basin'] == 'IND')
kriv_aq_IND_as = df['Pb aq (Mg/yr)'][subset].sum() / terr_reservoir
kriv_p_IND_as = df['Pb part (Mg/yr)'][subset].sum() / terr_reservoir

subset = (df['region']=='Asia') & (df['basin'] == 'ARC')
kriv_aq_ARC_as = df['Pb aq (Mg/yr)'][subset].sum() / terr_reservoir
kriv_p_ARC_as = df['Pb part (Mg/yr)'][subset].sum() / terr_reservoir


subset = (df['region']=='Oceania') & (df['basin'] == 'PAC')
kriv_aq_PAC_oceania = df['Pb aq (Mg/yr)'][subset].sum() / terr_reservoir
kriv_p_PAC_oceania = df['Pb part (Mg/yr)'][subset].sum() / terr_reservoir


NA_terr_reservoir = 8e9 * .163 # Mg
subset = (df['region']=='NorthAmer') & (df['basin'] == 'NAT')
kriv_aq_NAT = df['Pb aq (Mg/yr)'][subset].sum() / NA_terr_reservoir
kriv_p_NAT = df['Pb part (Mg/yr)'][subset].sum() / NA_terr_reservoir

subset = (df['region']=='NorthAmer') & (df['basin'] == 'NPA')
kriv_aq_NPA = df['Pb aq (Mg/yr)'][subset].sum() / NA_terr_reservoir
kriv_p_NPA = df['Pb part (Mg/yr)'][subset].sum() / NA_terr_reservoir


SA_terr_reservoir = 8e9 * .12 # Mg
subset = (df['region']=='SouthAmer') & (df['basin'] == 'SAT')
kriv_aq_SAT = df['Pb aq (Mg/yr)'][subset].sum() / SA_terr_reservoir
kriv_p_SAT = df['Pb part (Mg/yr)'][subset].sum() / SA_terr_reservoir

subset = (df['region']=='SouthAmer') & (df['basin'] == 'PAC')
kriv_aq_PAC = df['Pb aq (Mg/yr)'][subset].sum() / SA_terr_reservoir
kriv_p_PAC = df['Pb part (Mg/yr)'][subset].sum() / SA_terr_reservoir



f_buried = 0.9 # fraction of particulate that doesn't make it to ocean


river_rates = [Rate('asia river aq to PAC', kriv_aq_PAC_as, 'Asia_land', 'pacs'),
               Rate('asia river p to PAC', kriv_p_PAC_as * (1-f_buried), 'Asia_land', 'pacs'),
               Rate('asia river p to margin', kriv_p_PAC_as * f_buried, 'Asia_land', None),

               Rate('asia river aq to IND', kriv_aq_IND_as, 'Asia_land', 'inds'),
               Rate('asia river p to IND', kriv_p_IND_as * (1-f_buried), 'Asia_land', 'inds'),
               Rate('asia river p to margin', kriv_p_IND_as * f_buried, 'Asia_land', None),
         
               Rate('asia river aq to ARC', kriv_aq_ARC_as, 'Asia_land', 'arcs'),
               Rate('asia river p to ARC', kriv_p_ARC_as * (1-f_buried), 'Asia_land', 'arcs'),
               Rate('asia river p to margin', kriv_p_ARC_as * f_buried, 'Asia_land', None),
         
               Rate('oceania river aq to PAC', kriv_aq_PAC_oceania, 'Oceania_land', 'pacs'),
               Rate('oceania river p to PAC', kriv_p_PAC_oceania * (1-f_buried), 'Oceania_land', 'pacs'),
               Rate('oceania river p to margin', kriv_p_PAC_oceania * f_buried, 'Oceania_land', None),

               Rate('river aq to NAT', kriv_aq_NAT, 'NorthAmer_land', 'nats'),
               Rate('river p to NAT', kriv_p_NAT * (1-f_buried), 'NorthAmer_land', 'nats'),
               Rate('river p to margin', kriv_p_NAT * f_buried, 'NorthAmer_land', None),
               Rate('river aq to NPA', kriv_aq_NPA, 'NorthAmer_land', 'npas'),
               Rate('river p to NPA', kriv_p_NPA * (1-f_buried), 'NorthAmer_land', 'npas'),
               Rate('river p to margin', kriv_p_NPA * f_buried, 'NorthAmer_land', None),
         
               Rate('river aq to SAT', kriv_aq_SAT, 'SouthAmer_land', 'sats'),
               Rate('river p to SAT', kriv_p_SAT * (1-f_buried), 'SouthAmer_land', 'sats'),
               Rate('river p to margin', kriv_p_SAT * f_buried, 'SouthAmer_land', None),
               Rate('river aq to PAC', kriv_aq_PAC, 'SouthAmer_land', 'pacs'),
               Rate('river p to PAC', kriv_p_PAC * (1-f_buried), 'SouthAmer_land', 'pacs'),
               Rate('river p to margin', kriv_p_PAC * f_buried, 'SouthAmer_land', None),
]

# List the names of the compartments
compartments = ['NorthAmer', 'SouthAmer', 'AfricaMidEast', 'Oceania', 'Europe', 'Asia', 'USSR',
                'NorthAmer_atm', 'SouthAmer_atm', 'AfricaMidEast_atm', 'Oceania_atm', 'Europe_atm',
                'Asia_atm', 'USSR_atm','ARCs','ARCd', 'NPAs', 'NPAd', 'NATs', 'NATd', 'SOCs', 'SOCd',
               'PACs', 'PACd', 'SATs', 'SATd', 'INDs', 'INDd', 'MEDs', 'MEDd']

land_surface_areas_percentages = {'NorthAmer_land': 16.3, 'SouthAmer_land': 12.0,
                                  'AfricaMidEast_land': 26.6, 'Oceania_land': 5.2,
                                  'Europe_land': 3.7, 'Asia_land': 14.8, 'USSR_land': 13.1}

land_compartments = ['AfricaMidEast_land',
                      'Europe_land', 'USSR_land']
atm_compartments = ['NorthAmer_atm', 'SouthAmer_atm', 'AfricaMidEast_atm', 'Oceania_atm', 'Europe_atm',
                'Asia_atm', 'USSR_atm']
ocn_surface_compartments = ['ARC', 'NPA', 'NAT', 'SOC','PAC',  'SAT', 'IND', 'MED']

terr_reservoir = 8e9 # Mg
f_buried = 0.9 # fraction of particulate that doesn't make it to ocean

# Specify River fluxes to all ocean basins from all continents
for i, land_compartment in enumerate(land_compartments):
    for j, ocn_compartment in enumerate(ocn_surface_compartments):
        subset = (df['region']==land_compartment[:-5]) & (df['basin'] == ocn_compartment)
        k_aq = df['Pb aq (Mg/yr)'][subset].sum() / (terr_reservoir * land_surface_areas_percentages[land_compartment]/100)
        k_part = df['Pb part (Mg/yr)'][subset].sum() / (terr_reservoir * land_surface_areas_percentages[land_compartment]/100)
        if k_aq > 0:
            river_rates.append(Rate('River_Dissolved_'+land_compartment+'2'+ocn_compartment.lower()+'s', k_aq, land_compartment, ocn_compartment.lower()+'s'))
        if k_part > 0:
            river_rates.append(Rate('River_Particles_'+land_compartment+'2'+ocn_compartment.lower()+'s', k_part * (1-f_buried), land_compartment, ocn_compartment.lower()+'s'))
            river_rates.append(Rate('River_Particles_'+land_compartment+'buried', k_part * (f_buried), land_compartment, None))
        
#-----------------------------------------------------------------------
# OCEAN SINKING RATES
#----------------------------------------------------------------------

ocean_sinking_rates = []

Hg_C_ratio = 6e-7

Pb_Hg_ratio = 90
Pb_C_ratio = Hg_C_ratio * Pb_Hg_ratio

Pb_C_ratios = {'NPAs': 1.6e-6, 'NPAd':1.6e-6}

# ocean_concs inds 38 indd 5 pmol/kg
pmolkg_conv = 1000*1e-12*207*1e-6 # to Mg/m3
ngkg_conv = 1e-9*1000*1e-6 # to Mg/m3

ocean_concs = {'INDs': 38*pmolkg_conv, 'INDd': 5*pmolkg_conv, #pmol/kg
               'ARCs': 5.66e-12, 'ARCd': 3.5e-12, #Mg/m3
               'PACs': 15*pmolkg_conv, 'PACd': 5*pmolkg_conv,
               'NPAs': 10*ngkg_conv, 'NPAd': 1.8*ngkg_conv,
               'SOCs': 30*pmolkg_conv, 'SOCd': 4*pmolkg_conv,
               'SATs': 23*pmolkg_conv, 'SATd': 23*pmolkg_conv,
               'NATs': 140*pmolkg_conv, 'NATd': 45*pmolkg_conv,
               'MEDs': 5e-12, 'MEDd': 5e-12
}

# g C m-2 yr-1
primary_productivity = {'ARC':21, 'NPA': 149, 'NAT': 160, 'SOC': 59, 
                        'PAC': 107, 'SAT': 150, 'IND': 122, 'MED': 64}
flux_exponent = {'ARC':-0.68, 'NPA': -0.73, 'NAT': -0.68, 'SOC': -0.68, 
                        'PAC': -0.73, 'SAT': -0.68, 'IND': -0.73, 'MED': -0.68}
g2Mg = 1e-6

# m
ocean_volumes = {'ARCs':1.4e16, 'ARCd':2.8e15, 'NPAs': 2.5e16, 'NPAd': 7.5e16,
                 'NATs': 2.2e16, 'NATd': 4.4e16, 'SOCs': 6.8e16, 'SOCd':2e17, 
                  'PACs': 1.2e17, 'PACd': 3.5e17, 'SATs': 4.7e16, 'SATd':1.5e17,
                 'INDs': 4.9e16, 'INDd': 1.4e17, 'MEDs': 3.8e15, 'MEDd': 6.2e14}

ocean_Pb_reservoirs = {key:val*ocean_concs[key] for key,val in ocean_volumes.items()}

ocean_surface_areas = {'ARCs':1.6e13, 'ARCd':1.6e13, 'NPAs': 2.7e13, 'NPAd': 2.7e13,
                 'NATs': 2.5e13, 'NATd': 2.5e13, 'SOCs': 7.4e13, 'SOCd':7.4e13, 
                  'PACs': 1.3e14, 'PACd': 1.3e14, 'SATs': 5.1e13, 'SATd':5.1e13,
                 'INDs': 5.4e13, 'INDd': 5.4e13, 'MEDs': 4.0e12, 'MEDd': 4.0e12}

basin_list = [key for key,val in ocean_volumes.items()]

depths = {}
for key in ocean_volumes:
    if key.endswith('s'):
        depths[key] = 1000.
    else:
        depths[key] = 1000 + ocean_volumes[key]/ocean_surface_areas[key]

for basin in basin_list:
    Cflux = get_carbon_areaflux(primary_productivity[basin[:3]], flux_exponent[basin[:3]],
                                depths[basin])
    Pbflux = get_Pb_flux(Cflux, ocean_surface_areas[basin], Pb_C_ratios.get(basin, Pb_C_ratio))
    k_sink = Pbflux / ocean_Pb_reservoirs[basin]
    if basin.endswith('s'):
        destination = (basin[:3]+'d').lower()
    else:
        destination = None
    ocean_sinking_rates.append(Rate(f'{basin} sinking', k_sink,
                                    basin.lower(), destination))

#-----------------------------------------------------------------------
# OCEAN CIRCULATION RATES
#----------------------------------------------------------------------

ocean_circulation_rates = []

u            = 10**6 * 60 * 60 * 24 * 365 # unit conversion from Sv to m3/day

BoxVolume  = np.array([1.43e16, 2.80e15, 2.46e16, 7.54e16,
                        2.24e16, 4.43e16, 6.76e16, 2.07e17,
                        1.17e17, 3.52e17, 4.65e16, 1.46e17,
                        4.86e16, 1.39e17, 3.78e15, 6.21e14])

# Ocean surface exposed to atmosphere (m2)
OceanSfc   = [1.60e13,2.70e13,
              2.50e13,7.40e13,
              1.28e14,5.10e13,
              5.40e13,4.00e12]

# Ocean surface area (m2)
OceanSA    = [1.60e13,1.6e13,2.70e13,2.7e13,
              2.50e13,2.5e13,7.40e13,7.4e13,
              1.28e14,1.28e14,5.10e13,5.1e13,
              5.40e13,5.4e13,4.00e12,4.0e12]

# Horizontal transport in the surface ocean                                                         
F_Arcs_Nats    = u *  0.17   # Arctic to N. Atlantic                                               
F_Nats_Meds    = u *  0.03   # N. Atlantic to Medsterranean                                        
F_Sats_Nats    = u * 16.58   # S. Atlantic to N. Atlantic                                           
F_Socs_Sats    = u *  4.02   # Southern Ocean to S. Atlantic                                        
F_Inds_Sats    = u *  9.96   # Indsan to S. Atlantic                                                
F_Socs_Inds    = u *  0.38   # Southern Ocean to Indsan                                             
F_Pacs_Inds    = u * 13.65   # Pacsfic to Indsan                                                    
F_Socs_Pacs    = u * 12.12   # Southern Ocean to Pacsfic                                            
F_Pacs_Npas    = u *  1.18   # Pacsfic to N. Pacsfic                                                
F_Npas_Arcs    = u *  1.02   # N. Pacsfic to Arctic                                                 
    
# Horizontal transport in the deep ocean                                                             
F_ARCd_NATd    = u * 0.85    # Arctic to N. Atlantic                                                
F_MEDd_NATd    = u * 0.03    # Medsterranean to N. Atlantic                                         
F_NATd_SATd    = u * 17.6    # N. Atlantic to S. Atlantic                                           
F_SATd_SOCd    = u * 10.9    # S. Atlantic to Southern Ocean                                        
F_SATd_INDd    = u * 4.1     # S. Atlantic to Indsan                                                
F_INDd_SOCd    = u * 9.62    # Indsan to Southern Ocean                                             
F_SOCd_PACd    = u * 4.0     # Southern Ocean to Pacsfic                                            
F_PACd_INDd    = u * 1.45    # Pacsfc to Indsan                                                     
F_NPAd_PACd    = u * 0.16    # N. Pacsfic to Pacsfic      

# Vertical
F_Arcs_ARCd    = u * 0.85    # Arctic downwelling
F_Npas_NPAd    = u * 0.16    # N. Pacsfic downwelling
F_Nats_NATd    = u * 16.72   # N. Atlantic downwelling
F_SOCd_Socs    = u * 16.52   # Southern Ocean upwelling                                             
F_PACd_Pacs    = u * 2.71    # Pacsfic upwelling                                                    
F_SATd_Sats    = u * 2.6     # S. Atlantic upwelling                                                
F_Inds_INDd    = u * 4.07    # Indsan downwelling                                                   
F_Meds_MEDd    = u * 0.03    # Medsterranean downwelling   

F_Arcs_Nats    = u * (0.17 + 3.8)           # Arctic to N. Atlantic                            
F_ARCd_NATd    = u * (0.85 + 6.5)           # Arctic to N. Atlantic                              
F_Arcs_Nats    = u * (0.17 + 3.8 +6.5*2/3)  # Arctic to N. Atlantic                            
F_ARCd_NATd    = u * (0.85 + 6.5*1/3)       # Arctic to N. Atlantic                                 
F_Nats_Arcs    = u * (3.8 + 6.5*2/3)        # N. Atlantic to Arctic                                
F_NATd_ARCd    = u *  6.5*1/3               # N. Atlantic to Arctic                                
F_SOCd_Socs    = u * (16.52 + 11)           # Southern Ocean upwelling                              
F_Socs_SOCd    = u *  11                    # Southern Ocean downwelling                          

# horizontal transport in the surface ocean                                                          
ko_Arcs_Nats   = F_Arcs_Nats / BoxVolume[0]          
ocean_circulation_rates.append(Rate('circ_arcs_nats', k=ko_Arcs_Nats, cto='nats', cfrom='arcs',
                        ))
ko_Nats_Meds   = F_Nats_Meds / BoxVolume[4]        
ocean_circulation_rates.append(Rate('circ_nats_meds', k=ko_Nats_Meds, cto='meds', cfrom='nats',
                        ))
ko_Sats_Nats   = F_Sats_Nats / BoxVolume[10]         
ocean_circulation_rates.append(Rate('circ_sats_nats', k=ko_Sats_Nats, cto='nats', cfrom='sats',
                        ))
ko_Socs_Sats   = F_Socs_Sats / BoxVolume[6]          
ocean_circulation_rates.append(Rate('circ_socs_sats', k=ko_Socs_Sats, cto='sats', cfrom='socs',
                        ))
ko_Inds_Sats   = F_Inds_Sats / BoxVolume[12]         
ocean_circulation_rates.append(Rate('circ_inds_sats', k=ko_Inds_Sats, cto='sats', cfrom='inds',
                        ))
ko_Pacs_Inds   = F_Pacs_Inds / BoxVolume[8]               
ocean_circulation_rates.append(Rate('circ_pacs_inds', k=ko_Pacs_Inds, cto='inds', cfrom='pacs',
                        ))
ko_Socs_Pacs   = F_Socs_Pacs / BoxVolume[6]              
ocean_circulation_rates.append(Rate('circ_socs_pacs', k=ko_Socs_Pacs, cto='pacs', cfrom='socs',
                        ))
ko_Pacs_Npas   = F_Pacs_Npas / BoxVolume[8]           
ocean_circulation_rates.append(Rate('circ_pacs_npas', k=ko_Pacs_Npas, cto='npas', cfrom='pacs',
                        ))
ko_Npas_Arcs   = F_Npas_Arcs / BoxVolume[2]               
ocean_circulation_rates.append(Rate('circ_npas_arcs', k=ko_Npas_Arcs, cto='arcs', cfrom='npas',
                        ))
ko_Socs_Inds   = F_Socs_Inds / BoxVolume[6]               
ocean_circulation_rates.append(Rate('circ_socs_inds', k=ko_Socs_Inds, cto='inds', cfrom='socs',
                        ))
ko_Nats_Arcs    = F_Nats_Arcs / BoxVolume[4]                            
ocean_circulation_rates.append(Rate('circ_nats_arcs', k=ko_Nats_Arcs, cto='arcs', cfrom='nats',
                        ))
                                                                                                     
# horizontal transport in the deep ocean                                                             
ko_ARCd_NATd   = F_ARCd_NATd / BoxVolume[1]                
ocean_circulation_rates.append(Rate('circ_arcd_natd', k=ko_ARCd_NATd, cto='natd', cfrom='arcd',
                        ))
ko_MEDd_NATd   = F_MEDd_NATd / BoxVolume[15]               
ocean_circulation_rates.append(Rate('circ_medd_natd', k=ko_MEDd_NATd, cto='natd', cfrom='medd',
                        ))
ko_NATd_SATd   = F_NATd_SATd / BoxVolume[5]                
ocean_circulation_rates.append(Rate('circ_natd_satd', k=ko_NATd_SATd, cto='satd', cfrom='natd',
                        ))
ko_SATd_SOCd   = F_SATd_SOCd / BoxVolume[11]            
ocean_circulation_rates.append(Rate('circ_satd_socd', k=ko_SATd_SOCd, cto='socd', cfrom='satd',
                        ))
ko_SATd_INDd   = F_SATd_INDd / BoxVolume[11]                    
ocean_circulation_rates.append(Rate('circ_satd_indd', k=ko_SATd_INDd, cto='indd', cfrom='satd',
                        ))
ko_INDd_SOCd   = F_INDd_SOCd / BoxVolume[13]            
ocean_circulation_rates.append(Rate('circ_indd_socd', k=ko_INDd_SOCd, cto='socd', cfrom='indd',
                        ))
ko_SOCd_PACd   = F_SOCd_PACd / BoxVolume[7]                    
ocean_circulation_rates.append(Rate('circ_socd_pacd', k=ko_SOCd_PACd, cto='pacd', cfrom='socd',
                        ))
ko_PACd_INDd   = F_PACd_INDd / BoxVolume[9]                    
ocean_circulation_rates.append(Rate('circ_pacd_indd', k=ko_PACd_INDd, cto='indd', cfrom='pacd',
                        ))
ko_NPAd_PACd   = F_NPAd_PACd / BoxVolume[3]      
ocean_circulation_rates.append(Rate('circ_npad_pacd', k=ko_NPAd_PACd, cto='pacd', cfrom='npad',
                        ))
ko_NATd_ARCd    = F_NATd_ARCd / BoxVolume[5]                              
ocean_circulation_rates.append(Rate('circ_natd_arcd', k=ko_NATd_ARCd, cto='arcd', cfrom='natd',
                        ))

# vertical transport                                                                                 
ko_Arcs_ARCd   = F_Arcs_ARCd / BoxVolume[0]                       
ocean_circulation_rates.append(Rate('circ_arcs_arcd', k=ko_Arcs_ARCd, cto='arcd', cfrom='arcs',
                        ))
ko_Npas_NPAd   = F_Npas_NPAd / BoxVolume[2]                       
ocean_circulation_rates.append(Rate('circ_npas_npad', k=ko_Npas_NPAd, cto='npad', cfrom='npas',
                        ))
ko_Nats_NATd   = F_Nats_NATd / BoxVolume[4]                       
ocean_circulation_rates.append(Rate('circ_nats_natd', k=ko_Nats_NATd, cto='natd', cfrom='nats',
                        ))
ko_SOCd_Socs   = F_SOCd_Socs / BoxVolume[7]                     
ocean_circulation_rates.append(Rate('circ_socd_socs', k=ko_SOCd_Socs, cto='socs', cfrom='socd',
                        ))
ko_PACd_Pacs   = F_PACd_Pacs / BoxVolume[9]                    
ocean_circulation_rates.append(Rate('circ_pacd_pacs', k=ko_PACd_Pacs, cto='pacs', cfrom='pacd',
                        ))
ko_SATd_Sats   = F_SATd_Sats / BoxVolume[11]                   
ocean_circulation_rates.append(Rate('circ_satd_sats', k=ko_SATd_Sats, cto='sats', cfrom='satd',
                        ))
ko_Inds_INDd   = F_Inds_INDd / BoxVolume[12]                      
ocean_circulation_rates.append(Rate('circ_inds_indd', k=ko_Inds_INDd, cto='indd', cfrom='inds',
                        ))
ko_Meds_MEDd   = F_Meds_MEDd / BoxVolume[14]      
ocean_circulation_rates.append(Rate('circ_meds_medd', k=ko_Meds_MEDd, cto='medd', cfrom='meds',
                        ))
ko_Socs_SOCd    = F_Socs_SOCd / BoxVolume[6]    
ocean_circulation_rates.append(Rate('circ_socs_socd', k=ko_Socs_SOCd, cto='socd', cfrom='socs',
                        ))

