from parcels import AdvectionRK4, Field, FieldSet, JITParticle, ScipyParticle
from parcels import ParticleFile, ParticleSet, Variable, VectorField, ErrorCode
from parcels.tools.converters import GeographicPolar 
from datetime import timedelta as delta
from os import path
from glob import glob
import numpy as np
import dask
import math
import xarray as xr
from netCDF4 import Dataset
import warnings
import matplotlib.pyplot as plt
import pickle
warnings.simplefilter('ignore', category=xr.SerializationWarning)
from operator import attrgetter

### USER INPUT ###

domain = [-125, -69, -57, 35]
data_out = '/storage/shared/oceanparcels/output_data/data_Steffie/'
fname = 'particles_epibionts_obs_30d'

length_run = 410         #unit: length of total run
advection_duration = 100 #unit: days (how long does one particle floats around)
advection_duration_sec = advection_duration*86400
output_frequency = 12    #unit: hours
deltatime = -1           #unit: hours (dt of integration, negative when backwards)

### ADD HYDRODYNAMIC FIELDS ###

# get filenames
data_path = '/storage/shared/oceanparcels/input_data/MOi/'
ufiles = sorted(glob(data_path+'psy4v3r1/psy4v3r1-daily_U_201[7-9]*.nc'))
vfiles = [f.replace('_U_', '_V_') for f in ufiles]
wfiles = [f.replace('_U_', '_W_') for f in ufiles]
mesh_mask = data_path + 'domain_ORCA0083-N006/coordinates.nc'

# get indices for domain
def getclosest_ij(lats,lons,latpt,lonpt):
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_lat = (lats-latpt)**2          # find squared distance of every point on grid
    dist_lon = (lons-lonpt)**2
    minindex_lat = dist_lat.argmin()    # 1D index of minimum dist_sq element
    minindex_lon = dist_lon.argmin()
    return minindex_lat, minindex_lon   # Get 2D index for latvals and lonvals arrays from 1D index

dfile = Dataset(mesh_mask)
lon = dfile.variables['nav_lon'][1500,:]
lat = dfile.variables['nav_lat'][:,1100]
iy_min, ix_min = getclosest_ij(lat, lon, domain[2], domain[0])
iy_max, ix_max = getclosest_ij(lat, lon, domain[3], domain[1])

# make fieldset
filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles},
             'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles}}
variables = {'U': 'vozocrtx', 'V': 'vomecrty'}
dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
              'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}}
indices = {'lon': range(ix_min, ix_max),
           'lat': range(iy_min, iy_max),
           'depth': range(0, 2)}
fieldset = FieldSet.from_nemo(filenames, variables, dimensions, indices=indices)

### ADD OTHER FIELDS ###

file = open('input/inputfiles_epibionts', 'rb')
inputfiles = pickle.load(file)
file.close()

fieldset.add_field(Field('distance',
                         data = inputfiles['distance'],
                         lon = inputfiles['lon'],
                         lat = inputfiles['lat'],
                         mesh='spherical',
                         interp_method = 'linear'))

fieldset.add_field(Field('landmask',
                         data = inputfiles['landmask'],
                         lon = inputfiles['lon'],
                         lat = inputfiles['lat'],
                         mesh='spherical',
                         interp_method = 'nearest'))

fieldset.add_field(Field('coastcells',
                         data = inputfiles['coastcells'],
                         lon = inputfiles['lon'],
                         lat = inputfiles['lat'],
                         mesh='spherical',
                         interp_method = 'nearest'))

fieldset.add_constant('advection_duration',advection_duration_sec)
fieldset.add_constant('lon_max',lon[ix_max-5])
fieldset.add_constant('lon_min',lon[ix_min+5])
fieldset.add_constant('lat_max',lat[iy_max-5])
fieldset.add_constant('lat_min',lat[iy_min+5])

### ADD KERNELS ###

def Age(particle, fieldset, time):
    """Update age of a particle """
    particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > fieldset.advection_duration:
        particle.delete()

def delete_particle(particle, fieldset, time):
    """Delete if still out of domain"""
    particle.delete()

def AdvectionRK4(particle, fieldset, time):
    """ Only advect particles that are not out of bounds"""
    if particle.domain == 0:

        (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)

        if (lon1 < fieldset.lon_max and
            lon1 > fieldset.lon_min and
            lat1 < fieldset.lat_max and
            lat1 > fieldset.lat_min):

            (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
            lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)

            if (lon2 < fieldset.lon_max and
                lon2 > fieldset.lon_min and
                lat2 < fieldset.lat_max and
                lat2 > fieldset.lat_min):

                (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
                lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)

                if (lon3 < fieldset.lon_max and
                    lon3 > fieldset.lon_min and
                    lat3 < fieldset.lat_max and
                    lat3 > fieldset.lat_min):

                    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
                    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
                    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt

                else:
                    particle.domain = 1
            else:
                particle.domain = 1
        else:
            particle.domain = 1

def DomainTesting(particle, fieldset, time):
    """Check whether particle is near domain edge (=1): stop advection"""
    if particle.domain > 0:
        particle.domain = 1
    elif (particle.lon < fieldset.lon_max and
          particle.lon > fieldset.lon_min and
          particle.lat < fieldset.lat_max and
          particle.lat > fieldset.lat_min):
        particle.domain = 0
    else:
        particle.domain = 1

def Distances(particle, fieldset, time):
    """Save all distances of interest"""
    if particle.domain == 0:
        particle.distance_to_coast = fieldset.distance[time, particle.depth, particle.lat, particle.lon]
    
        #save distance_to_release
        R = 6373.0
        lat_1 = particle.lat*math.pi/180
        lon_1 = particle.lon*math.pi/180
        lat_2 = particle.releaselat*math.pi/180
        lon_2 = particle.releaselon*math.pi/180
        dlon = lon_2 - lon_1
        dlat = lat_2 - lat_1
        a = math.sin(dlat / 2)**2 + math.cos(lat_1) * math.cos(lat_2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        particle.distance_to_release = R*c

        #save distance_traveled
        lat_1 = particle.lat*math.pi/180
        lon_1 = particle.lon*math.pi/180
        lat_2 = particle.prev_lat*math.pi/180
        lon_2 = particle.prev_lon*math.pi/180
        dlon = lon_2 - lon_1
        dlat = lat_2 - lat_1
        a = math.sin(dlat / 2)**2 + math.cos(lat_1) * math.cos(lat_2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        particle.distance_traveled = R*c
        particle.prev_lon = particle.lon
        particle.prev_lat = particle.lat

def CoastTesting(particle, fieldset, time):
    """Check whether particle is in a coastal cell or on land"""
    if particle.domain == 0:

        landcheck = fieldset.landmask[time, particle.depth, particle.lat, particle.lon]
        if landcheck == 1:
            particle.beaching_flag = 1
        else:
            particle.beaching_flag = 0
        coastcheck = fieldset.coastcells[time, particle.depth, particle.lat, particle.lon]
        if coastcheck != 0:
            particle.coastcell_ID = coastcheck
            particle.openocean_flag = 0
        else:
            particle.coastcell_ID = 0
            particle.openocean_flag = 1

### ADD PARTICLE SET ###

startlon = inputfiles['releaselon'][:]
startlat = inputfiles['releaselat'][:]
startID = inputfiles['releaseID'][:]
starttime = inputfiles['releasetime'][:]

class PacificParticle(JITParticle):
    age = Variable('age', dtype=np.float32, to_write=False, initial = 0.)
    distance_to_coast = Variable('distance_to_coast', dtype=np.float32, initial = 0.)
    distance_to_release = Variable('distance_to_release', dtype=np.float32, initial = 0.)
    distance_traveled = Variable('distance_traveled', dtype=np.float32, initial = 0.) 
    beaching_flag = Variable('beaching_flag', dtype=np.int32, initial = 0.)
    coastcell_flag = Variable('openocean_flag', dtype=np.int32, initial = 0.)
    coastcell_ID = Variable('coastcell_ID', dtype=np.float32, initial = 0.)
    domain = Variable('domain', dtype=np.float32, to_write=False, initial = 0.)
    releaselon = Variable('releaselon', dtype=np.float32, to_write=False, initial = startlon)
    releaselat = Variable('releaselat', dtype=np.float32, to_write=False, initial = startlat)
    prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False, initial = startlon)
    prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False, initial = startlat)
    
pset = ParticleSet(fieldset=fieldset,
                   pclass = PacificParticle,
                   lon = startlon,
                   lat = startlat,
                   time = starttime)

### EXECUTE ###

kernel = (pset.Kernel(Age) + 
          pset.Kernel(AdvectionRK4) +
          pset.Kernel(DomainTesting) +
          pset.Kernel(Distances) + 
          pset.Kernel(CoastTesting))

filename = path.join(data_out, fname + '.zarr')
outfile = pset.ParticleFile(name=filename, outputdt=delta(hours=output_frequency))

pset.execute(kernel,
             runtime=delta(days=length_run),
             dt=delta(hours=deltatime),
             output_file=outfile,
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})

outfile.export()
outfile.close()
