import PyCC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel,get_body
from astropy.time import Time
import astropy.units as u
from astropy import constants as const

t = Time('2016-03-20T12:30:00')
bodies = ['sun', 'earth', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune','pluto']
body_masses = {

    "sun": const.M_sun.to(u.kg),
    "earth": const.M_earth.to(u.kg),
    "moon": 7.34767309e22 * u.kg,
    "mercury": 3.285e23 * u.kg,
    "venus": 4.867e24 * u.kg,
    "mars": 6.39e23 * u.kg,
    "jupiter": 1.89813e27 * u.kg,
    "saturn": 5.683e26 * u.kg,
    "uranus": 8.681e25 * u.kg,
    "neptune": 1.024e26 * u.kg,
    "pluto": 1.309e22 * u.kg

}
positions = []
velocities = []
masses = []
for body in bodies:
    pos,vel = get_body_barycentric_posvel(body,t,ephemeris='jpl')
    positions.append(np.array(pos.xyz.to(u.meter)))
    velocities.append(np.array(vel.xyz.to(u.meter/u.second)))
    masses.append(float(body_masses[body] / u.kg))
positions = np.array(positions)
velocities = np.array(velocities)
masses = np.array(masses)

masses = pd.DataFrame(np.reshape(masses,(1,)+masses.shape).T,columns=["mass"])
positions = pd.DataFrame(positions,columns=["x","y","z"])
velocities = pd.DataFrame(velocities,columns=["vx","vy","vz"])
df = pd.concat((positions,velocities,masses),axis=1)

out,stats = PyCC.evaluate(df=df,save=False,dt=50000,steps=10000,algo="treecode")
print(stats)

fig = plt.figure()
ax = plt.axes(projection = "3d")
au = 1.496e11
pos = {}
for idx,name in enumerate(bodies):
    pos[name] = out[out["id"] == idx].loc[:,["x","y","z"]].to_numpy() / au
ax.scatter(pos["sun"][:,0],pos["sun"][:,1],pos["sun"][:,2])
for name in bodies[1:]:
    ax.plot(pos[name][:,0],pos[name][:,1],pos[name][:,2],label=name)
plt.legend()
plt.show()