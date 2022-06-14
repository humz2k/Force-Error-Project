from sim import *
import matplotlib.pyplot as plt

def density(r,Rs,p0):
    return p0/((r/Rs)*((1+(r/Rs))**2))

def calcVol(r):
    return (4/3)*np.pi*(r**3)

radius = 100
c = 100
Rs = radius/c

particles = ParticleGenerator.NFW(radius,1000,c=c)

rs = np.linalg.norm(particles,axis=1)

vol = (4/3)*np.pi*(radius**3)

y,x = np.histogram(rs,bins=50)

p0 = np.sum(y)/vol
print(vol)

little = x[:-1]
big = x[1:]

volumes = calcVol(big)-calcVol(little)
densities = y/volumes

plt.plot(x[:-1]/Rs,densities)
#plt.xscale('log')
#plt.yscale('log')
plt.plot(x[:-1],density(x[:-1],Rs,1))
plt.show()