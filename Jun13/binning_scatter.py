from calculations import *
import matplotlib.pyplot as plt
import numpy as np
from math import floor,ceil

plt.rcParams['figure.figsize'] = [8, 8]

def get_bins(density=None,n_particles=None,radius=None,file=None,eps=0,dr=5):

    rs,phis = get_program_for_particles(density=density,radius=radius,n_particles=n_particles,eps=eps)
    analytic = np.array([get_phi(density=density,radius=radius,point=r/radius) for r in rs])

    start,end = np.min(rs),np.max(rs)

    bins = np.arange(start,end,dr)

    return {"rs":rs,"phis":phis,"analytic":analytic,"bins":bins,"dr":dr}

def plot_bins(rs,phis,analytic,bins,dr,means=None,maxs=None,mins=None):
    plt.scatter(rs,phis-analytic - 1.,s=2,alpha=0.9,zorder=1)

    ys = [np.min(phis-analytic -1.),np.max(phis-analytic -1.)]
    for bin in bins:
        plt.plot([bin,bin],ys,color="black",alpha=0.5,zorder=0)
    plt.plot([bin+dr,bin+dr],ys,color="black",alpha=0.5,zorder=0)

    if means != None:
        plt.plot(bins+dr/2,means,color="red",zorder=2)

    if maxs != None:
        plt.plot(bins+dr/2,maxs,color="red",zorder=2)

    if mins != None:
        plt.plot(bins+dr/2,mins,color="red",zorder=2)

    plt.plot([np.min(rs),np.max(rs)],[-1,-1],color="green",zorder=3,alpha=0.8)

    plt.ticklabel_format(useOffset=False)
    plt.show()

def analyse_bins(rs,phis,analytic,bins,dr):
    means = []
    maxs = []
    mins = []
    ratios = phis-analytic -1.
    rs = np.array(rs)
    for bin in bins:
        larger = rs >= bin
        smaller = rs < bin + dr
        perfect = np.logical_and(larger,smaller)
        try:
            maxs.append(np.max(ratios[perfect]))
        except:
            maxs.append(-1)
        try:
            mins.append(np.min(ratios[perfect]))
        except:
            mins.append(-1)
        means.append(np.mean(ratios[perfect]))
    return {"means":means,"maxs":maxs,"mins":mins}

bins = get_bins(density=100,n_particles=2000,radius=100,eps=0,dr=2)
stats = analyse_bins(**bins)

plot_bins(**bins,**stats)
