import numpy as np
import scipy as sp
import scipy.optimize as opt


def transform_param(p1, p2, p3, p4):
    nmax = np.abs(p1)
    xmax = np.abs(p2)
    x0 = -1000.0/2.0*(np.tanh((p3-0.0)/1000.0)+1.0) - 10.1
    lam = 200.0/2.0*(np.tanh((p4-500.0)/1000.0)+1.0) + 0.1
    
    return (nmax, xmax, x0, lam)


def gaisser_hillas_log(x, nmax, xmax, x0, lam):
    return np.log(nmax) \
           + ((xmax-x0)/lam) * np.log((x-x0)/(xmax-x0)) \
           + ((xmax-x)/lam)


def gaisser_hillas(x, nmax, xmax, x0, lam):
    return nmax \
           * np.power((x-x0)/(xmax-x0), (xmax-x0)/lam) \
           * np.exp((xmax-x)/lam)


def gaisser_hillas_log_fithelper(x, p1, p2, p3, p4):
    nmax, xmax, x0, lam = transform_param(p1,p2,p3,p4)
    return gaisser_hillas_log(x, nmax, xmax, x0, lam)


def gaisser_hillas_fithelper(x, p1, p2, p3, p4):
    nmax, xmax, x0, lam = transform_param(p1,p2,p3,p4)
    return gaisser_hillas(x, nmax, xmax, x0, lam)


def fit_x0(dep, dat, nmax, xmax, x0, lam):
    pars = [x0]
    f1 = lambda x,p3: gaisser_hillas_log_fithelper(x,nmax,xmax,p3,lam)
    f2 = lambda x,p3:     gaisser_hillas_fithelper(x,nmax,xmax,p3,lam)
    #pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    x0 = pars[0]
    
    return (nmax, xmax, x0, lam)

def fit_lam(dep, dat, nmax, xmax, x0, lam):
    pars = [lam]
    f1 = lambda x,p4: gaisser_hillas_log_fithelper(x,nmax,xmax,x0,p4)
    f2 = lambda x,p4:     gaisser_hillas_fithelper(x,nmax,xmax,x0,p4)
    pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    lam = pars[0]
    
    return (nmax, xmax, x0, lam)

def fit_xmax(dep, dat, nmax, xmax, x0, lam):
    pars = [xmax]
    f1 = lambda x,p2: gaisser_hillas_log_fithelper(x,nmax,p2,x0,lam)
    f2 = lambda x,p2:     gaisser_hillas_fithelper(x,nmax,p2,x0,lam)
    pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    xmax = pars[0]
    
    return (nmax, xmax, x0, lam)

def fit_x0_lam(dep, dat, nmax, xmax, x0, lam):
    pars = [x0, lam]
    f1 = lambda x,p3,p4: gaisser_hillas_log_fithelper(x,nmax,xmax,p3,p4)
    f2 = lambda x,p3,p4:     gaisser_hillas_fithelper(x,nmax,xmax,p3,p4)
    #pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    x0, lam = pars
    
    return (nmax, xmax, x0, lam)

def fit_xmax_lam(dep, dat, nmax, xmax, x0, lam):
    pars = [xmax, lam]
    f1 = lambda x,p2,p4: gaisser_hillas_log_fithelper(x,nmax,p2,x0,p4)
    f2 = lambda x,p2,p4:     gaisser_hillas_fithelper(x,nmax,p2,x0,p4)
    pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    xmax, lam = pars
    
    return (nmax, xmax, x0, lam)

def fit_nmax_xmax(dep, dat, nmax, xmax, x0, lam):
    pars = [nmax, xmax]
    f1 = lambda x,p1,p2: gaisser_hillas_log_fithelper(x,p1,p2,x0,lam)
    f2 = lambda x,p1,p2:     gaisser_hillas_fithelper(x,p1,p2,x0,lam)
    pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    nmax, xmax = pars
    
    return (nmax, xmax, x0, lam)

def fit_nmax_xmax_x0(dep, dat, nmax, xmax, x0, lam):
    pars = [nmax, xmax, x0]
    f1 = lambda x,p1,p2,p3: gaisser_hillas_log_fithelper(x,p1,p2,p3,lam)
    f2 = lambda x,p1,p2,p3:     gaisser_hillas_fithelper(x,p1,p2,p3,lam)
    #pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    nmax, xmax, x0 = pars
    
    return (nmax, xmax, x0, lam)

def fit_nmax_xmax_x0_lam(dep, dat, nmax, xmax, x0, lam):
    pars = [nmax, xmax, x0, lam]
    f1 = lambda x,p1,p2,p3,p4: gaisser_hillas_log_fithelper(x,p1,p2,p3,p4)
    f2 = lambda x,p1,p2,p3,p4:     gaisser_hillas_fithelper(x,p1,p2,p3,p4)
    #pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    nmax, xmax, x0, lam = pars
    
    return (nmax, xmax, x0, lam)


def gaisser_hillas_fit(depth, data):
    dep = depth
    dat = data

    nmax = np.max(dat)
    xmax = dep[np.where(dat == nmax)[0][0]]
    x0 = 0.0
    lam = 0.0
    
    nmax, xmax, x0, lam = fit_x0(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_lam(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_xmax(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_lam(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_x0(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_nmax_xmax(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_lam(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_x0(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_nmax_xmax_x0(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_lam(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_xmax_lam(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_nmax_xmax_x0(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_lam(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_xmax_lam(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_nmax_xmax_x0(dep, dat, nmax, xmax, x0, lam)
    
    return transform_param(nmax, xmax, x0, lam)

