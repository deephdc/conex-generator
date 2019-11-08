import numpy as np
import scipy as sp
import scipy.optimize as opt


# bounds
bound_x0_upper = 200.0
bound_x0_lower = -200.0
scale_x0 = 1000.0
zero_x0 = -100.0

bound_lam_upper = 1000.0
bound_lam_lower = 1.0
scale_lam = 1000.0
zero_lam = 50.0

# preprocessing
range_x0 = bound_x0_upper - bound_x0_lower
shift_x0 = range_x0 - bound_x0_upper
inverse_x0 = np.arctanh((zero_x0 + shift_x0)*2.0/range_x0 - 1.0)*scale_x0

range_lam = bound_lam_upper - bound_lam_lower
shift_lam = range_lam - bound_lam_upper
inverse_lam = np.arctanh((zero_lam + shift_lam)*2.0/range_lam - 1.0)*scale_lam


def transform_param(p1, p2, p3, p4):
    nmax = np.abs(p1)
    xmax = np.abs(p2)
    x0 =  range_x0  * (np.tanh((p3 + inverse_x0) /scale_x0) +1.0)/2.0 - shift_x0
    lam = range_lam * (np.tanh((p4 + inverse_lam)/scale_lam)+1.0)/2.0 - shift_lam

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
    #pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    lam = pars[0]
    
    return (nmax, xmax, x0, lam)

def fit_xmax(dep, dat, nmax, xmax, x0, lam):
    pars = [xmax]
    f1 = lambda x,p2: gaisser_hillas_log_fithelper(x,nmax,p2,x0,lam)
    f2 = lambda x,p2:     gaisser_hillas_fithelper(x,nmax,p2,x0,lam)
    #pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
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
    #pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    xmax, lam = pars
    
    return (nmax, xmax, x0, lam)

def fit_nmax_xmax(dep, dat, nmax, xmax, x0, lam):
    pars = [nmax, xmax]
    f1 = lambda x,p1,p2: gaisser_hillas_log_fithelper(x,p1,p2,x0,lam)
    f2 = lambda x,p1,p2:     gaisser_hillas_fithelper(x,p1,p2,x0,lam)
    #pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
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

def fit_nmax_xmax_lam(dep, dat, nmax, xmax, x0, lam):
    pars = [nmax, xmax, lam]
    f1 = lambda x,p1,p2,p4: gaisser_hillas_log_fithelper(x,p1,p2,x0,p4)
    f2 = lambda x,p1,p2,p4:     gaisser_hillas_fithelper(x,p1,p2,x0,p4)
    #pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    nmax, xmax, lam = pars
    
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
    ind = np.where(depth >= bound_x0_upper + 50.0)[0]
    dep = depth[ind]
    dat = data[ind]

    nmax = np.max(dat)
    xmax = dep[np.where(dat == nmax)[0][0]]
    x0 =  0.0
    lam = -100.0
    
    nmax, xmax, x0, lam = fit_lam(dep, dat, nmax, xmax, x0, lam)

    nmax, xmax, x0, lam = fit_x0(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_lam(dep, dat, nmax, xmax, x0, lam)
    #nmax, xmax, x0, lam = fit_x0_lam(dep, dat, nmax, xmax, x0, lam)

    nmax, xmax, x0, lam = fit_xmax(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_lam(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_xmax_lam(dep, dat, nmax, xmax, x0, lam)

    nmax, xmax, x0, lam = fit_x0(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_lam(dep, dat, nmax, xmax, x0, lam)
    #nmax, xmax, x0, lam = fit_x0_lam(dep, dat, nmax, xmax, x0, lam)

    nmax, xmax, x0, lam = fit_nmax_xmax(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_lam(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_xmax_lam(dep, dat, nmax, xmax, x0, lam)
    nmax, xmax, x0, lam = fit_nmax_xmax_lam(dep, dat, nmax, xmax, x0, lam)

    nmax, xmax, x0, lam = fit_nmax_xmax_x0_lam(dep, dat, nmax, xmax, x0, lam)
    
    return transform_param(nmax, xmax, x0, lam)

