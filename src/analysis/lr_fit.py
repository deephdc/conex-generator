"""Do note that this module has not been tested properly and is in a
work-in-progress state"""

import numpy as np
import scipy as sp
import scipy.optimize as opt


param_names = [
        "nmax",
        "xmax",
        "lparam",
        "rparam"
]


def transform_param(p1, p2, p3, p4):
    nmax = np.abs(p1)
    xmax = np.abs(p2)
    lparam = p3
    rparam = p4

    return (nmax, xmax, lparam, rparam)

def transform_param_inv(nmax, xmax, lparam, rparam):
    p1 = np.abs(nmax)
    p2 = np.abs(xmax)
    p3 = lparam
    p4 = rparam

    return (p1, p2, p3, p4)


def lr_curve(x, nmax, xmax, lparam, rparam):
    xp = x - xmax
    return nmax \
           * np.power(1+ rparam*xp/lparam, rparam**-2) \
           * np.exp(-xp/lparam/rparam)

def lr_curve_log(x, nmax, xmax, lparam, rparam):
    xp = x - xmax
    return np.log(nmax) \
           + rparam**-2 * np.log(1+ rparam*xp/lparam) \
           - xp/lparam/rparam


def lr_curve_fithelper(x, p1, p2, p3, p4):
    nmax, xmax, lparam, rparam = transform_param(p1,p2,p3,p4)
    return lr_curve(x, nmax, xmax, lparam, rparam)

def lr_curve_log_fithelper(x, p1, p2, p3, p4):
    nmax, xmax, lparam, rparam = transform_param(p1,p2,p3,p4)
    return lr_curve_log(x, nmax, xmax, lparam, rparam)

def create_fit_lambda(func, param, startval):
    index_list = []
    jj = 0
    for name in param_names:
        if name not in param:
            index_list.append(None)
        else:
            index_list.append(param.index(name))
            jj += 1

    print([startval[ii] if name not in param else [index_list[ii]] for ii,name in enumerate(param_names)])

    f = lambda x,*args: func(x, *[startval[ii] if name not in param else args[index_list[ii]] for ii,name in enumerate(param_names)])
    return f

def get_initial_fit_pars(param, startval):
    return [startval[param_names.index(name)] for name in param]

def get_final_fit_result(param, startval, pars):
    index_list = []
    jj = 0
    for name in param_names:
        if name not in param:
            index_list.append(None)
        else:
            index_list.append(param.index(name))
            jj += 1

    return [startval[ii] if name not in param else pars[index_list[ii]] for ii,name in enumerate(param_names)]

def fit_param(dep, dat, param, startval):
    assert len(startval) == len(param_names)

    f1 = create_fit_lambda(lr_curve_fithelper, param, startval)
    f2 = create_fit_lambda(lr_curve_log_fithelper, param, startval)

    pars = get_initial_fit_pars(param, startval)
    pars = opt.curve_fit(f1, dep, np.log(dat), p0=pars)[0].tolist()
    pars = opt.curve_fit(f2, dep, dat,         p0=pars)[0].tolist()
    
    return get_final_fit_result(param, startval, pars)


import gaisser_hillas_fit as gs
import matplotlib.pyplot as plt

x = np.linspace(300,1500,1000)
y = gs.gaisser_hillas(x, 1, 700, 0, 50)
res = fit_param(x,y,["rparam", "lparam"],[1,700,200,0.1])

