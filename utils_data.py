#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code contains the functions for data-generation of dynamical systems
"""

import numpy as np
from scipy.integrate import solve_ivp

"""
Generalized stiffness matrix
"""
def kmat(k, dof):
    """
    Parameters
    ----------
    k : stiffness at each degree of freedom.
    dof : degree-of-freedom..

    Returns
    -------
    kk : stiffness matrix.
    """
    kk = np.zeros([2*dof, 2*dof])
    for i in range(2*dof):
        if i == 0:
            kk[i,1] = 1
        elif i%2 == 0:
            kk[i,i+1] = 1
        elif i == 1:
            kk[i,i-1] = -k
            kk[i,i+1] = k
        elif i == 2*dof-1:
            kk[i,i-1] = -k
            kk[i,i-3] = k
        else:
            kk[i,i+1] = k
            kk[i,i-1] = -2*k
            kk[i,i-3] = k
    return kk


"""
The Response Generation Part: Linear system:
"""
def linear(x0, tparam):
    """
    Parameters
    ----------
    x0 : vector, initial condition.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
    # Function for the dynamical systems:
    def F(t,x,params):
        k, m = params
        y = np.dot(np.array([[0, 1], [-k/m, 0]]), x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    k = 5000
    m = 10
    params = [k, m]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(params,))
    xt = np.vstack(sol.y)
    return xt, t_eval


"""
Linear system: forced excitation
"""
def linear_forced(x0, tparam):
    """
    Parameters
    ----------
    x0 : vector, initial condition.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
    # Function for the dynamical systems:
    def F(t, x, params):
        m, k, A = params
        y = np.dot(np.array([[0, 1], [-k/m, 0]]), x) + A*np.dot([0,1/m], np.sin(2*np.pi*t))
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    k = 5000
    m = 10 
    A = 10
    xt = np.vstack(x0)
    uv = []
    # Time integration:
    for i in range(len(t_eval)-1):
        tspan = [t_eval[i], t_eval[i+1]]
        params = [m, k, A] 
        sol = solve_ivp(F, tspan, x0, method='RK45', t_eval= None, args=(params,))
        solx = np.vstack(sol.y[:,-1])
        xt = np.append(xt, solx, axis=1) # -1 sign is for the last element in array
        x0 = np.ravel(solx)
    uv = np.sin(2*np.pi*t_eval)
    return xt, uv, t_eval


"""
Plane pendulum:
"""
def plane_pendulum(x0, tparam):
    """
    Parameters
    ----------
    x0 : vector, initial condition.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
    # Function for the dynamical systems:
    def F(t,x,params):
        g, l = params
        y = np.array([x[1], -(g*np.sin(x[0]))/l])
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    g = 9.81
    l = 2
    params = [g, l]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(params,))
    xt = np.vstack(sol.y)
    return xt, t_eval


"""
Atomic chain:
"""
def triatomic_molecule(x0, tparam):
    """
    Parameters
    ----------
    x0 : vector, initial condition.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
    # Function for the dynamical systems:
    def F(t,x,params):
        m1, m2, k = params
        y = np.dot(np.array([[0,1,0,0,0,0],
                             [-k/m1,0,k/m1,0,0,0],
                             [0,0,0,1,0,0],
                             [k/m2,0,-2*k/m2,0,k/m2,0],
                             [0,0,0,0,0,1],
                             [0,0,k/m1,0,-k/m1,0]]), x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    # m1, m2, k = 1.9945*10**-26, 2.6567*10**-26, 1860
    m1, m2, k = 1, 1, 1870
    params = [m1, m2, k]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(params,))
    xt = np.vstack(sol.y)
    return xt, t_eval

def atomic_chain(x0, dof, tparam):
    """
    Parameters
    ----------
    x0 : vector, initial condition.
    dof : integer, degree-of-freedom. 
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
    # Function for the dynamical systems:
    def F(t,x,params):
        y = np.dot(params, x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    m1, m2, k = 1, 1, 1870
    kk = kmat(k, dof)
    kk[1::4] = kk[1::4]/m1
    kk[3::4] = kk[3::4]/m2
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(kk,))
    xt = np.vstack(sol.y)
    return xt, t_eval


"""
3-DOF system:
"""
def mdof_system(x0, tparam):
    """
    Parameters
    ----------
    x0 : vector, initial condition.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
    # Function for the dynamical systems:
    def F(t,x,params):
        m1, m2, m3, k1, k2, k3 = params
        y = np.dot(np.array([[0,1,0,0,0,0],
                             [-(k1+k2)/m1,0,k2/m1,0,0,0],
                             [0,0,0,1,0,0],
                             [k2/m2,0,-(k2+k3)/m2,0,k3/m2,0],
                             [0,0,0,0,0,1],
                             [0,0,k3/m3,0,-k3/m3,0]]), x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    m1, m2, m3, k1, k2, k3 = 10, 10, 10, 5000, 5000, 5000
    params = [m1, m2, m3, k1, k2, k3]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(params,))
    xt = np.vstack(sol.y)
    return xt, t_eval
