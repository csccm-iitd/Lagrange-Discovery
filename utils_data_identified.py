#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code contains the functions for data-generation of dynamical systems
"""

import numpy as np
import beam3fun
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
    k = 500.1407*10
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
    k = 499.997*10
    m = 10 
    A = 2.005*10/2
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
    g = 9.817113
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
    m1, m2, k = 1, 1, 1870.46
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
    val = 500.1033*10
    m1, m2, m3, k1, k2, k3 = 10, 10, 10, val, val, val
    params = [m1, m2, m3, k1, k2, k3]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(params,))
    xt = np.vstack(sol.y)
    return xt, t_eval


"""
Code for string vibration in FINITE DIFFERENCE
"""
def string(L,stopTime,c=24.9839,dx=0.01,dt=0.001):    
    # dx = 0.01     # Spacing of points on string
    # dt = 0.001    # Size of time step
    # c = 5         # Speed of wave propagation
    # L = 10        # Length of string
    # stopTime = 5  # Time to run the simulation
    
    r = c*dt/dx 
    n = int(L/dx + 1)
    t  = np.arange(0, stopTime+dt, dt)
    mesh = np.arange(0, L+dt, dx)
    sol = np.zeros([len(mesh), len(t)])
    
    # Set current and past to the graph of a plucked string
    current = 0.1 - 0.1*np.cos( 2*np.pi/L*mesh ) 
    past = current
    sol[:, 0] = current
    
    for i in range(len(t)):
        future = np.zeros(n)
    
        # Calculate the future position of the string
        future[0] = 0 
        future[1:n-2] = r**2*( current[0:n-3]+ current[2:n-1] ) + 2*(1-r**2)*current[1:n-2] - past[1:n-2]
        future[n-1] = 0 
        sol[:, i] = current
        
        # Settings up for the next time step
        past = current 
        current = future 
    
    Vel = np.zeros([sol.shape[0], sol.shape[1]])
    for i in range(1, sol.shape[1]-1):
        Vel[:,i] = (sol[:,i+1] - sol[:,i-1])/(2*dt)
    Vel[:,0] = (-3.0/2*sol[:,0] + 2*sol[:,1] - sol[:,2]/2) / dt
    Vel[:,sol.shape[1]-1] = (3.0/2*sol[:,sol.shape[1]-1] - 2*sol[:,sol.shape[1]-2] + sol[:,sol.shape[1]-3]/2) / dt
    
    xt = np.zeros([2*sol.shape[0],sol.shape[1]])
    xt[::2] = sol
    xt[1::2] = Vel

    return xt


"""
Codes for free vibration of a cantilever
"""
def cantilever(params,T,dt,Ne=100):
    """
    Parameters
    ----------
    params : list, the system parameters.
    T : scalar, terminal time.
    dt : float, time step.
    Ne : integer, numver of finite element. The default is 100.

    Returns
    -------
    Dis : matrix, displacement.
    Vel : matrix, velocity.
    Acc : matrix, acceleration.
    """
    rho, b, d, A, L, E, I = params
    c1 = 0
    c2 = 0
    xx = np.arange(1/Ne, 1+1/Ne, 1/Ne)
    
    [Ma, Ka, _, _] = beam3fun.Beam3(rho,A,E,I,L/Ne,Ne+1,'cantilever')

    Ca = (c1*Ma + c2*Ka)
    F = lambda t : 0             # free vibration
    # % for forced vibration e.g. F = lambda t: np.sin(2*t)
    
    # % ------------------------------------------------
    Lambda = 1.875104069/L
    # Lambda = 4.694091133/L
    # Lambda = 7.854757438/L
    # Lambda = 10.99554073/L
    # Lambda = 14.13716839/L
    # Lambda = 17.27875953/L

    h1 = np.cosh(Lambda*xx) -np.cos(Lambda*xx) -(np.cos(Lambda*L)+np.cosh(Lambda*L)) \
        /(np.sin(Lambda*L)+np.sinh(Lambda*L))*(np.sinh(Lambda*xx)-np.sin(Lambda*xx))
    h2 = Lambda*(np.sinh(Lambda*xx)+np.sin(Lambda*xx))-(np.cos(Lambda*L)+np.cosh(Lambda*L)) \
        /(np.sin(Lambda*L)+np.sinh(Lambda*L))*(np.cosh(Lambda*xx)-np.cos(Lambda*xx))*Lambda
    
    D0 = np.zeros(2*Ne)
    D0[0::2] = h1
    D0[1::2] = h2
    V0 = np.zeros(2*Ne)
    D, V, A = beam3fun.Newmark(Ma,Ca,Ka,F,D0,V0,dt,T)
    
    # % -------------------------------------------------
    Dis = D[0::2]
    # Vel = V[0::2]
    Acc = A[0::2]
    
    Vel = np.zeros([Dis.shape[0], Dis.shape[1]])
    for i in range(1, Dis.shape[1]-1):
        Vel[:,i] = (Dis[:,i+1] - Dis[:,i-1])/(2*dt)
    Vel[:,0] = (-3.0/2*Dis[:,0] + 2*Dis[:,1] - Dis[:,2]/2) / dt
    Vel[:,Dis.shape[1]-1] = (3.0/2*Dis[:,Dis.shape[1]-1] - 2*Dis[:,Dis.shape[1]-2] + Dis[:,Dis.shape[1]-3]/2) / dt
    
    xt = np.zeros([2*Dis.shape[0],Dis.shape[1]])
    xt[::2] = Dis
    xt[1::2] = Vel
        
    return Dis, Vel, Acc
