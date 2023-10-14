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

"""
Code for string vibration in FINITE DIFFERENCE
"""
def string(L,stopTime,c,dx=0.01,dt=0.001):    
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


"""
Code for channel flow using Navier-Stokes equation
    - Note the presence of the pseudo-time variable nit. 
    - This sub-iteration in the Poisson calculation helps ensure a divergence-free field.
"""
nit = 100

def build_up_b_channel(rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    
    # Periodic BC Pressure @ x = 2
    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                          2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                               (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

    # Periodic BC Pressure @ x = 0
    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 -
                         2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                              (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))-
                         ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))
    return b

def pressure_poisson_periodic_channel(p, dx, dy, b):
    pn = np.zeros_like(p)
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Periodic BC Pressure @ x = 2
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2 +
                        (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) -
                       dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

        # Periodic BC Pressure @ x = 0
        p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2 +
                       (pn[2:, 0] + pn[0:-2, 0]) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])
        
        # Wall boundary conditions, pressure
        p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
    
    return p

def channel_flow(nt, u, v, dt, dx, dy, p, rho, nu, F):   
    ut = np.zeros((u.shape[0], u.shape[1], nt))
    vt = np.zeros((v.shape[0], v.shape[1], nt))
    pt = np.zeros((p.shape[0], p.shape[1], nt))

    for n in range(nt):
        un = u
        vn = v

        b = build_up_b_channel(rho, dt, dx, dy, u, v)
        p = pressure_poisson_periodic_channel(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                               dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + 
                         F[1:-1, 1:-1] * dt)
    
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                               dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
    
        # Periodic BC u @ x = 2     
        u[1:-1, -1] = (un[1:-1, -1] - 
                       un[1:-1, -1] * dt / dx * (un[1:-1, -1] - un[1:-1, -2]) -
                       vn[1:-1, -1] * dt / dy * (un[1:-1, -1] - un[0:-2, -1]) -
                       dt / (2 * rho * dx) * (p[1:-1, 0] - p[1:-1, -2]) + 
                       nu * (dt / dx**2 * (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                       dt / dy**2 * (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F[1:-1, -1] * dt)
    
        # Periodic BC u @ x = 0
        u[1:-1, 0] = (un[1:-1, 0] - 
                      un[1:-1, 0] * dt / dx * (un[1:-1, 0] - un[1:-1, -1]) -
                      vn[1:-1, 0] * dt / dy * (un[1:-1, 0] - un[0:-2, 0]) - 
                      dt / (2 * rho * dx) * (p[1:-1, 1] - p[1:-1, -1]) + 
                      nu * (dt / dx**2 * (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                      dt / dy**2 * (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F[1:-1, 0] * dt)
    
        # Periodic BC v @ x = 2
        v[1:-1, -1] = (vn[1:-1, -1] - 
                       un[1:-1, -1] * dt / dx * (vn[1:-1, -1] - vn[1:-1, -2]) - 
                       vn[1:-1, -1] * dt / dy * (vn[1:-1, -1] - vn[0:-2, -1]) -
                       dt / (2 * rho * dy) * (p[2:, -1] - p[0:-2, -1]) +
                       nu * (dt / dx**2 * (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                       dt / dy**2 * (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))
    
        # Periodic BC v @ x = 0
        v[1:-1, 0] = (vn[1:-1, 0] - 
                      un[1:-1, 0] * dt / dx * (vn[1:-1, 0] - vn[1:-1, -1]) -
                      vn[1:-1, 0] * dt / dy * (vn[1:-1, 0] - vn[0:-2, 0]) -
                      dt / (2 * rho * dy) * (p[2:, 0] - p[0:-2, 0]) +
                      nu * (dt / dx**2 * (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                      dt / dy**2 * (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))
    
        # Wall BC: u,v = 0 @ y = 0,2
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :]= 0
        
        ut[..., n] = u
        vt[..., n] = v
        pt[..., n] = p
    
    return ut, vt, pt
