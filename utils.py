#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code contains the useful functions for Lagrange discovery.
   The code is associated with...
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty
"""

import sympy as sym
import numpy as np

"""
Sparse-Least square regression
"""
def sparsifyDynamics(library,target,lam,iteration=10):
    """
    It performs the least-squares sparse-regression. 
    
    Parameters
    ----------
    library : matrix, the design matrix of candidate basis functions.
    target : vector, the target vector.
    lam : scalar, the sparsification constant.
    iteration : integer, number of sequential threshold iterations.

    Returns
    -------
    Xi : vector, the sparse parameter vector.
    """
    Xi = np.matmul(np.linalg.pinv(library), target.T) # initial guess: Least-squares
    for k in range(iteration):
        smallinds = np.where(abs(Xi) < lam)   # find small coefficients
        Xi[smallinds] = 0
        for ind in range(Xi.shape[1]):
            biginds = np.where(abs(Xi[:,ind]) > lam)
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.matmul(np.linalg.pinv(library[:, biginds[0]]), target[ind, :].T) 
    return Xi

"""
Euler-Lagrangian operator on library
"""
def euler_lagrange(D,xvar,xt,dt):
    """
    It obtains the Euler-Lagrange library by evaluating the Euler-Lagrange 
    operator on the actual Lagrange library matrix, 
    !! For SDOF systems !! 
    However, it does not provide the target vector directly, further preprocessing is required.
    
    Parameters
    ----------
    D : matrix, the Lagrange matrix.
    xvar : symbolic vector, the variables.
    xt : vector, the numerical responses at the variables 'xvar'.
    dt : scalar, the time step.

    Returns
    -------
    Dxdx : matrix, the spatial derivatives.
    Dydt : matrix, the temporal derivatives.
    """
    Drx = []
    for kk in range(len(xvar)):
        fun = [sym.diff(D[i], xvar[kk]) for i in range(len(D))]
        Drx.append(fun)
        
    diff_Drx = []
    for kk in range(len(xvar)):
        fun = sym.lambdify([xvar], Drx[kk], 'numpy')
        diff_Drx.append(fun)
    
    Dxdx = []
    for kk in range(len(xvar)):
        Dxtemp = np.zeros([xt.shape[1], D.shape[0]])
        for i in range(xt.shape[1]):
            Dxtemp[i,:] = diff_Drx[kk](xt[0:len(xvar),i])
        Dxdx.append(Dxtemp)
    
    momentum_index = xvar[1::2] # every second component
    momentum_library = Dxdx[1::2]
    Dydt = []
    nd = len(D)
    for j in range(len(momentum_index)):
        Dydt_temp = np.zeros((Dxdx[0].shape[0]-1, Dxdx[0].shape[1]))
        for i in range(nd):
            temp = FiniteDiff(momentum_library[j][:,i],dt,1)
            Dydt_temp[:,i] = temp[1:]
        Dydt.append(Dydt_temp)
        
    for kk in range(len(xvar)): # correct the shape
        Dxdx[kk] = Dxdx[kk][1:, :]
    return Dxdx, Dydt

def euler_lagrange_library(D,xvar,xt,dt):
    """
    It obtains the Euler-Lagrange library by evaluating the Euler-Lagrange 
    operator on the actual Lagrange library matrix, 
    And, provides the target vector by removing the squared velocity terms
    from the design matrix, as explained in the paper.
    
    Parameters
    ----------
    D : matrix, the Lagrange matrix.
    xvar : symbolic vector, the variables.
    xt : vector, the numerical responses at the variables 'xvar'.
    dt : scalar, the time step.

    Returns
    -------
    Rl : matrix, the Euler-Lagrange library obtained after removing ith squared 
    velocity terms.
    Dydt : matrix, the collection of the squared velocity terms from 
    Euler-Lagrange library.
    """
    # Derivative with respect to displacment (change in potential energy):
    Drx = []
    for kk in range(len(xvar)):
        fun = [sym.diff(D[i], xvar[kk]) for i in range(len(D))]
        Drx.append(fun)
    
    diff_Drx = []
    for kk in range(len(xvar)):
        fun = sym.lambdify([xvar], Drx[kk], 'numpy')
        diff_Drx.append(fun)
        
    # Compute the numerial library:
    Dxdx = []
    for j in range(len(xvar)):
        Dxtemp = np.zeros([xt.shape[1], D.shape[0]])
        for i in range(xt.shape[1]):
            Dxtemp[i,:] = diff_Drx[j](xt[0:len(xvar),i])
        Dxdx.append(Dxtemp)
    
    momentum_index = xvar[1::2] # every second component
    momentum_library = Dxdx[1::2]    
    Dydt = []
    nd = len(D)
    for j in range(len(momentum_index)):
        Dydt_temp = np.zeros((xt.shape[1], nd))
        for i in range(nd):
            Dydt_temp[:,i] = FiniteDiff(momentum_library[j][:,i],dt,1)
        Dydt.append(Dydt_temp)
        
    Dxdx = Dxdx[0::2]    
    Rl = []
    for i in range(len(momentum_index)):
        Rl.append(Dydt[i] + Dxdx[i][:,:])
    
    dxdt = np.zeros([len(momentum_index),xt.shape[1]])
    for i in range(len(momentum_index)):
        dxdt[i,:] = Rl[i][:, np.where(D == momentum_index[i]**2)].squeeze()
     
    for i in range(len(momentum_index)):
        Rl[i] = np.delete(Rl[i], np.where(D == momentum_index[i]**2), axis=1)
    
    return Rl, dxdt


"""
The Dictionary creation part:
"""
def library_sdof(xt,polyn,harmonic,force=None):
    """
    Obtains the Lagrangian library from the system responses (for SDOF ODEs)
    
    Parameters
    ----------
    xt : vector, the numerical responses.
    polyn : scalar, the polynomial order.
    harmonic : boolean function, if 1 adds harmonic functions in the library.
    force : vector, the force vector, if available.

    Returns
    -------
    D : matrix, the design matrix Lagrangian.
    ind : scalar, the dimension of the candidate basis functions.
    """
    # poly order 0
    D = 1            
    if polyn >= 1:
        # poly order 1
        for i in range(len(xt)):
            new = xt[i]
            D = np.append(D, new) 
    if polyn >= 2: 
        # ploy order 2
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                new = np.multiply(xt[i], xt[j])
                D = np.append(D, new) 
    if polyn >= 3:    
        # ploy order 3
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                for k in  range(j,len(xt)):
                    new = np.multiply(np.multiply(xt[i], xt[j]), xt[k])
                    D = np.append(D, new) 
    if polyn >= 4:
        # ploy order 4
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                for k in  range(j,len(xt)):
                    for l in range(k,len(xt)):
                        new = np.multiply(np.multiply(xt[i], xt[j]), xt[k])
                        new = np.multiply(new, xt[l])
                        D = np.append(D, new) 
    if polyn >= 5:
        # ploy order 5
        for i in range(len(xt)):
            for j in  range(i,len(xt)):
                for k in  range(j,len(xt)):
                    for l in  range(k,len(xt)):
                        for m in  range(l,len(xt)):
                            new = np.multiply(xt[i], xt[j])
                            new = np.multiply(new, xt[k])
                            new = np.multiply(new, xt[l])
                            new = np.multiply(new, xt[m])
                            D = np.append(D, new) 
    # for i in range(len(xt)):
    if harmonic == 1:
        # for sin(x)
        for i in range(len(xt)):
            new = sym.sin(xt[i])
            D = np.append(D, new)      
        # for cos(x)
        for i in range(len(xt)):
            new = sym.cos(xt[i])
            D = np.append(D, new)
    # for force vector:
    if force:
        for i in range(len(xt)):
            D = np.append(D, force*xt[i])
    ind = len(D)
    return D, ind

def library_mdof(xt,polyn,funofvelocity=None,harmonic=1):
    """
    Obtains the Lagrangian library from the system responses (for MDOF ODEs)
    
    Parameters
    ----------
    xt : vector, the numerical responses.
    polyn : scalar, the polynomial order.
    funofvelocity : boolean, if 1 adds basis functions for velocity terms,
                    not recommended.
    harmonic : boolean function, if 1 adds harmonic functions in the library.

    Returns
    -------
    D : matrix, the design matrix Lagrangian.
    ind : scalar, the dimension of the candidate basis functions.
    """
    xdisp = xt[::2]
    xvel = xt[1::2]
    # poly order 0
    D = 1            
    if len(xt) >= 1: # comment this function for noise -- atom
        # states
        for i in range(len(xt)):
            new = xt[i]
            D = np.append(D, new) 
        # square
        for i in range(len(xt)):
            new = xt[i]**2
            D = np.append(D, new) 
    for p in range(polyn):
        # difference -- poly order p
        for i in range(len(xdisp)):
            for j in range(i+1, len(xdisp)):
                new = (xdisp[j]-xdisp[i])**(p+1)
                D = np.append(D, new)
        if funofvelocity:
            for i in range(len(xvel)):
                for j in range(i+1, len(xvel)):
                    new = (xvel[j]-xvel[i])**(p+1)
                    D = np.append(D, new)
    if harmonic == 1:
        for i in range(len(xdisp)):
            for j in range(i+1, len(xdisp)):
                new = sym.sin(xdisp[j]-xdisp[i])
                D = np.append(D, new)
        for i in range(len(xvel)):
            for j in range(i+1, len(xvel)):
                new = sym.cos(xvel[j]-xvel[i])
                D = np.append(D, new)
    
    ind = len(D)
    return D, ind

def library_pde(xt,Type,dx,polyn=4):
    """
    Obtains the Lagrangian library from the system responses (for PDEs)
    
    Parameters
    ----------
    xt : vector, the numerical responses.
    Type : string, the order of PDE.
           order1 = first or second order systems, like wave equation 
           order1 = fourth-order systems, like Euler-Bernoulli equation
    dx : scalar, the grid spacing.
    polyn : scalar, the polynomial order.

    Returns
    -------
    D : matrix, the design matrix Lagrangian.
    ind : scalar, the dimension of the candidate basis functions.
    """
    xt_disp = xt[::2]
    xt_vel = xt[1::2]
    # poly order 0
    D = 1            
    # square
    for i in range(len(xt_vel)):
        new = xt_vel[i]**2
        D = np.append(D, new) 
    if Type == 'order1':
        if polyn:
            # difference -- poly order 1
            D = np.append(D, xt_disp[0]/dx)
            for i in range(1,len(xt_disp)-1):
                new = (xt_disp[i+1] - xt_disp[i])/dx
                D = np.append(D, new)
            D = np.append(D, -1*xt_disp[-1]/dx)
    
        if polyn >= 2:
            # difference -- poly order 2
            D = np.append(D, xt_disp[0]**2/(dx**2))
            for i in range(len(xt_disp)-1):
                new = (xt_disp[i+1] - xt_disp[i])**2/(dx**2)
                D = np.append(D, new)
            D = np.append(D, xt_disp[-1]**2/(dx**2))
            
        if polyn >= 3:
            # difference -- poly order 3
            D = np.append(D, xt_disp[0]**3/(dx**3))
            for i in range(len(xt_disp)-1):
                new = (xt_disp[i+1] - xt_disp[i])**3/(dx**3)
                D = np.append(D, new)
            D = np.append(D, xt_disp[-1]**3/(dx**3))
            
        if polyn >= 4:
            # difference -- poly order 4
            D = np.append(D, xt_disp[0]**4/(dx**4))
            for i in range(len(xt_disp)-1):
                new = (xt_disp[i+1] - xt_disp[i])**4/(dx**4)
                D = np.append(D, new)
            D = np.append(D, xt_disp[-1]**4/(dx**4))
                    
    if Type == 'order2':
        if polyn:
            # difference -- poly order 1
            D = np.append(D, xt_disp[0]/dx)
            for i in range(1,len(xt_disp)-1):
                new = (xt_disp[i+1] - xt_disp[i])/dx
                D = np.append(D, new)
            D = np.append(D, -1*xt_disp[-1]/dx)
            
            # difference -- poly order 1
            D = np.append(D, xt_disp[0]/dx**2)
            D = np.append(D, (xt_disp[1] - 2*xt_disp[0])/(dx**2))
            for i in range(1, len(xt_disp)-1):
                new = (xt_disp[i+1] - 2*xt_disp[i] + xt_disp[i-1])/(dx**2)
                D = np.append(D, new)
            D = np.append(D, (-2*xt_disp[-1] + xt_disp[-2])/(dx**2))
            D = np.append(D, xt_disp[-1]/dx**2)
            
        if polyn >= 2:
            # difference -- poly order 2
            D = np.append(D, xt_disp[0]**2/dx**4)
            D = np.append(D, (xt_disp[1] - 2*xt_disp[0])**2/(dx**4))
            for i in range(1, len(xt_disp)-1):
                new = (xt_disp[i+1] - 2*xt_disp[i] + xt_disp[i-1])**2/(dx**4)
                D = np.append(D, new)
            D = np.append(D, (-2*xt_disp[-1] + xt_disp[-2])**2/(dx**4))
            D = np.append(D, xt_disp[-1]**2/dx**4)
            
        if polyn >= 3:
            # difference -- poly order 2
            D = np.append(D, (xt_disp[1] - 2*xt_disp[0])**3/(dx**6))
            for i in range(1, len(xt_disp)-1):
                new = (xt_disp[i+1] - 2*xt_disp[i] + xt_disp[i-1])**3/(dx**6)
                D = np.append(D, new)
            D = np.append(D, (-2*xt_disp[-1] + xt_disp[-2])**3/(dx**6))
            
        if polyn >= 4:
            # difference -- poly order 2
            D = np.append(D, (xt_disp[1] - 2*xt_disp[0])**4/(dx**8))
            for i in range(1, len(xt_disp)-1):
                new = (xt_disp[i+1] - 2*xt_disp[i] + xt_disp[i-1])**4/(dx**8)
                D = np.append(D, new)
            D = np.append(D, (-2*xt_disp[-1] + xt_disp[-2])**4/(dx**8))        
    ind = len(D)
    return D, ind

"""
For numerical derivative using 4th order accuarcy
"""
def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3
    
    Parameters
    ----------
    u : vector, data to be differentiated
    dx : scalar, Grid spacing.  Assumes uniform spacing
    d : order of derivative
    
    Returns
    -------
    ux : vector, the derivative vector
    """
    
    n = u.size
    ux = np.zeros(n)
    
    if d == 1:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-u[i-1]) / (2*dx)
        
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux
    
    if d == 2:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2
        
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux
    
    if d == 3:
        for i in range(2,n-2):
            ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3
        
        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux
    
    if d > 3:
        return FiniteDiff(FiniteDiff(u,dx,3), dx, d-3)

def PolyDiff(u, x, deg = 3, diff = 1, width = 5):
    """
    It evaluates the derivative using polynomial fit
    
    Parameters
    ----------
    u : vector, values of some function
    x : vector, x-coordinates where values are known
    deg : integer, degree of polynomial to use
    diff : integer, maximum order derivative we want
    width : integer, width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    
    Returns
    -------
    du : vector, the derivative 
    See https://github.com/snagcliffs/PDE-FIND/blob/master/PDE_FIND.py
    """
    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2*width,diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        # Note code originally used an even number of points here.
        # This is an oversight in the original code fixed in 2022.
        points = np.arange(j - width, j + width + 1)

        # Fit to a polynomial
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives
        for d in range(1,diff+1):
            du[j-width, d-1] = poly.deriv(m=d)(x[j])
    return du

"""
Non equal sum
"""
def nonequal_sum(mat):
    """
    It removes the identical basis functions from the parameter matrix
    
    Parameters
    ----------
    mat : matrix, the observation matrix.

    Returns
    -------
    The matrix with identical basis functions removed.
    """
    for i in range(1, mat.shape[1]):
        mat[ mat[:,i-1] !=0, i ] = 0
    return np.array(mat)

def reshape(x, shape_0, shape_1):
    """
    It reshapes a matrix to a vector
    
    Parameters
    ----------
    x : matrix, the matrix to be reshaped.
    shape_0 : integer, first dimension of shape.
    shape_1 : integer, second dimension of shape.

    Returns
    -------
    Vector, the reshaped matrix
    """
    var = []
    for i in range(shape_0):
        for j in range(shape_1):
            var.append(x[i,j,:])
    return np.array(var)

def rebuild(x, shape_0, shape_1):
    """
    It reconstruct a matrix from a vector
    
    Parameters
    ----------
    x : vector, the vector to be converted to matrix.
    shape_0 : first dimension of reconstructed matrix.
    shape_1 : second dimension of reconstructed matrix.

    Returns
    -------
    var : the retrieved matrix 
    """
    var = np.zeros((shape_0, shape_1, x.shape[-1]))
    for i in range(shape_0):
        for j in range(shape_1):
            var[i,j,:] = x[(shape_0*i) + (shape_1-(shape_1-j)), :]
    return var
    
