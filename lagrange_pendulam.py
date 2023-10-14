#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of Pendulum" 
"""

import numpy as np
import matplotlib.pyplot as plt
import utils_data
import utils
import sympy as sym
import seaborn as sns

# %%
""" Generating system response """

# The time parameters:
x0 = np.array([0.5, 0])
dt, t0, T = 0.001, 0, 10
tparam = [dt, t0, T]
xt, t_eval = utils_data.plane_pendulum(x0, tparam)

fig1 = plt.figure(figsize=(10,8))
plt.subplot(2,1,1); plt.plot(t_eval, xt[0,:]); plt.grid(True); plt.margins(0)
plt.subplot(2,1,2); plt.plot(t_eval, xt[1,:]); plt.grid(True); plt.margins(0)

# %%
""" Generating the design matrix """

x, y = sym.symbols('x, y')
xvar = [x, y]
D, nd = utils.library_sdof(xvar, polyn=5, harmonic=1)

Rl, dxdt = utils.euler_lagrange_library(D,xvar,xt,dt)

# %%
""" Sparse regression: sequential least squares """

lam = 5      # lam is the sparsification constant
Xi = utils.sparsifyDynamics(-1*Rl[0], dxdt, lam)
Xi = np.insert(Xi,np.where(D == y**2)[0], 1, axis=0)
print(Xi)

""" Lagrangian """
L = sym.Array(0.5*np.dot(D,Xi))
H = (sym.diff(L, y)*y - L)
print('Lagrangian: %s, Hamiltonian: %s' % (L, H))

Lfun = sym.lambdify([x,y], L, 'numpy') 
Hfun = sym.lambdify([x,y], H, 'numpy')

# %%
""" Hamiltonian """
l, g = 2, 9.81
H_a = 0.5*xt[1,:]**2 + (g/l)*np.cos(xt[0,:])
H_i = Hfun(xt[0,:], xt[1,:])

L_a = 0.5*xt[1,:]**2 - (g/l)*np.cos(xt[0,:])
L_i = Lfun(xt[0,:], xt[1,:])

print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22
t_eval = np.linspace(0,T,len(xt[0]))

fig2 = plt.figure(figsize=(8,5))
plt.plot(t_eval, H_a, 'b', label='Actual')
plt.plot(t_eval, H_i[0], 'r--', linewidth=2, label='Identified')
plt.title('Pendulum')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([-10,10])
plt.grid(True)
plt.legend()
plt.margins(0)

# %%
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 16

Xi_plot = Xi.T
Xi_plot[Xi_plot != 0 ] = -1

fig3, ax = plt.subplots(figsize=(10,0.5))
ax = sns.heatmap( Xi_plot, linewidth=0.5, cmap='Set1', cbar=False, yticklabels=[0])
ax.set_xlabel('Library functions')
ax.set_ylabel('DOF')
ax.set_title('(c) Pendulum', pad=50)

ax.text(0,-0.25, str(1), color='k', rotation=90)
ax.text(1,-0.25, str(r'$\theta$'), color='k', rotation=90)
ax.text(2,-0.25, str(r'$\dot{\theta}$'), color='k', rotation=90)
ax.text(3,-0.25, str(r'$\theta^2$'), color='k', rotation=90)
ax.text(4,-0.25, str(r'$\theta \dot{\theta}$'), color='k', rotation=90)
ax.text(5,-0.25, str(r'$\dot{\theta}^2$'), color='b', rotation=90)
ax.text(6,-0.25, str(r'$\theta^3$'), color='k', rotation=90)
ax.text(7,-0.25, str(r'$\theta^2 \dot{\theta}$'), color='k', rotation=90)
ax.text(8,-0.25, str(r'$\theta \dot{\theta}^2$'), color='k', rotation=90)
ax.text(9,-0.25, str(r'$\dot{\theta}^3$'), color='k', rotation=90)
ax.text(10,-0.25, str(r'$\theta^4$'), color='k', rotation=90)
ax.text(11,-0.25, str(r'$\theta^3\dot{\theta}$'), color='k', rotation=90)
ax.text(12,-0.25, str(r'$\theta^2 \dot{\theta}^2$'), color='k', rotation=90)
ax.text(13,-0.25, str(r'$\theta \dot{\theta}^3$'), color='k', rotation=90)
ax.text(14,-0.25, str(r'$\dot{\theta}^4$'), color='k', rotation=90)
ax.text(15,-0.25, str(r'$\theta^5$'), color='k', rotation=90)
ax.text(16,-0.25, str(r'$\theta^4 \dot{\theta}$'), color='k', rotation=90)
ax.text(17,-0.25, str(r'$\theta^3 \dot{\theta}^2$'), color='k', rotation=90)
ax.text(18,-0.25, str(r'$\theta^2 \dot{\theta}^3$'), color='k', rotation=90)
ax.text(19,-0.25, str(r'$\theta \dot{\theta}^4$'), color='k', rotation=90)
ax.text(20,-0.25, str(r'$\dot{\theta}^5$'), color='k', rotation=90)
ax.text(21,-0.25, str(r'$Sin (\theta)$'), color='k', rotation=90)
ax.text(22,-0.25, str(r'$Sin (\dot{\theta})$'), color='k', rotation=90)
ax.text(23,-0.25, str(r'$Cos (\theta)$'), color='b', rotation=90)
ax.text(24,-0.25, str(r'$Cos (\dot{\theta})$'), color='k', rotation=90)

