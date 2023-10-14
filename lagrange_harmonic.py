#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of Harmonic Oscillator" 
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
x0 = np.array([1, 0])
dt, t0, T = 0.0001, 0, 5
tparam = [dt, t0, T]
xt, t_eval = utils_data.linear(x0, tparam)

fig1 = plt.figure(figsize=(10,8))
plt.subplot(2,1,1); plt.plot(t_eval, xt[0,:]); plt.grid(True); plt.margins(0)
plt.subplot(2,1,2); plt.plot(t_eval, xt[1,:]); plt.grid(True); plt.margins(0)

# %%
""" Generating the design matrix """

x, y = sym.symbols('x, y')
xvar = [x, y]
D, nd = utils.library_sdof(xvar, polyn=5, harmonic=1)

Dxdx, Dydt = utils.euler_lagrange(D, xvar, xt, dt)
Rl = Dydt[0] - Dxdx[0]

dxdt = Rl[:, np.where(D == y**2)]
dxdt = dxdt.reshape(1, len(dxdt))

Rl = np.delete(-1*Rl, np.where(D == y**2), axis=1)

# %%
""" Sparse regression: sequential least squares """

lam = 100      # lam is the sparsification constant
Xi = utils.sparsifyDynamics(Rl,dxdt,lam)
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
m, k = 10, 5000
H_a = 0.5*xt[1,:]**2 + 0.5*(k/m)*xt[0,:]**2
H_i = Hfun(xt[0,:], xt[1,:])

L_a = 0.5*xt[1,:]**2 - 0.5*(k/m)*xt[0,:]**2
L_i = Lfun(xt[0,:], xt[1,:])

print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22

fig2 = plt.figure(figsize=(8,5))
plt.plot(t_eval, H_a, 'b', label='Actual')
plt.plot(t_eval, H_i[0], 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([240,260])
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
ax.set_title('Harmonic Oscillator', pad=50)

ax.text(0,-0.25, str(1), color='k', rotation=90)
ax.text(1,-0.25, str(r'$x$'), color='k', rotation=90)
ax.text(2,-0.25, str(r'$\dot{x}$'), color='k', rotation=90)
ax.text(3,-0.25, str(r'$x^2$'), color='b', rotation=90)
ax.text(4,-0.25, str(r'$x \dot{x}$'), color='k', rotation=90)
ax.text(5,-0.25, str(r'$\dot{x}^2$'), color='b', rotation=90)
ax.text(6,-0.25, str(r'$x^3$'), color='k', rotation=90)
ax.text(7,-0.25, str(r'$x^2 \dot{x}$'), color='k', rotation=90)
ax.text(8,-0.25, str(r'$x \dot{x}^2$'), color='k', rotation=90)
ax.text(9,-0.25, str(r'$\dot{x}^3$'), color='k', rotation=90)
ax.text(10,-0.25, str(r'$x^4$'), color='k', rotation=90)
ax.text(11,-0.25, str(r'$x^3\dot{x}$'), color='k', rotation=90)
ax.text(12,-0.25, str(r'$x^2 \dot{x}^2$'), color='k', rotation=90)
ax.text(13,-0.25, str(r'$x \dot{x}^3$'), color='k', rotation=90)
ax.text(14,-0.25, str(r'$\dot{x}^4$'), color='k', rotation=90)
ax.text(15,-0.25, str(r'$x^5$'), color='k', rotation=90)
ax.text(16,-0.25, str(r'$x^4 \dot{x}$'), color='k', rotation=90)
ax.text(17,-0.25, str(r'$x^3 \dot{x}^2$'), color='k', rotation=90)
ax.text(18,-0.25, str(r'$x^2 \dot{x}^3$'), color='k', rotation=90)
ax.text(19,-0.25, str(r'$x \dot{x}^4$'), color='k', rotation=90)
ax.text(20,-0.25, str(r'$\dot{x}^5$'), color='k', rotation=90)
ax.text(21,-0.25, str(r'$Sin (x)$'), color='k', rotation=90)
ax.text(22,-0.25, str(r'$Sin (\dot{x})$'), color='k', rotation=90)
ax.text(23,-0.25, str(r'$Cos (x)$'), color='k', rotation=90)
ax.text(24,-0.25, str(r'$Cos (\dot{x})$'), color='k', rotation=90)
