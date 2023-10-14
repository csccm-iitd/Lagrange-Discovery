#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of Forced Oscillator" 
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
x0 = np.array([1,0])
dt, t0, T = 0.00005, 0, 2
tparam = [dt, t0, T]
xt, ut, t_eval = utils_data.linear_forced(x0, tparam)

fig1 = plt.figure(figsize=(10,8))
plt.subplot(2,1,1); plt.plot(t_eval, xt[0,:]); plt.grid(True); plt.margins(0)
plt.subplot(2,1,2); plt.plot(t_eval, xt[1,:]); plt.grid(True); plt.margins(0)

# %%
""" Generating the design matrix """

x, y, u = sym.symbols('x, y, u')
xvar = [x, y]
D, nd = utils.library_sdof(xvar, polyn=5, harmonic=1, force=u)

Rl, dxdt = utils.euler_lagrange_library(D, [x,y,u], np.vstack((xt,ut)), dt)

# %%
""" compute Sparse regression: sequential least squares """

lam = 0.5      # lam is the sparsification constant
Xi = utils.sparsifyDynamics(Rl[0],dxdt,lam)
Xi = np.insert(Xi,np.where(D == y**2)[0], 1, axis=0)
print(Xi)

""" Lagrangian """
L = sym.Array(0.5*np.dot(D,Xi))
H = (sym.diff(L, y)*y - L)
print('Lagrangian: %s, Hamiltonian: %s' % (L, H))

Lfun = sym.lambdify([x,y,u], L, 'numpy') 
Hfun = sym.lambdify([x,y,u], H, 'numpy')

# %%
""" Hamiltonian """
m, k, F = 10, 5000, 10
H_a = 0.5*xt[1,:]**2 + 0.5*(k/m)*xt[0,:]**2 - (F/m)*ut*xt[0,:]
H_i = Hfun(xt[0,:], xt[1,:], ut)

L_a = 0.5*xt[1,:]**2 - 0.5*(k/m)*xt[0,:]**2 + (F/m)*ut*xt[0,:]
L_i = Lfun(xt[0,:], xt[1,:], ut)

print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22
t_eval = np.linspace(0,T,len(xt[0]))

fig2 = plt.figure(figsize=(8,5))
plt.plot(t_eval, H_a, 'b', label='Actual')
plt.plot(t_eval, H_i[0], 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([0,500])
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
ax.set_title('(b) Forced Oscillator', pad=50)

plt.text(0,-0.25, str(1), color='k', rotation=90)
plt.text(1,-0.25, str(r'$x$'), color='k', rotation=90)
plt.text(2,-0.25, str(r'$\dot{x}$'), color='k', rotation=90)
plt.text(3,-0.25, str(r'$x^2$'), color='b', rotation=90)
plt.text(4,-0.25, str(r'$x \dot{x}$'), color='k', rotation=90)
plt.text(5,-0.25, str(r'$\dot{x}^2$'), color='b', rotation=90)
plt.text(6,-0.25, str(r'$x^3$'), color='k', rotation=90)
plt.text(7,-0.25, str(r'$x^2 \dot{x}$'), color='k', rotation=90)
plt.text(8,-0.25, str(r'$x \dot{x}^2$'), color='k', rotation=90)
plt.text(9,-0.25, str(r'$\dot{x}^3$'), color='k', rotation=90)
plt.text(10,-0.25, str(r'$x^4$'), color='k', rotation=90)
plt.text(11,-0.25, str(r'$x^3\dot{x}$'), color='k', rotation=90)
plt.text(12,-0.25, str(r'$x^2 \dot{x}^2$'), color='k', rotation=90)
plt.text(13,-0.25, str(r'$x \dot{x}^3$'), color='k', rotation=90)
plt.text(14,-0.25, str(r'$\dot{x}^4$'), color='k', rotation=90)
plt.text(15,-0.25, str(r'$x^5$'), color='k', rotation=90)
plt.text(16,-0.25, str(r'$x^4 \dot{x}$'), color='k', rotation=90)
plt.text(17,-0.25, str(r'$x^3 \dot{x}^2$'), color='k', rotation=90)
plt.text(18,-0.25, str(r'$x^2 \dot{x}^3$'), color='k', rotation=90)
plt.text(19,-0.25, str(r'$x \dot{x}^4$'), color='k', rotation=90)
plt.text(20,-0.25, str(r'$\dot{x}^5$'), color='k', rotation=90)
plt.text(21,-0.25, str(r'$Sin (x)$'), color='k', rotation=90)
plt.text(22,-0.25, str(r'$Sin (\dot{x})$'), color='k', rotation=90)
plt.text(23,-0.25, str(r'$Cos (x)$'), color='k', rotation=90)
plt.text(24,-0.25, str(r'$Cos (\dot{x})$'), color='k', rotation=90)
plt.text(25,-0.25, str(r'$xF(t)$'), color='b', rotation=90)
plt.text(26,-0.25, str(r'$\dot{x}F(t)$'), color='k', rotation=90)
