#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of 3DOF structural system" 
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
x0 = np.array([1, 0, 0, 0, 0, 0])
dt, t0, T = 0.0001, 0, 1
tparam = [dt, t0, T]
xt, t_eval = utils_data.mdof_system(x0, tparam)

fig1 = plt.figure(figsize=(10,10))
fig1.subplots_adjust(hspace=0.5)
plt.subplot(6,1,1); plt.plot(t_eval, xt[0,:], 'r'); plt.grid(True); plt.margins(0)
plt.subplot(6,1,2); plt.plot(t_eval, xt[1,:]); plt.grid(True); plt.margins(0)
plt.subplot(6,1,3); plt.plot(t_eval, xt[2,:], 'r'); plt.grid(True); plt.margins(0)
plt.subplot(6,1,4); plt.plot(t_eval, xt[3,:]); plt.grid(True); plt.margins(0)
plt.subplot(6,1,5); plt.plot(t_eval, xt[4,:], 'r'); plt.grid(True); plt.margins(0)
plt.subplot(6,1,6); plt.plot(t_eval, xt[5,:]); plt.grid(True); plt.margins(0)

# %%
""" Generating the design matrix """
xvar = [sym.symbols('x'+str(i)) for i in range(1, 6+1)]
D, nd = utils.library_mdof(xvar, polyn=6, funofvelocity=1, harmonic=1)

Rl, dxdt = utils.euler_lagrange_library(D, xvar, xt, dt)

# %%
""" Sparse regression: sequential least squares """
Xi = []
lam = 0.5     # lam is the sparsification constant
for i in range(3):
    Xi.append(utils.sparsifyDynamics(Rl[i], dxdt[i:i+1], lam))
Xi = (np.array(Xi).squeeze(-1)).T

xvar_vel = xvar[1::2]
Xi_final = np.zeros([nd, 3])
for i in range(3):
    Xi_final[:,i] = np.insert(Xi[:,i], np.where(D == xvar_vel[i]**2)[0], 1)
    
Xi = np.column_stack( (np.arange(0,nd,1), Xi_final) )
print(Xi)

# %%
""" Lagrangian """
xdot = xvar[1::2]
Xi_reduced = utils.nonequal_sum(np.array(Xi_final))
L = np.sum(sym.Array(0.5*np.dot(D,Xi_reduced)))
H = 0
for i in range(len(xdot)):
    H += (sym.diff(L, xdot[i])*xdot[i])
H = H - L
print('Lagrangian: %s, Hamiltonian: %s' % (L, H))

Lfun = sym.lambdify([xvar], L, 'numpy') 
Hfun = sym.lambdify([xvar], H, 'numpy')

# %%
""" Hamiltonian """
m1, m2, m3, k1, k2, k3 = 10, 10, 10, 5000, 5000, 5000
H_a = (0.5*xt[1,:]**2 + 0.5*xt[3,:]**2 + 0.5*xt[5,:]**2) + \
      (0.5*(k1/m1)*xt[0,:]**2 + 0.5*(k2/m2)*(xt[2,:]-xt[0,:])**2 + \
       0.5*(k3/m3)*(xt[4,:]-xt[2,:])**2)
H_i = Hfun(xt)
        
L_a = (0.5*xt[1,:]**2 + 0.5*xt[3,:]**2 + 0.5*xt[5,:]**2) - \
      (0.5*(k1/m1)*xt[0,:]**2 + 0.5*(k2/m2)*(xt[2,:]-xt[0,:])**2 + \
       0.5*(k3/m3)*(xt[4,:]-xt[2,:])**2)
L_i = Lfun(xt)
    
print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22
t_eval = np.linspace(0,T,len(xt[0]))

fig2 = plt.figure(figsize=(8,5))
plt.plot(t_eval, H_a, 'b', label='Actual')
plt.plot(t_eval, H_i, 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([0,1000])
plt.grid(True)
plt.legend()
plt.margins(0)

# %%
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 14

Xi_plot = np.array(Xi[:,1:].T)
Xi_plot[Xi_plot != 0 ] = -1

fig3, ax = plt.subplots(figsize=(10,1))
ax = sns.heatmap( Xi_plot, linewidth=0.5, cmap='Set1', cbar=False, yticklabels=[0,1,2])
ax.set_xlabel('Library functions')
ax.set_ylabel('DOF')
ax.set_title('(d) 3DOF dynamical system', pad=20)

ax.text(6.5,-0.25, str(r'$x_1^2$'), color='b', fontsize=10)
ax.text(8,-0.25, str(r'$\dot{x}_1^2$'), color='b', fontsize=10)
ax.text(10,-0.25, str(r'$\dot{x}_2^2$'), color='b', fontsize=10)
ax.text(12,-0.25, str(r'$\dot{x}_3^2$'), color='b', fontsize=10)
ax.text(20,-0.25, str(r'$(x_2 - x_1)^2$'), color='b', fontsize=10, ha='right')
ax.text(21,-0.25, str(r'$(x_3 - x_2)^2$'), color='b', fontsize=10)
