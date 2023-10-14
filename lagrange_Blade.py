#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of Flexion vibration of beam"
-- This code is implemented particle-wise, i.e., the Lagrangian is discovered 
   for each particle/spatial location. 
"""

import numpy as np
import matplotlib.pyplot as plt
import utils_data_pde
import utils
import sympy as sym
import seaborn as sns

# %%
""" Generating system response """

# The time parameters:
Ne = 100
dt = 0.0001     # Size of time step
L = 1           # Length of string
T_end = 2       # Time to run the simulation
rho = 8100
b, d = 0.02, 0.001
A = b*d
E = 2e10
I = (b*d**3)/12 
params = [rho, b, d, A, L, E, I]
c = (E*I)/(2*rho*A)

print('Wave coefficient-{}'.format(c))
print('r-{}'.format(c*dt**2/(1/Ne)**4))
t  = np.arange(0, T_end+dt, dt)
mesh = np.linspace(0, L, Ne)

dis, vel, acc = utils_data_pde.cantilever(params,T_end,dt,Ne)

xt = np.zeros([2*dis.shape[0],dis.shape[1]])
xt[::2] = dis
xt[1::2] = vel

# %%
fig1 = plt.figure(figsize=(10,8))
ax = plt.axes(projection ='3d')
a, b = np.meshgrid(t[:xt.shape[1]], mesh)
surf = ax.plot_surface(a, b, dis, cmap='nipy_spectral', antialiased=True)
ax.view_init(30,55)
fig1.colorbar(surf, shrink=0.85, aspect=20)
ax.set_xlabel('Time (s)', labelpad = 20, fontweight='bold');
ax.set_ylabel('x', labelpad = 20, fontweight='bold');
ax.set_zlabel('u(x,t)', labelpad = 10, fontweight='bold')
plt.margins(0)

# %%
""" Generating the design matrix """
xvar = [sym.symbols('x'+str(i)) for i in range(1, 2*int(Ne/2)+1)]
D, nd = utils.library_pde(xvar, Type='order2', dx=L/Ne, polyn=4)

Rl, dxdt = utils.euler_lagrange_library(D, xvar, xt, dt)

# %%
""" Sparse regression: sequential least squares """
Xi = []
lam = 0.1      # lam is the sparsification constant
for kk in range(int(len(xvar)/2)):
    print('Element-', kk)
    data = dxdt[kk:kk+1] 
    Xi.append(utils.sparsifyDynamics(Rl[kk], data, lam))

Xi = np.column_stack( (np.array(Xi).squeeze(-1)) )

predict = np.abs(np.mean(Xi[np.nonzero(Xi)]))
rel_error = 100*np.abs(predict-c)/c
print("Actual: %0.4f, Predicted: %0.4f, Relative error: %0.4f percent." % (c,predict,rel_error))

xvar_vel = xvar[1::2]
Xi_final = np.zeros([nd, int(len(xvar)/2)])
for kk in range(len(xvar_vel)):
    if len(np.where( dxdt[kk] != 0)[0]) < 5:
        Xi_final[:,kk] = np.insert(Xi[:,kk], np.where(D == xvar_vel[kk]**2)[0], 0)
    else:
        Xi_final[:,kk] = np.insert(Xi[:,kk], np.where(D == xvar_vel[kk]**2)[0], 1)

# %%
""" Lagrangian """
xvar_vel = xvar[1::2]
Xi_reduced = utils.nonequal_sum(np.array(Xi_final))
L = np.sum(sym.Array(np.dot(D,Xi_reduced)))
H = 0
for i in range(len(xvar_vel)):
    H += (sym.diff(L, xvar_vel[i])*xvar_vel[i])
H = H - L

Lfun = sym.lambdify([xvar], L, 'numpy') 
Hfun = sym.lambdify([xvar], H, 'numpy')

# %%
""" Hamiltonian """
T = 0.5*np.sum(vel[2:-2,:]**2, axis=0)
V = (c/(1/Ne)**4)*np.sum((-np.diff(dis[:-1,:], axis=0) + np.diff(dis[1:,:], axis=0))**2, axis=0)
    
H_a = 0.5*T + 0.5*V
L_a = 0.5*T - 0.5*V
    
T_i = 0.5*np.sum(vel[2:-2,:]**2, axis=0)
V_i = (0.1023/(1/Ne)**4)*np.sum((-np.diff(dis[:-1,:], axis=0) + np.diff(dis[1:,:], axis=0))**2, axis=0)

H_i = 0.5*T_i + 0.5*V_i
L_i = 0.5*T_i - 0.5*V_i

print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22
t_eval = np.arange(0,T_end+dt,dt)

fig2 = plt.figure(figsize=(8,6))
plt.plot(t_eval, H_a, 'b', label='Actual')
plt.plot(t_eval, H_i, 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([0,100])
plt.grid(True)
plt.legend()
plt.margins(0)

# %%
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 20

Xi_plot = np.array(Xi.T)
Xi_plot[Xi_plot != 0 ] = -1

fig3, ax = plt.subplots(figsize=(20,6))
ax = sns.heatmap( Xi_plot, linewidth=0.5, cmap='Set1', cbar=False, )
ax.set_xlabel('Library functions', fontsize=24)
ax.set_ylabel('Segments', fontsize=24)
ax.set_title('(a) Flexion Vibration of a Blade', fontweight='bold', pad=30)

ax.text(5, -0.5, r'$\{u(x,t)^2_{(i)}\}$', color='b', fontsize=20)
plt.text(32, -0.5 ,r'$\{u(x,t)_{(i-1)} -2u(x,t)_{(i)} +u(x,t)_{(i+1)}\}^2$',
          color='b', fontsize=22)
plt.margins(0)
