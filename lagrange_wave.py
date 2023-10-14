#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of the Wave equation"
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
dx = 0.01       # Spacing of points on string
dt = 0.0001     # Size of time step
c = 25          # Speed of wave propagation
L = 0.5         # Length of string
stopTime = 1    # Time to run the simulation
t  = np.arange(0, stopTime+dt, dt)
mesh = np.arange(0, L+dt, dx)

xt = utils_data_pde.string(L,stopTime,c,dx,dt)

dis = xt[::2]
vel = xt[1::2]

# %%
fig1 = plt.figure(figsize=(18,10))
plt.subplot(2,1,1); plt.imshow(dis, cmap='nipy_spectral', aspect='auto')
plt.subplot(2,1,2); plt.imshow(vel, cmap='nipy_spectral', aspect='auto')

# %%
l = 1000
fig11 = plt.figure(figsize=(10,8))
ax = plt.axes(projection ='3d')
a, b = np.meshgrid(t[:-1], mesh)
surf = ax.plot_surface(a[:,:l], b[:,:l], dis[:,:l], cmap='ocean', antialiased=False)
ax.view_init(30,-130)
fig11.colorbar(surf, shrink=0.85, aspect=20)
ax.set_xlabel('Time (s)', labelpad = 20, fontweight='bold');
ax.set_ylabel('x', labelpad = 20, fontweight='bold');
ax.set_zlabel('u(x,t)', labelpad = 10, fontweight='bold')
plt.margins(0)

# %%
""" Generating the design matrix """
xvar = [sym.symbols('x'+str(i)) for i in range(1, 2*int(L/dx)+1)]
D, nd = utils.library_pde(xvar, Type='order1', dx=dx, polyn=4)

Rl, dxdt = utils.euler_lagrange_library(D, xvar, xt, dt)

# %%
""" Sparse regression: sequential least squares """

Xi = []
lam = 150     # lam is the sparsification constant
data = dxdt
for kk in range(int(len(xvar)/2)):
    print('Element- ', kk)
    if len(np.where( data[kk] != 0)[0]) < 5: # to check, if the vector is full of zeros
        print('Zero vector encountered')
        Xi.append(np.zeros([Rl[0].shape[1],1]))
    else:
        Xi.append(utils.sparsifyDynamics(Rl[kk], data[kk:kk+1], lam))

Xi = np.column_stack( (np.array(Xi).squeeze(-1)) )

xvar_vel = xvar[1::2]
Xi_final = np.zeros([nd, int(len(xvar)/2)])
for kk in range(len(xvar_vel)):
    if len(np.where( data[kk] != 0)[0]) < 5:
        Xi_final[:,kk] = np.insert(Xi[:,kk], np.where(D == xvar_vel[kk]**2)[0], 0)
    else:
        Xi_final[:,kk] = np.insert(Xi[:,kk], np.where(D == xvar_vel[kk]**2)[0], 1)

# %%
predict = np.sqrt( np.abs(np.mean(Xi_final[Xi_final < 0])) )
rel_error = 100*np.abs(predict-c)/c
print("Actual: %d, Predicted: %0.4f, Relative error: %0.4f percent." % (c,predict,rel_error))

# %%
""" Lagrangian """
xvar_vel = xvar[1::2]
Xi_reduced = utils.nonequal_sum(np.array(Xi_final))
L = np.sum(sym.Array(0.5*np.dot(D,Xi_reduced)))
H = 0
for i in range(len(xvar_vel)):
    H += (sym.diff(L, xvar_vel[i])*xvar_vel[i])
H = H - L
print('Lagrangian: %s, Hamiltonian: %s' % (L, H))
Lfun = sym.lambdify([xvar], L, 'numpy') 
Hfun = sym.lambdify([xvar], H, 'numpy')

# %%
""" Hamiltonian """
const = c**2/dx**2/2
T = np.sum(vel[:-2]**2, axis=0)
V = const*(np.sum(np.diff(dis,axis=0)**2, axis=0))

H_a = 0.5*T + V
L_a = 0.5*T - V
    
H_i = Hfun(xt[:-2,:])
L_i = Lfun(xt[:-2,:])

print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22

fig2 = plt.figure(figsize=(8,6))
plt.plot(t, H_a, 'b', label='Actual')
plt.plot(t, H_i, 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([11000,13000])
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
ax.set_title('(a) Elastic Transversal Waves in a Solid', fontweight='bold', pad=30)

ax.text(5, -0.5, r'$\{u(x,t)^2_{(i)}\}$', color='b', fontsize=20)
ax.text(100, -0.5, r'$\{(u(x,t)_{(i-1)} - u(x,t)_{(i+1)})^2\}$', color='b', fontsize=20)
plt.margins(0)
