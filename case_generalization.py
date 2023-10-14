#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "30-dimensional generalization of the Atomic chain"
"""
import numpy as np
import matplotlib.pyplot as plt
import utils_data_identified
import utils_data

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20

# %%
""" Tri-atomic molecule response """

# The time parameters:
dof = 30
x0 = np.zeros(2*dof)
x0[0] = 1

dt, t0, T = 0.0001, 0, 3
tparam = [dt, t0, T]
xt_tri_a, t_eval = utils_data.atomic_chain(x0, dof, tparam)
xt_tri_i, t_eval = utils_data_identified.atomic_chain(x0, dof, tparam)

# %%
fig4 = plt.figure(figsize=(18,6))

ax = fig4.add_subplot(1, 3, 1, projection='3d')
ax.plot3D(xt_tri_a[4,:], xt_tri_a[5,:], t_eval, 'r', linewidth=2);
ax.plot3D(xt_tri_i[4,:], xt_tri_i[5,:], t_eval, 'b--', linewidth=2); plt.grid(True);
ax.set_title('Atom-3', y=1)
ax.set_xlabel('$x_1(t)$', labelpad = 15, fontweight='bold', color='m'); 
ax.set_zlabel('Time (s)', labelpad = 10, fontweight='bold', color='m'); 
ax.set_ylabel('$\dot{x}_1(t)$', labelpad = 15,  fontsize=24, fontweight='bold', color='m')
plt.margins(0)

ax = fig4.add_subplot(1, 3, 2, projection='3d')
ax.plot3D(xt_tri_a[24,:], xt_tri_a[25,:], t_eval, 'r', linewidth=2);
ax.plot3D(xt_tri_i[24,:], xt_tri_i[25,:], t_eval, 'b--', linewidth=2); plt.grid(True);
ax.set_title('Atom-15', y=1)
ax.set_xlabel('$x_2(t)$', labelpad = 15, fontweight='bold', color='m'); 
ax.set_zlabel('Time (s)', labelpad = 10, fontweight='bold', color='m'); 
ax.set_ylabel('$\dot{x}_2(t)$', labelpad = 15,  fontsize=24, fontweight='bold', color='m')
plt.margins(0)

ax = fig4.add_subplot(1, 3, 3, projection='3d')
ax.plot3D(xt_tri_a[58,:], xt_tri_a[59,:], t_eval, 'r', linewidth=2);
ax.plot3D(xt_tri_i[58,:], xt_tri_i[59,:], t_eval, 'b--', linewidth=2); plt.grid(True);
ax.set_title('Atom-30', y=1)
ax.set_xlabel('$x_3(t)$', labelpad = 15, fontweight='bold', color='m'); 
ax.set_zlabel('Time (s)', labelpad = 10, fontweight='bold', color='m'); 
ax.set_ylabel('$\dot{x}_3(t)$', labelpad = 15, fontsize=24, fontweight='bold', color='m')
plt.legend(['Truth', 'Discovered'], ncol=2, loc=1,  bbox_to_anchor=(1,0.9), columnspacing=0.5,
           handletextpad=0.1, handlelength=0.8, borderpad=0.1, frameon=1,
           fancybox=1, shadow=0, edgecolor=None)
plt.margins(0)
