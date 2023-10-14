"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Navier-Stokes equation"
-- This code is implemented on a global sense, i.e., the data matrix is 
   reshaped into a vector and then subsequent operations are performed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import utils_data_pde
import utils
import sympy as sym
import seaborn as sns
import pandas as pd 

plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 10

# %%
""" Channel Flow :: variable declarations """
nx = 256
ny = 256
T = 1

dx = 1 / (nx - 1)
dy = 1 / (ny - 1)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

rho = 900
nu = 0.0001
F = 0.1*(np.sin(2*np.pi*(X + Y)) + np.cos(2*np.pi*(X + Y)))
dt = .0001
nt = int(T/dt)
t = np.linspace(0, T, nt)

# initial conditions
u = 0.01*np.ones((ny, nx))
v = 0.001*np.ones((ny, nx))
p = np.ones((ny, nx))
b = np.zeros((ny, nx))
u, v, p = utils_data_pde.channel_flow(nt, u, v, dt, dx, dy, p, rho, nu, F)

print('Reynold Number-{}'.format( (rho*max(np.max(u), np.max(v))*(1/4)) / (rho*nu) ))

# Deleting the rows and columns which contain zeros
u, v, p = u[1:-1, 1:-1, :], v[1:-1, 1:-1, :], p[1:-1, 1:-1, :]
X, Y = X[1:-1, 1:-1], Y[1:-1, 1:-1]
F = F[1:-1, 1:-1]

# %%
fig1, axes = plt.subplots(nrows=3, ncols=3, figsize=(14,8), dpi=100)
fig1.subplots_adjust(hspace=0.25, wspace=0.25)
axes = axes.flatten()

index = 0
for i in range(1, nt):
    if i % (nt//3) == 0:
        print(i)
        # plotting the velocity in x-direction
        img = axes[index].imshow(u[:, :, i], aspect='auto', cmap='jet', origin='lower',
                           interpolation='Gaussian', extent=[0, 1, 0, 1])
        plt.colorbar(img, ax=axes[index])
        
        # plotting the velocity in y-direction
        img = axes[index+3].imshow(v[:, :, i], aspect='auto', cmap='jet', origin='lower',
                             interpolation='Gaussian', extent=[0, 1, 0, 1])
        plt.colorbar(img, ax=axes[index+3])
    
        # plotting velocity field
        axes[index+6].contourf(X[::2, ::2], Y[::2, ::2], p[::2, ::2, i], cmap='ocean', alpha=0.75)
        speed = np.sqrt(u[::2, ::2, i]*u[::2, ::2, i] + v[::2, ::2, i]*v[::2, ::2, i])
        axes[index+6].streamplot(X[::2, ::2], Y[::2, ::2], u[::2, ::2, i], v[::2, ::2, i], 
                                 color=u[::2, ::2, i], cmap='nipy_spectral', density=(1,1),
                                 linewidth = 2*speed/speed.max(),) 
        plt.colorbar(img, ax=axes[index+6])
        axes[index+6].set_xlabel('X')
        axes[index+6].set_ylabel('Y');
        index += 1

# %%
""" Generating the design matrix """

def dictionary():
    """ Form Lagrangian library: """
    # Energy libraries:
    D = np.array(['1', 'u**2', 'v**2', 'p**2', 'u', 'Grad(u,x)', 'Grad(u,y)',
                  'p', 'Grad(p,x)', 'Grad(p,y)', 'Fx'], dtype='object') 
    
    # D_x = Grad(D,(t,x))
    D_x = np.array(['0', '2u', '0', '0', 'u_x', 'u_xx', 'u_yy',
                    'p_x', 'p_xx', 'p_yy', 'F'], dtype='object') 
    
    # D_y = Grad(D,(t,y))
    D_y = np.array(['0', '0', '2v', '0', 'v_y', 'v_xx', 'v_yy',
                    'p_y', 'p_xx', 'p_yy', '0'], dtype='object')
    
    # Targets:
    # y = 'Dt(u,t)'
    Dh_x = 'u_t + u*u_x + v*u_y'
    Dh_y = 'v_t + u*v_x + v*v_y'
    return (Dh_x, Dh_y), (D, D_x, D_y)

print( dictionary() )

# %%
def library(u, v, p, force, nx, ny, x=None, y=None, t=None):
    shape_x, shape_y, shape_t = [*u.shape]
    
    # Time serivative of velocity field:
    u_t = np.zeros_like(u)
    v_t = np.zeros_like(v)
    for i in range(nx):
        for j in range(ny):
            u_t[i, j, :] = utils.FiniteDiff(u[i, j, :], dt, 1)
            v_t[i, j, :] = utils.FiniteDiff(v[i, j, :], dt, 1)   
            
    # Derivative of pressure fields:
    p_x = np.zeros_like(p)
    p_y = np.zeros_like(p)
    p_xx = np.zeros_like(p)
    p_yy = np.zeros_like(p)
    for i in range(nt):
        for j in range(ny):
            p_x[j, :, i] = utils.FiniteDiff(p[j, :, i], dx, 1)
            p_y[:, j, i] = utils.FiniteDiff(p[:, j, i], dy, 1)
            
            p_xx[j, :, i] = utils.FiniteDiff(p[j, :, i], dx, 2)
            p_yy[:, j, i] = utils.FiniteDiff(p[:, j, i], dy, 2)
    
    # Derivates of velocity fields:
    u_x = np.zeros_like(u)
    u_y = np.zeros_like(u)
    v_x = np.zeros_like(v)
    v_y = np.zeros_like(v)
    
    u_xx = np.zeros_like(u)
    u_yy = np.zeros_like(u)
    v_xx = np.zeros_like(v)
    v_yy = np.zeros_like(v)
    for i in range(nt):
        for j in range(ny):
            # 1st Derivative of velocity component in x-y direction
            u_x[j, :, i] = utils.FiniteDiff(u[j, :, i], dx, 1)
            u_y[:, j, i] = utils.FiniteDiff(u[:, j, i], dy, 1)
            v_x[j, :, i] = utils.FiniteDiff(v[j, :, i], dx, 1)
            v_y[:, j, i] = utils.FiniteDiff(v[:, j, i], dy, 1)

            # 2nd Derivative of velocity component in x-y direction        
            u_xx[j, :, i] = utils.FiniteDiff(u[j, :, i], dx, 2)
            u_yy[:, j, i] = utils.FiniteDiff(u[:, j, i], dy, 2)
            v_xx[j, :, i] = utils.FiniteDiff(v[j, :, i], dx, 2)
            v_yy[:, j, i] = utils.FiniteDiff(v[:, j, i], dy, 2)
            
    # Stacking for converting 2D grid into 1D grid:        
    u = utils.reshape(u, shape_x, shape_y)
    v = utils.reshape(v, shape_x, shape_y)
    p = utils.reshape(p, shape_x, shape_y)
    
    u_t = utils.reshape(u_t, shape_x, shape_y)
    v_t = utils.reshape(v_t, shape_x, shape_y)
    
    u_x = utils.reshape(u_x, shape_x, shape_y)
    u_y = utils.reshape(u_y, shape_x, shape_y)
    v_x = utils.reshape(v_x, shape_x, shape_y)
    v_y = utils.reshape(v_y, shape_x, shape_y)
    
    u_xx = utils.reshape(u_xx, shape_x, shape_y)
    u_yy = utils.reshape(u_yy, shape_x, shape_y)
    v_xx = utils.reshape(v_xx, shape_x, shape_y)
    v_yy = utils.reshape(v_yy, shape_x, shape_y)
    
    p_x = utils.reshape(p_x, shape_x, shape_y)
    p_y = utils.reshape(p_y, shape_x, shape_y)
    p_xx = utils.reshape(p_xx, shape_x, shape_y)
    p_yy = utils.reshape(p_yy, shape_x, shape_y)
    
    force = np.repeat(force[:,:,None], shape_t, axis=-1)
    force = utils.reshape(force, shape_x, shape_y)
    force = force.reshape(-1,1)
        
    # Stacking into vector shape for reducing library functions:
    u = u.reshape(-1,1)
    v = v.reshape(-1,1)
    p = p.reshape(-1,1)
    
    u_t = u_t.reshape(-1,1)
    v_t = v_t.reshape(-1,1)
    
    u_x = u_x.reshape(-1,1)
    u_y = u_y.reshape(-1,1)
    v_x = v_x.reshape(-1,1)
    v_y = v_y.reshape(-1,1)
    
    u_xx = u_xx.reshape(-1,1)
    u_yy = u_yy.reshape(-1,1)
    v_xx = v_xx.reshape(-1,1)
    v_yy = v_yy.reshape(-1,1)
    
    p_x = p_x.reshape(-1,1)
    p_y = p_y.reshape(-1,1)
    p_xx = p_xx.reshape(-1,1)
    p_yy = p_yy.reshape(-1,1)
    
    # Final Library function:    
    zeros = np.zeros((shape_x * shape_y * shape_t))[:, None]
    ones = np.ones((shape_x * shape_y * shape_t))[:, None]
                        
    D_x = np.column_stack(( zeros, 0*u, zeros, zeros, u_x, u_xx, u_yy, p_x, p_xx, p_yy, force ))
    D_y = np.column_stack(( zeros, zeros, 0*v, zeros, v_y, v_xx, v_yy, p_y, p_xx, p_yy, zeros ))
    
    return (u_t+u*u_x+v*u_y, v_t+u*v_x+v*v_y), (D_x, D_y)
        
# %%
(target_x, target_y), (D_x, D_y) = library(u, v, p, F, nx-2, ny-2)

# %%
Xi = utils.sparsifyDynamics(D_x, target_x.T, 0.0001)
Xi = np.append(Xi, utils.sparsifyDynamics(D_y, target_y.T, 0.0001), axis=1)

# %%
data = { 'Lagrangian ':dictionary()[1][0],
         'Functions x':dictionary()[1][1], 'Theta 0':Xi[:,0], 
         'Functions y':dictionary()[1][2], 'Theta 1':Xi[:,1] }
df = pd.DataFrame(data)
print(df)

# %%
def library_0(u, v, p, force, nx, ny, x=None, y=None, t=None):
    shape_x, shape_y, shape_t = [*u.shape]
            
    # Derivates of velocity fields:    
    u_x = np.zeros_like(u)
    u_y = np.zeros_like(u)
    v_x = np.zeros_like(v)
    v_y = np.zeros_like(v)
    for i in range(nt):
        for j in range(ny):
            # 2nd Derivative of velocity component in x-y direction        
            u_x[j, :, i] = utils.FiniteDiff(u[j, :, i], dx, 1)
            u_y[:, j, i] = utils.FiniteDiff(u[:, j, i], dy, 1)
            v_x[j, :, i] = utils.FiniteDiff(v[j, :, i], dx, 1)
            v_y[:, j, i] = utils.FiniteDiff(v[:, j, i], dy, 1)
            
    force = np.repeat(force[:,:,None], shape_t, axis=-1)    
    return (u_x, u_y, force*u), (v_x, v_y)

# %%
Lx, Ly = library_0(u, v, p, F, nx-2, ny-2)
(u_x, u_y, Fu), (v_x, v_y) = Lx, Ly

# %%
""" Hamiltonian """
T_a = 0.5*rho*u**2
V_a = - p + nu*(u_x + u_y) + Fu
H_a = np.sum( (T_a + V_a), (0,1) )
L_a = np.sum( (T_a - V_a), (0,1) )

T_i = 0.5*(1/0.001240)*u**2
V_i = - p + 0.000101*u_x + 0.000139*u_y + Fu
H_i = np.sum( (T_i + V_i), (0,1) )
L_i = np.sum( (T_i - V_i), (0,1) )

print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

# %%
""" Hamiltonian """
T_x_a = 0.5*rho*u**2
V_x_a = - p + nu*(u_x + u_y)*u_x + Fu
H_x_a = np.sum( (T_x_a - V_x_a), (0,1) )
L_x_a = np.sum( (T_x_a + V_x_a), (0,1) )

T_y_a = 0.5*rho*v**2
V_y_a = - p + nu*(v_x + v_y)*u_y 
H_y_a = np.sum( (T_y_a - V_y_a), (0,1) )
L_y_a = np.sum( (T_y_a + V_y_a), (0,1) )

L_a = np.sqrt( L_x_a**2 + L_y_a**2 )
H_a = np.sqrt( H_x_a**2 + H_y_a**2 )

T_x_i = 0.5*(1/0.001240)*u**2
V_x_i = - p + (0.000101*u_x**2 + 0.000139*u_y)*u_x + Fu
H_x_i = np.sum( (T_x_i - V_x_i), (0,1) )
L_x_i = np.sum( (T_x_i + V_x_i), (0,1) )

T_y_i = 0.5*(1/0.001240)*v**2
V_y_i = - p + (0.000101*v_x**2 + 0.000139*v_y)*u_y 
H_y_i = np.sum( (T_y_i - V_y_i), (0,1) )
L_y_i = np.sum( (T_y_i + V_y_i), (0,1) )

L_i = np.sqrt( L_x_i**2 + L_y_i**2 )
H_i = np.sqrt( H_x_i**2 + H_y_i**2 )

print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

# %%
""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22

fig2 = plt.figure(figsize=(8,6))
plt.plot(t, H_a, 'b', label='Actual')
plt.plot(t, H_i, 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.grid(True)
plt.legend()
plt.margins(0)
