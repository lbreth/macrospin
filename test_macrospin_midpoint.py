from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from micromagnetictestcases.macrospin.analytic_solution import macrospin_analytic_solution

# Material parameters
Ms = 8.6e5  # saturation magnetisation (A/m)
alpha = 0.02  # Gilbert damping
gamma = 2.211e5  # gyromagnetic ratio

# External magentic field.
B = 0.1  # (T)
mu0 = 4 * np.pi * 1e-7  # vacuum permeability
H = B / mu0
# meaningful time period is of order of nano seconds
dt = 1e-11
t_array = np.arange(0, 5e-9, dt)

############
# Simulation
############

# mesh parameters
d = 50e-9
thickness = 10e-9
nx = ny = 10
nz = 1

# create mesh
p1 = Point(0, 0, 0)
p2 = Point(d, d, thickness)
mesh = BoxMesh(p1, p2, nx, ny, nz)

# define function space for magnetization
M = VectorFunctionSpace(mesh, "CG", 1, dim=3)

# define initial M and normalise
m = Constant((1, 0, 0))
Heff = H * Constant((0, 0, 1))

# define initial value
u_n = interpolate(m, M)

# define variational problem
u = TrialFunction(M)
v = TestFunction(M)

F = dot(u,v)*dx + dt*gamma/2*dot(cross(u + u_n, Heff), v)*dx - alpha*dot(cross(u_n, u), v)*dx - dot(u_n, v)*dx
a, L = lhs(F), rhs(F)

# time stepping
u = Function(M)
t = 0
mx_simulation = np.zeros(t_array.shape)

for i, t in enumerate(t_array):

    mx_simulation[i] = u((0,0,0))[0]
    t += dt
    solve(a == L, u)

    # plot solution
    # plot(u)

    # update previous solution
    # u_n.assign(u)

# Hold plot
# interactive()

###################
# Analytic solution
###################
mx_analytic = macrospin_analytic_solution(alpha, gamma, H, t_array)

###################
# Plot comparison.
###################
plt.figure(figsize=(8, 5))
plt.plot(t_array / 1e-9, mx_analytic, 'o', label='analytic')
plt.plot(t_array / 1e-9, mx_simulation, linewidth=2, label='simulation')
plt.xlabel('t (ns)')
plt.ylabel('mx')
plt.grid()
plt.legend()
plt.savefig('macrospin.pdf', format='pdf', bbox_inches='tight')
