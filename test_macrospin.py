import numpy as np
import matplotlib.pyplot as plt
import dolfin as df
from micromagnetictestcases.macrospin.analytic_solution import macrospin_analytic_solution

# Material parameters
Ms = 8.6e5  # saturation magnetisation (A/m)
alpha = 0.1  # Gilbert damping
gamma = 2.211e5  # gyromagnetic ratio

# External magentic field.
B = 0.1  # (T)
mu0 = 4 * np.pi * 1e-7  # vacuum permeability
H = B / mu0
# meaningful time period is of order of nano seconds
dt = 0.01e-9
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
p1 = df.Point(0, 0, 0)
p2 = df.Point(d, d, thickness)
mesh = df.BoxMesh(p1, p2, nx, ny, nz)

# define function space
V = df.VectorFunctionSpace(mesh, "CG", degree=1, dim=3)

# define initial M and normalise
m = df.Constant((1, 0, 0))
m = df.project(m / df.sqrt(df.dot(m, m)), V)
# m = df.project(m / df.sqrt(df.dot(m, m)), V)
Heff = H * df.Constant((0, 0, 1))
# Heff = df.project(Heff / df.sqrt(df.dot(Heff, Heff)), V)

# define dmdt, test and trial functions
dmdt = df.Function(V)
v = df.TrialFunction(V)
w = df.TestFunction(V)

# define the exchange field
# f_ex = df.Constant(2*A/(mu0*Ms))

# results for m_x at times t_array in
mx_simulation = np.zeros(t_array.shape)

for i, t in enumerate(t_array):

    mx_simulation[i] = m((0,0,0))[0]

    a = df.inner(df.Constant(alpha)*v + df.cross(m, v), w)*df.dx
    L = df.inner(gamma*Heff, w)*df.dx
    df.solve(a == L, dmdt)

    m += dmdt * df.Constant(dt)
    m = df.project(m / df.sqrt(df.dot(m, m)), V)
    

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
