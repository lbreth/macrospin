from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
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
p1 = Point(0, 0, 0)
p2 = Point(d, d, thickness)
mesh = BoxMesh(p1, p2, nx, ny, nz)

# Set up mixed function space
VV = VectorFunctionSpace(mesh, "CG", 1, dim=3)
VS = FunctionSpace(mesh, "CG", 1)
#V = VV * VS

VFM = VectorElement("CG", mesh.ufl_cell(), 1, dim=3)
SFM = VectorElement("CG", mesh.ufl_cell(), 1, dim=1)

V = FunctionSpace(mesh, MixedElement([VFM, SFM]))

# define initial M and normalise
m = Constant((1, 0, 0))
m = project(m / sqrt(dot(m, m)), VV)
Heff = H * Constant((0, 0, 1))

# define dmdt, test and trial functions
dmdt = Function(VV)
(v, lam) = TrialFunction(V)
(w, sigma) = TestFunction(V)


# results for m_x at times t_array in
mx_simulation = np.zeros(t_array.shape)

for i, t in enumerate(t_array):

    mx_simulation[i] = m((0,0,0))[0]

    a = alpha*dot(v,w)*dx \
        + dot(cross(m,v),w)*dx \
        + sigma*inner(m,v)*dx \
        + lam*inner(m,w)*dx
    # a = df.inner(df.Constant(alpha)*v + df.cross(m, v), w)*df.dx
    L = inner(gamma*Heff, w)*dx
    solve(a == L, dmdt)

    m += dmdt * Constant(dt)
    m = project(m / sqrt(dot(m, m)), V)
    

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
