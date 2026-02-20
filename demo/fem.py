from pathlib import Path
from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh
import ufl
import pulse
import numpy as np
import time

# Import the refactored class
from crossbridge import RDQ18, calcium_trace

# --- MESH & PROBLEM SETUP ---
comm = MPI.COMM_WORLD
mesh_geo = mesh.create_unit_cube(comm, 2, 2, 2)

boundaries = [
    pulse.Marker(name="X0", marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0)),
    pulse.Marker(name="X1", marker=2, dim=2, locator=lambda x: np.isclose(x[0], 1)),
]

geo = pulse.Geometry(mesh=mesh_geo, boundaries=boundaries, metadata={"quadrature_degree": 2})

mat_params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
f0 = fem.Constant(mesh_geo, dolfinx.default_scalar_type((1.0, 0.0, 0.0)))
s0 = fem.Constant(mesh_geo, dolfinx.default_scalar_type((0.0, 1.0, 0.0)))
material = pulse.HolzapfelOgden(f0=f0, s0=s0, **mat_params)

# Function Space for Active Tension (DG0 for cell-wise constant)
V_dg0 = fem.functionspace(mesh_geo, ("DG", 0))
Ta_func = fem.Function(V_dg0)

active_model = pulse.ActiveStress(f0, activation=Ta_func)
comp_model = pulse.Incompressible()
cardiac_model = pulse.CardiacModel(
    material=material, active=active_model, compressibility=comp_model
)


def dirichlet_bc(V):
    facets = geo.facet_tags.find(1)
    mesh_geo.topology.create_connectivity(mesh_geo.topology.dim - 1, mesh_geo.topology.dim)
    dofs = fem.locate_dofs_topological(V, 2, facets)
    u_fixed = fem.Function(V)
    u_fixed.x.array[:] = 0.0
    return [fem.dirichletbc(u_fixed, dofs)]


traction_val = fem.Constant(mesh_geo, dolfinx.default_scalar_type(0.0))
neumann = pulse.NeumannBC(traction=traction_val, marker=2)
bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

problem = pulse.StaticProblem(model=cardiac_model, geometry=geo, bcs=bcs)

# --- SARCOMERE MODEL SETUP ---
# Determine local number of cells (for MPI compatibility)
map_c = mesh_geo.topology.index_map(mesh_geo.topology.dim)
num_cells = map_c.size_local

# Initialize the Sarcomere Model
# We set Ta_max here. Adjust physical parameters in the params dict if needed.
sarcomere = RDQ18(num_cells, Ta_max=60.0)

# Kinematics for SL feedback
F = pulse.kinematics.DeformationGradient(problem.u)
# SL = sqrt(dot(f0, C * f0)) * SL0
sl_expr = ufl.sqrt(ufl.dot(f0, ufl.dot(F.T * F, f0))) * sarcomere.p["l0"]
sl_func = fem.Function(V_dg0)
expr_sl = fem.Expression(sl_expr, V_dg0.element.interpolation_points)

# --- TIME STEPPING ---
T_end = 0.6
dt_mech = 0.005
dt_sarc = 2.5e-5  # Matches the default in the class

time_steps = np.arange(0, T_end, dt_mech)
outdir = Path("results_sarcomere_new")
outdir.mkdir(exist_ok=True)

# VTX Writer for ParaView
vtx = dolfinx.io.VTXWriter(mesh_geo.comm, outdir / "cube_contraction.bp", [problem.u], engine="BP4")

if comm.rank == 0:
    print(f"Starting Simulation: {num_cells} local cells.")
    print(f"Time: {T_end}s, Mech dt: {dt_mech}s, Sarc dt: {dt_sarc}s")
    print(f"Sub-steps per mech step: {int(dt_mech / dt_sarc)}")

# Initial State Check
sl_func.interpolate(expr_sl)
initial_sl = sl_func.x.array[:num_cells]

for t in time_steps:
    step_start = time.time()

    # 1. Get Calcium for this time window
    Ca_val = calcium_trace(t)

    # 2. Update Mechanics -> Sarcomere Geometry
    sl_func.interpolate(expr_sl)
    SL_vals = sl_func.x.array[:num_cells]

    # 3. Advance Sarcomere ODEs (Sub-cycling)
    sub_steps = int(dt_mech / dt_sarc)
    sarcomere.advance_ODE(dt_mech, Ca_val, SL_vals)

    # 4. Compute Active Tension
    permissivity = sarcomere.compute_permissivity()
    Ta_func.x.array[:num_cells] = sarcomere.Ta_max * permissivity

    # 5. Solve Mechanics
    try:
        problem.solve()
    except Exception as e:
        if comm.rank == 0:
            print(f"Mechanics solver failed at t={t:.3f}: {e}")
        break

    # 6. Output
    vtx.write(t)

    if comm.rank == 0:
        avg_Ta = np.mean(Ta_func.x.array)
        duration = time.time() - step_start
        print(f"t={t:.3f} | Ca={Ca_val:.2f} | Ta_avg={avg_Ta:.2f} kPa | Wall={duration:.2f}s")

vtx.close()
if comm.rank == 0:
    print("Simulation Complete.")
