import os 
import time
import mujoco
import mujoco_viewer
import numpy as np
from pathlib import Path
import ds2f.utils.mjc2_utils as mjc2
from ds2f.utils.xml_utils import XMLWrapper
from ds2f.utils.mjc_utils import MjSimWrapper
from ds2f.utils.mjc2_utils import init_plugins
import ds2f.utils.dlo_s2f.Dlo_s2f as Dlo_s2f
from ds2f.assets.genrope.gen_overall_native_xml import generate_overall_native_xml
from ds2f.utils.interp_utils import resample_wire_equal_distance_min_dev
from ds2f.utils.plotter import plot_nforces_timing

# Settings
do_render = False
# update stiffness, mass, and length as needed.
alpha_bar = 0.001196450659614982    # Obtained from simple PI
beta_bar = 0.001749108044378543
mass_per_length = 0.079/2.98
thickness = 0.006
j_damp = 0.002
someStiff = 0

n_pieces = 50   # adjust parllThreshold with change in n_pieces
scale_factor = 1000.0

torq_tol = 1e-8
tolC2 = 3e-4
tolC3 = 3e-4
solve_freq = 10
benchmark_seconds = 20.0  # Run benchmark for this many simulation seconds
n_force_lst = [1, 2, 3, 4, 5]  # Benchmark each force-count setting

assets_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'ds2f/assets'
)
xml_path = os.path.join(assets_path,'overall_s2f.xml')
graph_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'ds2f/data/figs'
)

init_plugins()


# init stiffnesses for capsule
J1 = np.pi * (thickness/2)**4/2.
Ix = np.pi * (thickness/2)**4/4.
stiff_vals = [
    beta_bar/J1,
    alpha_bar/Ix
]

wire_pos = np.zeros((n_pieces,3))

wire_pos = np.array([
    [0.0,0,0],
    [0.0,0,-0.1],
    [0.30,0,-0.1],
    [0.30,0,0],
])
wire_pos = resample_wire_equal_distance_min_dev(wire_pos, n_pieces, iters=300, step_size=0.4, rng_seed=0)

# Generate wire xml
# includes conversion of wire xpos to main body pose and subsequent connected quaternions
# Compute total arc length
segment_lengths = np.linalg.norm(np.diff(wire_pos, axis=0), axis=1)
total_length = np.sum(segment_lengths)

# Mass per unit length
mass = mass_per_length * total_length
# init_pos = np.array([0.0, 0.0, 0.5])
# init_quat = np.array([1.0, 0.0, 0.0, 0.0])
rgba_wire = "0.1 0.0533333 0.673333 1"
rgba_pipe = "1.0 1.0 1.0 0.1"
generate_overall_native_xml(
    n_pieces=n_pieces-1,
    thickness=thickness,
    mass=mass,
    j_damp=j_damp,
    con_val=(1,0),
    stiff_bend=stiff_vals[1],
    stiff_twist=stiff_vals[0],
    wire_pos=wire_pos,
    xml_path=xml_path,
    rgba=rgba_wire,
    someStiff=someStiff
)
xml = XMLWrapper(xml_path)

# # Load MuJoCo model and data
# model = mujoco.MjModel.from_xml_path(xml_path)
xml_string = xml.get_xml_string()
model = mujoco.MjModel.from_xml_string(xml_string)
mujoco.mj_saveLastXML(xml_path,model)
data = mujoco.MjData(model)
# model.opt.gravity[-1] = 1.0
model.opt.gravity[-1] = -9.81
# model.opt.gravity[-1] = 0.0

known_body_name = "B_first"
plgn_instance = model.body_plugin[
    mjc2.obj_name2id(model, "body", known_body_name)
]
start = model.plugin_stateadr[plgn_instance]
r_len = total_length
r_pieces = len(wire_pos) - 1
vec_bodyid = np.zeros(r_pieces, dtype=int)
for i in range(r_pieces):
    if i == 0:
        i_name = 'first'
    elif i == r_pieces-1:
        i_name = 'last'
    else:
        i_name = str(i)
    vec_bodyid[i] = mjc2.obj_name2id(
        model,"body",'B_' + i_name
    )
vec_bodyid_full = np.concatenate((
    vec_bodyid, [mjc2.obj_name2id(
        model,"body",'B_last2'
    )]
))

def get_storedtorque(cur_data):
    if plgn_instance == model.nplugin - 1:
        # Last plugin - use remaining state
        stored_torques = cur_data.plugin_state[start:start+model.nv]
        E_total = cur_data.plugin_state[-1]
    else:
        # Not last plugin - use next plugin's start as end
        # stored_torques = data.plugin_state[start:model.plugin_stateadr[plgn_instance+1]-1]
        stored_torques = cur_data.plugin_state[start:start+model.nv]
        E_total = cur_data.plugin_state[start+model.nv]
    return stored_torques, E_total

def run_single_benchmark(n_forces):
    data = mujoco.MjData(model)
    sim = MjSimWrapper(model, data)
    viewer = None

    if do_render:
        viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer.vopt.geomgroup[3] ^= 1
        viewer.cam.distance = 0.5
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = 0.0
        viewer.cam.lookat = np.array([0.0, 0.0, 0.0])
        viewer.render()
        viewer._paused = True
        viewer.perturbation_scale = 3.0

    dt = model.opt.timestep
    cur_time = 0.0

    sim.forward()
    sim.step()
    if do_render:
        viewer.render()
    sim.forward()
    cur_time += dt

    ropemass = mass
    ds2f = Dlo_s2f.DLO_s2f(
        r_len, r_pieces, ropemass*(-model.opt.gravity[-1]),
        boolErrs=False, boolSolveTorq=True,
        torque_tolerance_=torq_tol,
        tolC2_=tolC2, tolC3_=tolC3
    )

    ntorq = np.zeros_like(data.qfrc_passive.flatten())
    n_checks = 0
    timing_samples = []
    solvable_true_count = 0
    solver_call_count = 0
    last_solved_forces = None
    last_solved_positions = None
    last_progress = -1

    # Evenly distribute force application points along the wire (excluding both ends).
    if n_forces < 1:
        raise ValueError(f"n_forces must be >= 1, got {n_forces}")
    body_indices_float = np.linspace(0, len(vec_bodyid_full)-1, n_forces+2)[1:-1]
    body_indices = np.rint(body_indices_float).astype(int)
    if len(np.unique(body_indices)) != n_forces:
        raise ValueError(
            f"n_forces={n_forces} is too high for available pieces without duplicate indices."
        )
    force_body_ids = vec_bodyid_full[body_indices]
    for i, body_id in enumerate(force_body_ids):
        if body_id < 0 or body_id >= model.nbody:
            raise ValueError(f"Invalid body ID at force index {i}: {body_id} (nbody={model.nbody})")

    # Force parameters: sinusoidal rotation in y-z plane
    force_magnitude = np.sqrt(2) / 4  # Magnitude of force
    rotation_freq = 0.1  # Frequency in Hz (rotations per second)
    force_delay = 1.0  # Delay before forces start (seconds)
    phase_offsets = 2.0 * np.pi * np.arange(n_forces) / n_forces  # 2*pi/n phase spacing

    # Validate rotation frequency
    if rotation_freq < 0:
        raise ValueError(f"Rotation frequency must be non-negative, got {rotation_freq}")
    if rotation_freq == 0:
        print("Warning: rotation_freq is 0, force will be constant")

    while True:
        stored_torques, E_total = get_storedtorque(data)
        ntorq[:-3] = stored_torques.flatten()[3:]
        npos = data.xpos[vec_bodyid_full].copy().flatten()
        nquat = data.xquat[vec_bodyid_full].copy().flatten()

        # Calculate n time-dependent forces in the y-z plane with fixed phase offsets.
        omega = 2.0 * np.pi * rotation_freq
        f_api = np.zeros((n_forces, 3), dtype=np.float64)
        if cur_time >= force_delay:
            t_relative = cur_time - force_delay
            base_phase = omega * t_relative + np.pi
            f_api[:, 1] = force_magnitude * np.sin(base_phase + phase_offsets)
            f_api[:, 2] = force_magnitude * np.cos(base_phase + phase_offsets)

        # Apply each force at its body COM by converting to equivalent wrench at body origin.
        for i, body_id in enumerate(force_body_ids):
            com_pos = data.xipos[body_id].copy()
            body_origin = data.xpos[body_id].copy()
            offset = com_pos - body_origin
            data.xfrc_applied[body_id, :3] = f_api[i]
            data.xfrc_applied[body_id, 3:] = np.cross(offset, f_api[i])

        if n_checks % solve_freq == 0:
            t0 = time.perf_counter()
            solvable_check = ds2f.calculateExternalForces(ntorq, npos, nquat)
            timing_samples.append(time.perf_counter() - t0)
            solver_call_count += 1
            if solvable_check:
                solvable_true_count += 1
                n_force_detected = len(ds2f.force_sections)
                solved_forces = np.zeros((n_force_detected, 3), dtype=np.float64)
                solved_positions = np.zeros((n_force_detected, 3), dtype=np.float64)
                for i in range(n_force_detected):
                    solved_forces[i] = ds2f.force_sections[i].get_force()
                    solved_positions[i] = ds2f.force_sections[i].get_force_pos()
                last_solved_forces = solved_forces
                last_solved_positions = solved_positions

        progress = int(min(100, (cur_time / benchmark_seconds) * 100.0))
        if progress != last_progress:
            bar_len = 30
            filled = int((progress / 100.0) * bar_len)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\r[n_forces={n_forces}] |{bar}| {progress:3d}%", end="", flush=True)
            last_progress = progress

        if cur_time >= benchmark_seconds:
            print()
            break

        sim.step()
        sim.forward()
        if n_checks % 10 == 0 and do_render:
            # Visualize currently applied forces at each force application body COM.
            for body_id in force_body_ids:
                force_vec = data.xfrc_applied[body_id, :3].copy()
                force_pos = data.xipos[body_id].copy()
                viewer.add_vector_marker(
                    force_pos,
                    0.5 * force_vec,
                    width=0.002,
                    color=[0.1, 0.1, 0.1, 0.5]
                )
            # Visualize latest solved external forces.
            if last_solved_forces is not None and last_solved_positions is not None:
                for i in range(last_solved_forces.shape[0]):
                    rgba_arrow = [0.1, 1.0, 0.1, 0.5]
                    if i == 0 or i == last_solved_forces.shape[0] - 1:
                        rgba_arrow = [1.0, 0.1, 0.1, 0.5]
                    viewer.add_vector_marker(
                        last_solved_positions[i],
                        0.5 * last_solved_forces[i],
                        width=0.002,
                        color=rgba_arrow
                    )
            viewer.render()
        n_checks += 1
        cur_time += dt

    if do_render and viewer is not None:
        viewer.close()

    if len(timing_samples) == 0:
        return {
            "n_forces": n_forces,
            "sim_time": cur_time,
            "solver_calls": solver_call_count,
            "solvable_ratio": 0.0,
            "avg_time": np.nan,
            "var_time": np.nan,
        }

    timings = np.array(timing_samples, dtype=np.float64)
    return {
        "n_forces": n_forces,
        "sim_time": cur_time,
        "solver_calls": solver_call_count,
        "solvable_ratio": (solvable_true_count / solver_call_count) if solver_call_count > 0 else 0.0,
        "avg_time": float(np.mean(timings)),
        "var_time": float(np.var(timings)),
    }


all_results = []
for n_forces in n_force_lst:
    result = run_single_benchmark(n_forces)
    all_results.append(result)
    print(
        f"[n_forces={result['n_forces']}] calls={result['solver_calls']}, "
        f"solvable_ratio={result['solvable_ratio']:.4f}, "
        f"avg={result['avg_time']:.6e}s, var={result['var_time']:.6e}s^2"
    )

print("\n=== ds2f Speed Test Summary ===")
for r in all_results:
    print(
        f"n_forces={r['n_forces']}: "
        f"avg={r['avg_time']:.6e}s, var={r['var_time']:.6e}s^2, "
        f"calls={r['solver_calls']}, solvable_ratio={r['solvable_ratio']:.4f}"
    )
print("================================")

# Prepare data for plotting
n_forces_list = [r['n_forces'] for r in all_results]
avg_times = [r['avg_time'] for r in all_results]
# Convert variance to standard deviation
std_times = [np.sqrt(r['var_time']) if not np.isnan(r['var_time']) else 0.0 for r in all_results]

# Filter out NaN values
valid_indices = [i for i, avg in enumerate(avg_times) if not np.isnan(avg)]
if len(valid_indices) > 0:
    n_forces_plot = [n_forces_list[i] for i in valid_indices]
    avg_times_plot = [avg_times[i] for i in valid_indices]
    std_times_plot = [std_times[i] for i in valid_indices]
    
    # Generate plot
    script_dir = Path(__file__).parent
    graph_path = script_dir.parent / 'ds2f' / 'data' / 'figs'
    graph_path.mkdir(parents=True, exist_ok=True)
    plot_path = graph_path / 'ncontacts_speedtest_plot.pdf'
    
    # Convert times to milliseconds
    avg_times_plot_ms = [t * 1000.0 for t in avg_times_plot]
    std_times_plot_ms = [t * 1000.0 for t in std_times_plot]
    
    plot_nforces_timing(
        n_forces_list=n_forces_plot,
        avg_times=avg_times_plot_ms,
        sd_times=std_times_plot_ms,
        save_path=str(plot_path),
        show_plot=False,
        unit="ms"
    )
else:
    print("WARNING: No valid data points for plotting")

