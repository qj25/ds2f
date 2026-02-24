import os 
import time
import mujoco
import mujoco_viewer
import numpy as np
import ds2f.utils.mjc2_utils as mjc2
from ds2f.utils.xml_utils import XMLWrapper
from ds2f.utils.mjc_utils import MjSimWrapper
from ds2f.utils.mjc2_utils import init_plugins
import ds2f.utils.dlo_s2f.Dlo_s2f as Dlo_s2f
from ds2f.assets.genrope.gen_overall_native_xml import generate_overall_native_xml
from ds2f.utils.interp_utils import resample_wire_equal_distance_min_dev
from ds2f.utils.plotter import plotter_process_force, plotter_process_fp
import multiprocessing as mp

# Settings
do_render = True
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

# RT plotter
bool_rtplot = True
base_t = 1.0
plot_active = False

assets_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'ds2f/assets'
)
xml_path = os.path.join(assets_path,'overall_s2f.xml')
graph_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'ds2f/data/figs'
)
forcegraph_path1 = os.path.join(graph_path,'force1_magnitude_interactsim.pdf')
forcegraph_path2 = os.path.join(graph_path,'force2_magnitude_interactsim.pdf')

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

def get_storedtorque():
    if plgn_instance == model.nplugin - 1:
        # Last plugin - use remaining state
        stored_torques = data.plugin_state[start:start+model.nv]
        E_total = data.plugin_state[-1]
    else:
        # Not last plugin - use next plugin's start as end
        # stored_torques = data.plugin_state[start:model.plugin_stateadr[plgn_instance+1]-1]
        stored_torques = data.plugin_state[start:start+model.nv]
        E_total = data.plugin_state[start+model.nv]
    return stored_torques, E_total

sim = MjSimWrapper(model, data)
if do_render:
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.vopt.geomgroup[3] ^= 1
    dist=0.5
    azi=90.0
    elev=0.0
    lookat=np.array([0.15, 0.0, 0.0])
    viewer.cam.distance = dist
    viewer.cam.azimuth = azi
    viewer.cam.elevation = elev
    viewer.cam.lookat = lookat
    viewer.render()
    viewer._paused = True
    viewer.perturbation_scale = 3.0

dt = model.opt.timestep
cur_time = 0.0

sim.forward()
# print(f"xpos = {data.xpos[vec_bodyid_full]}")
# print(f"xquat = {data.xquat[vec_bodyid_full]}")
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
# Dummy data for ntorq, npos, nquat (replace with real data as needed)
ntorq = np.zeros((r_pieces+1)*3)
npos = np.zeros((r_pieces+1)*3)
nquat = np.zeros((r_pieces+1)*4)

ntorq = np.zeros_like(data.qfrc_passive.flatten())

queue1 = mp.Queue()
p1 = mp.Process(
    target=plotter_process_force,
    args=(queue1, forcegraph_path1)
)
p1.start()
queue2 = mp.Queue()
p2 = mp.Process(
    target=plotter_process_force,
    args=(queue2, forcegraph_path2)
)
p2.start()


n_checks = 0
offline_t = 0.0

# Find the body IDs at 1/3 and 2/3 points of the wire
body_idx_1 = int(n_pieces / 3)
body_idx_2 = int(2 * n_pieces / 3)
body_id_1 = vec_bodyid_full[body_idx_1]
body_id_2 = vec_bodyid_full[body_idx_2]

# Verify body IDs are valid
if body_id_1 < 0 or body_id_1 >= model.nbody:
    raise ValueError(f"Invalid body ID 1: {body_id_1} (nbody={model.nbody})")
if body_id_2 < 0 or body_id_2 >= model.nbody:
    raise ValueError(f"Invalid body ID 2: {body_id_2} (nbody={model.nbody})")

# Force parameters: sinusoidal rotation in y-z plane
force_magnitude = np.sqrt(2) / 4  # Magnitude of force
rotation_freq = 0.1  # Frequency in Hz (rotations per second)
force_delay = 1.0  # Delay before forces start (seconds)

# Validate rotation frequency
if rotation_freq < 0:
    raise ValueError(f"Rotation frequency must be non-negative, got {rotation_freq}")
if rotation_freq == 0:
    # Zero frequency means constant force - use initial direction
    print("Warning: rotation_freq is 0, force will be constant")

while True:
    stored_torques, E_total = get_storedtorque()
    ntorq[:-3] = stored_torques.flatten()[3:]
    npos = data.xpos[vec_bodyid_full].copy().flatten()
    nquat = data.xquat[vec_bodyid_full].copy().flatten()

    # Calculate time-dependent forces: sinusoidal rotation in y-z plane
    # Force 1 starts in negative z direction after delay; Force 2 is pi/2 out of phase.
    omega = 2.0 * np.pi * rotation_freq
    if cur_time < force_delay:
        f_api_1 = np.array([0., 0., 0.])
        f_api_2 = np.array([0., 0., 0.])
    else:
        t_relative = cur_time - force_delay
        f_api_1 = np.array([
            0.0,  # x component (no force in x direction)
            force_magnitude * np.sin(omega * t_relative + np.pi),  # y component (phase shifted)
            force_magnitude * np.cos(omega * t_relative + np.pi)   # z component (starts at -z)
        ], dtype=np.float64)
        f_api_2 = np.array([
            0.0,  # x component (no force in x direction)
            force_magnitude * np.sin(omega * t_relative + np.pi + np.pi/2),  # y component
            force_magnitude * np.cos(omega * t_relative + np.pi + np.pi/2)   # z component
        ], dtype=np.float64)
    
    # Get mass center positions (xipos gives center of mass in world coordinates)
    com_pos_1 = data.xipos[body_id_1].copy()
    com_pos_2 = data.xipos[body_id_2].copy()
    
    # Get body origin positions
    body_origin_1 = data.xpos[body_id_1].copy()
    body_origin_2 = data.xpos[body_id_2].copy()
    
    # Calculate offset from body origin to center of mass
    offset_1 = com_pos_1 - body_origin_1
    offset_2 = com_pos_2 - body_origin_2
    
    # Apply force at mass center by converting to equivalent force and torque at body origin
    # Force at origin = Force at COM
    # Torque at origin = (COM_pos - origin_pos) × Force
    data.xfrc_applied[body_id_1, :3] = f_api_1
    data.xfrc_applied[body_id_1, 3:] = np.cross(offset_1, f_api_1)
    data.xfrc_applied[body_id_2, :3] = f_api_2
    data.xfrc_applied[body_id_2, 3:] = np.cross(offset_2, f_api_2)
    
    # Get positions where forces are applied (at the mass centers)
    fvpos_api_1 = com_pos_1.copy()
    fvpos_api_2 = com_pos_2.copy()
    
    if n_checks % solve_freq == 0:
        solvable_check = ds2f.calculateExternalForces(ntorq, npos, nquat)
        # print("solvable_check:", solvable_check)
        if solvable_check:
            n_force_detected = len(ds2f.force_sections)
            ef = np.zeros((n_force_detected,3))
            et = np.zeros((n_force_detected,3))
            fp = np.zeros((n_force_detected,3))
            for ii in range(n_force_detected):
                ef[ii] = ds2f.force_sections[ii].get_force()
                et[ii] = ds2f.force_sections[ii].get_torque()
                fp[ii] = ds2f.force_sections[ii].get_force_pos()
            print("external_force:", ef)
            print("external_torque:", et) 
            print("external_position:", fp) 

            print("api_force_1:", f_api_1)
            print("api_forcepos_1:", fvpos_api_1)
            print("api_force_2:", f_api_2)
            print("api_forcepos_2:", fvpos_api_2)

            # make arrow vector for estimated forces
            if do_render:
                for ii in range(n_force_detected):
                    rgba_arrow = [0.1,1.0,0.1,0.5]
                    if (ii == 0 or ii == n_force_detected-1):
                        rgba_arrow = [1.0,0.1,0.1,0.5]
                    viewer.add_vector_marker(
                        fp[ii], 0.5*ef[ii], width=0.002, color=rgba_arrow
                    )
                rgba_arrow = [0.1,0.1,0.1,0.5]
                viewer.add_vector_marker(
                    fvpos_api_1, 0.5*f_api_1, width=0.002, color=rgba_arrow
                )
                viewer.add_vector_marker(
                    fvpos_api_2, 0.5*f_api_2, width=0.002, color=rgba_arrow
                )
            
            # Use forces in order (excluding endpoints) and compare in plot
            if (
                bool_rtplot 
                and cur_time>base_t 
                and len(ef) >= 4
            ): 
                # Use forces in order: ef[1] and ef[2] (excluding endpoints ef[0] and ef[-1])
                f_pred_1 = ef[1]
                f_pred_2 = ef[2]

                # API forces are always applied, so always plot
                plot_active = True

                # plotting force vector components for force 1
                force_dict_1 = {
                    "$X$": f_api_1[0],
                    "$Y$": f_api_1[1],
                    "$Z$": f_api_1[2],
                    "$X_{est}$": f_pred_1[0],
                    "$Y_{est}$": f_pred_1[1],
                    "$Z_{est}$": f_pred_1[2],
                }
                queue1.put((cur_time-offline_t, force_dict_1))

                # plotting force vector components for force 2
                force_dict_2 = {
                    "$X$": f_api_2[0],
                    "$Y$": f_api_2[1],
                    "$Z$": f_api_2[2],
                    "$X_{est}$": f_pred_2[0],
                    "$Y_{est}$": f_pred_2[1],
                    "$Z_{est}$": f_pred_2[2],
                }
                queue2.put((cur_time-offline_t, force_dict_2))
        else:
            # If solvable_check is False, we don't have estimated forces, so don't plot
            pass

        if not plot_active:
            offline_t += dt * solve_freq
        # if (np.linalg.norm(f_viewer[:3])<1e-6):
            # offline_t += dt * solve_freq
    sim.step()
    sim.forward()
    if n_checks % 10 == 0:
        if do_render:
            viewer.render()
    n_checks += 1
    cur_time += dt

