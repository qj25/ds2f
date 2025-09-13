import os 
import mujoco
import mujoco_viewer
import numpy as np
import ds2f.utils.mjc2_utils as mjc2
from ds2f.utils.xml_utils import XMLWrapper
# from ds2f.controllers.wire_plugin.WireStandalone import WireStandalone
from ds2f.utils.mjc_utils import MjSimWrapper
from ds2f.utils.mjc2_utils import init_plugins
from ds2f.utils.smoothing_utils import get_initialsmooth
import ds2f.utils.dlo_s2f.Dlo_s2f as Dlo_s2f
# from ds2f.utils.real2sim_utils import compute_wire_frames
from ds2f.assets.genrope.gen_overall_native_xml import generate_overall_native_xml
from ds2f.utils.interp_utils import resample_wire_equal_distance_min_dev
import argparse

# Settings
use_specified_s2f = False
do_render = True
# update stiffness, mass, and length as needed.
alpha_bar = 0.001196450659614982    # Obtained from simple PI
beta_bar = 0.001749108044378543
mass_per_length = 0.079/2.98
thickness = 0.006
j_damp = 0.002

# init stiffnesses for capsule
J1 = np.pi * (thickness/2)**4/2.
Ix = np.pi * (thickness/2)**4/4.
stiff_vals = [
    beta_bar/J1,
    alpha_bar/Ix
]


n_pieces = 33   # adjust parllThreshold with change in n_pieces

use_predef_endspos = False
scale_factor = 1000.0

torq_tol = 1e-8
tolC2 = 3e-4
tolC3 = 3e-4
n_maxsmoothsteps = 1000000
n_avg = 1
m_p = 2.0e-3
# m_p = 8.0e-3
f_multi = 20.0
reset_freq = 3
pos_tol = 1e-7,
someStiff = 0

known_endspos = np.array([
    [176.70291008e-3,  291.7665392e-3,  1.03399510e-1],
    [-142.56707392e-3,  306.81070719e-3,  99.74315524e-3],
]) * 1000.0



# Init paths
assets_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'ds2f/assets'
)
xml_path = os.path.join(assets_path,'overall_s2f.xml')

init_plugins()





# Argument parser for user input
parser = argparse.ArgumentParser(description='WireStandalone Example')
parser.add_argument('--data_dir', type=str, default='dloimg_data', help='Directory containing dataset with .npy files')
parser.add_argument('--pos_file', type=str, default='pos3d_000000.npy', help='Numpy file with wire positions')
args = parser.parse_args()

if args.pos_file.isdigit():
    args.pos_file = "pos3d_" + args.pos_file + ".npy"





# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .npy file
npy_rel_path = f'../../data/{args.data_dir}/outputs'
npy_dir_path = os.path.normpath(os.path.join(script_dir, npy_rel_path))
npy_abs_path = npy_dir_path + f'/pos3d/{args.pos_file}'
# Load the numpy file
wire_pos = np.load(npy_abs_path)
print("Loaded wire_pos shape:", wire_pos.shape)
# input(f"wire_pos = {wire_pos}")
wire_pos_og = wire_pos.copy()
# if not use_predef_endspos:
    # known_endspos[0] = wire_pos[0].copy()
    # known_endspos[-1] = wire_pos[-1].copy()
    # known_endspos[0,2] += np.linalg.norm(wire_pos[0]-wire_pos[1])
    # known_endspos[-1,2] += np.linalg.norm(wire_pos[-1]-wire_pos[-2])
# wire_pos = np.vstack([known_endspos[0], wire_pos, known_endspos[1]])
# input(f"wire_pos = {wire_pos}")
# Edit wire pos
wire_pos /= scale_factor
wire_pos = resample_wire_equal_distance_min_dev(wire_pos, n_pieces, iters=300, step_size=0.4, rng_seed=0)
# End vertical ends
pos_tmp = wire_pos[0].copy()
wire_pos[0] = wire_pos[1].copy()
wire_pos[0,2] += np.linalg.norm(pos_tmp-wire_pos[1])
pos_tmp = wire_pos[-1].copy()
wire_pos[-1] = wire_pos[-2].copy()
wire_pos[-1,2] += np.linalg.norm(pos_tmp-wire_pos[-2])

# Smoothing
wire_pos = get_initialsmooth(
    wire_pos, mass_per_length, n_pieces, thickness, j_damp,
    stiff_vals, xml_path, n_maxsmoothsteps=n_maxsmoothsteps,
    do_render=True,
    n_avg=n_avg, m_p=m_p, reset_freq=reset_freq,
    f_multi=f_multi,
    pos_tol=pos_tol, someStiff=someStiff
)

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
    lookat=np.array([0.0, 0.0, 0.0])
    viewer.cam.distance = dist
    viewer.cam.azimuth = azi
    viewer.cam.elevation = elev
    viewer.cam.lookat = lookat
    viewer.render()
    viewer._paused = True

sim.forward()
# sim.step()
if do_render:
    viewer.render()
# sim.forward()

# Use Dlo_s2f package as in jointtorq_test.py
if not use_specified_s2f:
    # Example parameters (should match your model)
    ropemass = mass
    ds2f = Dlo_s2f.DLO_s2f(
        r_len, r_pieces, ropemass*(-model.opt.gravity[-1]),
        boolErrs=True, boolSolveTorq=True,
        torque_tolerance_=torq_tol,
        tolC2_=tolC2, tolC3_=tolC3
    )
    # Dummy data for ntorq, npos, nquat (replace with real data as needed)
    ntorq = np.zeros((r_pieces+1)*3)
    npos = np.zeros((r_pieces+1)*3)
    nquat = np.zeros((r_pieces+1)*4)
    
    ntorq = np.zeros_like(data.qfrc_passive.flatten())
    stored_torques, E_total = get_storedtorque()
    # ntorq[:-3] = data.qfrc_passive.flatten()[3:]
    ntorq[:-3] = stored_torques.flatten()[3:]
    npos = data.xpos[vec_bodyid_full].copy().flatten()
    nquat = data.xquat[vec_bodyid_full].copy().flatten()

    # Call calculateExternalForces
    solvable_check = ds2f.calculateExternalForces(ntorq, npos, nquat)
    print("solvable_check:", solvable_check)
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

if solvable_check:
    # convert to mm and save
    fp_scaled = fp * scale_factor
    npos_scaled = npos * scale_factor
    force_folder = '/forces'
    pos3d_folder = '/pos3d'
    file_id = "_".join(args.pos_file.split("_")[-1:]) 
    os.makedirs(npy_dir_path + force_folder, exist_ok=True)
    np.save(npy_dir_path + force_folder + f'/fp_{file_id}',fp_scaled)
    np.save(npy_dir_path + force_folder + f'/ef_{file_id}',ef)
    np.save(npy_dir_path + pos3d_folder + f'/pos3dsmooth_{file_id}',npos_scaled)


for i in range(20000000):
# for i in range(10000):
# for i in range(n_steps):
    # if i == 10000: data.eq_active[0] = 0
    # data.eq_active[0] = 0
    # print(data.qfrc_passive)
    # print(i)
    # stored_torques = get_storedtorque()
    # print("qfrc_passive main:",data.qfrc_passive.reshape((-1,3)))
    # print("stored_torques:",stored_torques.reshape((-1,3)))
    if solvable_check:
        for ii in range(n_force_detected):
            viewer.add_vector_marker(
                fp[ii], 0.5*ef[ii], width=0.002
            )
    sim.step()
    sim.forward()
    if do_render:
        viewer.render()