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
from ds2f.utils.dlo_utils import point_arc_length
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

n_pieces = 33   # adjust parllThreshold with change in n_pieces
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
forcegraph_path = os.path.join(graph_path,'force_interactsim.pdf')
positiongraph_path = os.path.join(graph_path,'position_interactsim.pdf')

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
    lookat=np.array([0.0, 0.0, 0.0])
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
    boolErrs=False, boolSolveTorq=False,
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
    args=(queue1, forcegraph_path)
    # args=(queue, num_series, None)
)
p1.start()
queue2 = mp.Queue()
p2 = mp.Process(
    target=plotter_process_fp,
    args=(queue2, positiongraph_path)
    # args=(queue, num_series, None)
)
p2.start()


n_checks = 0
offline_t = 0.0

while True:
    stored_torques, E_total = get_storedtorque()
    ntorq[:-3] = stored_torques.flatten()[3:]
    npos = data.xpos[vec_bodyid_full].copy().flatten()
    nquat = data.xquat[vec_bodyid_full].copy().flatten()

    add_fusr = False
    if add_fusr:
        # have to apply at every step
        # as each render call zeros current xfrc_applied
        objid = mjc2.obj_name2id(
            model,
            "body", "B_12"
        )
        objid2 = mjc2.obj_name2id(
            model,
            "body", "B_16"
        )
        f_usr = np.array([0., 0., 0.1, 0.0, 0.0, 0.])
        data.xfrc_applied[objid] += f_usr.copy()
        fpos_usr = (
            data.xpos[objid]
            + data.xpos[objid+1]
        ) / 2.0
    fvpos_usr, f_viewer, f_active = viewer.getuserFT()
    # print("f_viewer:", f_viewer)
    
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

            print("usr_force:", f_viewer)
            print("usr_forcepos:", fvpos_usr)

            # make arrow vector for estimated forces
            for ii in range(n_force_detected):
                rgba_arrow = [0.1,1.0,0.1,0.5]
                if (ii == 0 or ii == n_force_detected-1):
                    rgba_arrow = [1.0,0.1,0.1,0.5]
                viewer.add_vector_marker(
                    fp[ii], 0.5*ef[ii], width=0.002, color=rgba_arrow
                )
            rgba_arrow = [0.1,0.1,0.1,0.5]
            viewer.add_vector_marker(
                fvpos_usr, 0.5*f_viewer[:3], width=0.002, color=rgba_arrow
            )
            
            # find closest ef and compare in plot
            if (
                bool_rtplot 
                and cur_time>base_t 
            ): 
                if len(ef) < 3:
                    f_pred = [None,None,None]
                    l_pos_est = None
                else:
                    f_diff = np.linalg.norm(ef[1:-1] - f_viewer[:3], axis=1)
                    id_closest = np.argmin(f_diff) + 1
                    f_pred = ef[id_closest]
                    fp_pred = fp[id_closest]
                    l_pos_est = point_arc_length(npos.reshape((n_pieces,3)), fp_pred)
                    # if l_pos_est < 0.05 or l_pos_est > 0.40:
                    #     print("Large deviation in force prediction, pausing |========================")
                    #     print(f"prediction OFF: {fp_pred}")
                    #     print(f"l_pos_est: {l_pos_est}")
                    #     print("external_force:", ef)
                    #     print("external_torque:", et) 
                    #     print("external_position:", fp)
                    #     viewer._paused = True


                l_pos_act = point_arc_length(npos.reshape((n_pieces,3)), fvpos_usr)

                if (np.linalg.norm(f_viewer[:3])<1e-6):
                    f_viewer[:3] = [None,None,None]
                    f_pred = [None,None,None]
                    l_pos_est = None
                    l_pos_act = None
                else:
                    plot_active = True

                # plotting force vector
                force_dict = {
                    "$X$": f_viewer[0],
                    "$Y$": f_viewer[1],
                    "$Z$": f_viewer[2],
                    "$X_{est}$": f_pred[0],
                    "$Y_{est}$": f_pred[1],
                    "$Z_{est}$": f_pred[2],
                }
                queue1.put((cur_time-offline_t, force_dict))

                # plotting arc length location
                fp_dict = {
                    "$S$": l_pos_act,
                    "$S_{est}$": l_pos_est,
                }
                queue2.put((cur_time-offline_t, fp_dict))
        else:
            if (
                bool_rtplot 
                and cur_time>base_t
            ): 
                f_pred = [None,None,None]
                l_pos_est = None
                if (np.linalg.norm(f_viewer[:3])<1e-6):
                    f_viewer[:3] = [None,None,None]
                    l_pos_act = None
                else:
                    plot_active = True
                    
                l_pos_act = point_arc_length(npos.reshape((n_pieces,3)), fvpos_usr)

                # plotting force vector
                force_dict = {
                    "$X$": f_viewer[0],
                    "$Y$": f_viewer[1],
                    "$Z$": f_viewer[2],
                    "$X_{est}$": f_pred[0],
                    "$Y_{est}$": f_pred[1],
                    "$Z_{est}$": f_pred[2],
                }
                queue1.put((cur_time-offline_t, force_dict))
                # plotting arc length location
                fp_dict = {
                    "$S$": l_pos_act,
                    "$S_{est}$": l_pos_est,
                }
                queue2.put((cur_time-offline_t, fp_dict))

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
