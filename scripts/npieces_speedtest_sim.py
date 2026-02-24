import os 
import time
import mujoco
import mujoco_viewer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ds2f.utils.mjc2_utils as mjc2
from ds2f.utils.xml_utils import XMLWrapper
from ds2f.utils.mjc_utils import MjSimWrapper
from ds2f.utils.mjc2_utils import init_plugins
import ds2f.utils.dlo_s2f.Dlo_s2f as Dlo_s2f
from ds2f.assets.genrope.gen_overall_native_xml import generate_overall_native_xml
from ds2f.utils.interp_utils import resample_wire_equal_distance_min_dev


def benchmark_npieces(n_pieces, benchmark_seconds=20.0, solve_freq=10):
    """
    Benchmark ds2f computation time for a given number of pieces.
    
    Args:
        n_pieces: Number of discrete pieces
        benchmark_seconds: Duration of benchmark in simulation seconds
        solve_freq: Frequency of solving (every N steps)
    
    Returns:
        Dictionary with:
            - avg_time: Average computation time per call (seconds)
            - std_time: Standard deviation of computation time
            - n_samples: Number of samples collected
    """
    # Settings
    do_render = False
    alpha_bar = 0.001196450659614982
    beta_bar = 0.001749108044378543
    mass_per_length = 0.079/2.98
    thickness = 0.006
    j_damp = 0.002
    someStiff = 0
    
    torq_tol = 1e-8
    tolC2 = 3e-4
    tolC3 = 3e-4
    
    assets_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'ds2f/assets'
    )
    xml_path = os.path.join(assets_path,'overall_s2f.xml')
    
    # init stiffnesses for capsule
    J1 = np.pi * (thickness/2)**4/2.
    Ix = np.pi * (thickness/2)**4/4.
    stiff_vals = [
        beta_bar/J1,
        alpha_bar/Ix
    ]
    
    wire_pos = np.array([
        [0.0,0,0],
        [0.0,0,-0.1],
        [0.30,0,-0.1],
        [0.30,0,0],
    ])
    wire_pos = resample_wire_equal_distance_min_dev(wire_pos, n_pieces, iters=300, step_size=0.4, rng_seed=0)
    
    # Generate wire xml
    segment_lengths = np.linalg.norm(np.diff(wire_pos, axis=0), axis=1)
    total_length = np.sum(segment_lengths)
    mass = mass_per_length * total_length
    
    rgba_wire = "0.1 0.0533333 0.673333 1"
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
    
    xml_string = xml.get_xml_string()
    model = mujoco.MjModel.from_xml_string(xml_string)
    mujoco.mj_saveLastXML(xml_path,model)
    data = mujoco.MjData(model)
    model.opt.gravity[-1] = -9.81
    
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
            stored_torques = data.plugin_state[start:start+model.nv]
            E_total = data.plugin_state[-1]
        else:
            stored_torques = data.plugin_state[start:start+model.nv]
            E_total = data.plugin_state[start+model.nv]
        return stored_torques, E_total
    
    sim = MjSimWrapper(model, data)
    
    dt = model.opt.timestep
    cur_time = 0.0
    
    sim.forward()
    sim.step()
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
    
    # Find the body IDs at 1/3 and 2/3 points of the wire
    body_idx_1 = int(n_pieces / 3)
    body_idx_2 = int(2 * n_pieces / 3)
    body_id_1 = vec_bodyid_full[body_idx_1]
    body_id_2 = vec_bodyid_full[body_idx_2]
    
    # Force parameters
    force_magnitude = np.sqrt(2) / 4
    rotation_freq = 0.1
    force_delay = 1.0
    
    # Timing storage
    computation_times = []
    n_checks = 0
    
    # Run benchmark
    while cur_time < benchmark_seconds:
        stored_torques, E_total = get_storedtorque()
        ntorq[:-3] = stored_torques.flatten()[3:]
        npos = data.xpos[vec_bodyid_full].copy().flatten()
        nquat = data.xquat[vec_bodyid_full].copy().flatten()
        
        # Calculate time-dependent forces
        omega = 2.0 * np.pi * rotation_freq
        if cur_time < force_delay:
            f_api_1 = np.array([0., 0., 0.])
            f_api_2 = np.array([0., 0., 0.])
        else:
            t_relative = cur_time - force_delay
            f_api_1 = np.array([
                0.0,
                force_magnitude * np.sin(omega * t_relative + np.pi),
                force_magnitude * np.cos(omega * t_relative + np.pi)
            ], dtype=np.float64)
            f_api_2 = np.array([
                0.0,
                force_magnitude * np.sin(omega * t_relative + np.pi + np.pi/2),
                force_magnitude * np.cos(omega * t_relative + np.pi + np.pi/2)
            ], dtype=np.float64)
        
        # Apply forces at mass centers
        com_pos_1 = data.xipos[body_id_1].copy()
        com_pos_2 = data.xipos[body_id_2].copy()
        body_origin_1 = data.xpos[body_id_1].copy()
        body_origin_2 = data.xpos[body_id_2].copy()
        offset_1 = com_pos_1 - body_origin_1
        offset_2 = com_pos_2 - body_origin_2
        
        data.xfrc_applied[body_id_1, :3] = f_api_1
        data.xfrc_applied[body_id_1, 3:] = np.cross(offset_1, f_api_1)
        data.xfrc_applied[body_id_2, :3] = f_api_2
        data.xfrc_applied[body_id_2, 3:] = np.cross(offset_2, f_api_2)
        
        # Time the ds2f computation
        if n_checks % solve_freq == 0:
            start_time = time.perf_counter()
            solvable_check = ds2f.calculateExternalForces(ntorq, npos, nquat)
            end_time = time.perf_counter()
            
            if solvable_check:
                computation_times.append(end_time - start_time)
        
        sim.step()
        sim.forward()
        n_checks += 1
        cur_time += dt
    
    # Calculate statistics
    if len(computation_times) > 0:
        avg_time = np.mean(computation_times)
        std_time = np.std(computation_times)
        n_samples = len(computation_times)
    else:
        avg_time = None
        std_time = None
        n_samples = 0
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'n_samples': n_samples
    }


if __name__ == "__main__":
    # Initialize plugins
    init_plugins()
    
    # Benchmark parameters
    n_pieces_list = [10, 20, 30, 50, 100, 150]
    benchmark_seconds = 20.0
    solve_freq = 10
    
    print(f"Benchmarking ds2f computation time for n_pieces: {n_pieces_list}")
    print(f"Benchmark duration: {benchmark_seconds} simulation seconds per configuration")
    print()
    
    # Storage for results
    results = {
        'n_pieces': [],
        'avg_times': [],
        'std_times': [],
        'n_samples': []
    }
    
    # Run benchmarks
    for i, n_pieces in enumerate(n_pieces_list):
        print(f"[{i+1}/{len(n_pieces_list)}] Benchmarking n_pieces={n_pieces}...")
        result = benchmark_npieces(n_pieces, benchmark_seconds, solve_freq)
        
        if result['avg_time'] is not None:
            results['n_pieces'].append(n_pieces)
            results['avg_times'].append(result['avg_time'])
            results['std_times'].append(result['std_time'])
            results['n_samples'].append(result['n_samples'])
            print(f"n_pieces={n_pieces:3d}: avg_time={result['avg_time']*1000:.3f}ms, "
                  f"std={result['std_time']*1000:.3f}ms, samples={result['n_samples']}")
        else:
            print(f"n_pieces={n_pieces:3d}: No valid samples collected")
    
    print()
    print(f"Completed {len(results['n_pieces'])} successful benchmarks")
    
    # Check if we have results to plot
    if len(results['n_pieces']) == 0:
        print("ERROR: No successful benchmarks. Cannot generate plot.")
        exit(1)
    
    # Convert to numpy arrays
    n_pieces_arr = np.array(results['n_pieces'])
    avg_times_arr = np.array(results['avg_times'])
    std_times_arr = np.array(results['std_times'])
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot with error bars
    # Custom colors: line uses #035993, error bars use a slightly lighter variant
    line_color = '#035993'  # Hex color directly
    # For error bars, use RGB tuple with opacity: (R/255, G/255, B/255, alpha)
    # #035993 = RGB(3, 89, 147), using a lighter variant for error bars
    error_color = '#8ea6c5' # RGBA tuple with opacity
    
    ax.errorbar(n_pieces_arr, avg_times_arr * 1000, yerr=std_times_arr * 1000,
                fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=1.5,
                label='ds2f computation time', color=line_color,
                ecolor=error_color, elinewidth=1.5)
    
    ax.set_xlabel('Number of Discrete Pieces ($n_{pieces}$)', fontsize=12)
    ax.set_ylabel('Computation Time (ms)', fontsize=12)
    ax.set_title('ds2f Computation Time vs. Number of Pieces', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    script_dir = Path(__file__).parent
    graph_path = script_dir.parent / 'ds2f' / 'data' / 'figs'
    graph_path.mkdir(parents=True, exist_ok=True)
    plot_path = graph_path / 'npieces_speedtest_plot.pdf'
    plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"\nPlot saved to: {plot_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Number of pieces tested: {len(n_pieces_arr)}")
    print(f"Computation time range: [{np.min(avg_times_arr)*1000:.3f}ms, {np.max(avg_times_arr)*1000:.3f}ms]")
    print(f"Average samples per configuration: {np.mean(results['n_samples']):.1f}")

