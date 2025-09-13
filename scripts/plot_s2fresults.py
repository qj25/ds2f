import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from ds2f.utils.dlo_utils import ang_btwn
from mpl_toolkits.mplot3d import Axes3D

def plot_wire_with_forces(wire_points, fpos_list, fvec_list, plot_path=None, y_heatmap_range=(150, 450)):
    """
    Plots a 3D wire and multiple sets of force vectors with distinct colors.

    Args:
        wire_points (np.ndarray): (N, 3) array of wire points.
        fpos_list (list[np.ndarray]): List of (M_i, 3) arrays of vector origins.
        fvec_list (list[np.ndarray]): List of (M_i, 3) arrays of vector directions.
        y_heatmap_range (tuple): Range for Y-axis heatmap coloring.
    """
    assert wire_points.shape[1] == 3, "wire_points must be (N, 3)"
    assert len(fpos_list) == len(fvec_list), "Mismatch between fpos and fvec sets"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=17, azim=-57)
    # ax.grid(False)

    ax.xaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)  # Light gray, semi-transparent
    ax.yaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)
    ax.zaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)

    # --- Heatmap coloring based on y values ---
    y_vals = wire_points[:, 1]
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=y_heatmap_range[0], vmax=y_heatmap_range[1])
    colors = cmap(norm(y_vals))
    colors[:, 3] = 1.0  # Remove transparency

    # Plot wire with heatmap colors
    c_wire = 'blue'
    # ax.scatter(wire_points[:, 0], wire_points[:, 1], wire_points[:, 2], c=colors, s=10)
    ax.scatter(wire_points[:, 0], wire_points[:, 1], wire_points[:, 2], c=c_wire, alpha=0.2, s=10)
    ax.plot(wire_pos[:, 0], wire_pos[:, 1], wire_pos[:, 2], color=c_wire, alpha=0.5, linewidth=2)

    # --- Plot multiple sets of force vectors ---
    color_cycle = ['red', 'black', 'green', 'green']

    # --- Get maximum force ---
    def get_max_force_scale(fvec_list, f_thres=100.0):
        max_magnitude = 0.0
        for fvec in fvec_list:
            if len(fvec) > 0:
                magnitudes = np.linalg.norm(fvec, axis=1)
                max_magnitude = max(max_magnitude, np.max(magnitudes))
        if max_magnitude > f_thres:
            return f_thres/max_magnitude
        return 1.0
    f_scale = get_max_force_scale(fvec_list)

    for i, (fpos, fvec) in enumerate(zip(fpos_list, fvec_list)):
        if len(fpos.shape) > 2:
            fpos = fpos[0]
            fvec = fvec[0]
        assert fpos.shape == fvec.shape and fpos.shape[1] == 3, f"fpos/fvec at index {i} must have shape (N, 3)"
        color = color_cycle[i % len(color_cycle)]
        for origin, vector in zip(fpos, fvec):
            vec_scaled = vector * f_scale
            vector /= np.linalg.norm(vector)
            ax.quiver(
                origin[0], origin[1], origin[2],
                vector[0], vector[1], vector[2],
                color=color,
                length=np.linalg.norm(vec_scaled),
                normalize=False,
                arrow_length_ratio=0.15,
                alpha=0.7
            )

    # --- Equal scaling for all axes ---
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range)
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    set_axes_equal(ax)
    # ax.set_xticks(np.arange(-200, 200, 100))  # Change range and step as needed
    # ax.set_yticks(np.arange(200, 400, 100))
    # ax.set_zticks(np.arange(-150, 150, 100))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("X", labelpad=-14)
    ax.set_ylabel("Y", labelpad=-14)
    ax.set_zlabel("Z", labelpad=-14)
    # plt.title("3D Wire with Multiple Force Vector Sets")
    plt.rcParams.update({'pdf.fonttype': 42})
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()

def get_fractional_point(wire_points, fraction):
    """
    Returns the 3D point at a given fractional arc length along the wire.

    Args:
        wire_points (np.ndarray): (N, 3) array of 3D points along the wire.
        fraction (float): Desired fraction along the arc length (0.0 to 1.0).

    Returns:
        np.ndarray: 3D point at the specified fraction.
    """
    wire_points = np.asarray(wire_points)
    assert 0.0 <= fraction <= 1.0, "Fraction must be between 0 and 1"
    diffs = np.linalg.norm(np.diff(wire_points, axis=0), axis=1)
    cum_dist = np.concatenate([[0], np.cumsum(diffs)])
    total_length = cum_dist[-1]
    target_length = fraction * total_length

    # Find the segment that contains the target length
    idx = np.searchsorted(cum_dist, target_length) - 1
    idx = np.clip(idx, 0, len(diffs) - 1)

    # Linearly interpolate within the segment
    seg_start = wire_points[idx]
    seg_end = wire_points[idx + 1]
    seg_len = diffs[idx]
    seg_fraction = (target_length - cum_dist[idx]) / seg_len

    return seg_start + seg_fraction * (seg_end - seg_start)

def average_last_n_rows(csv_path, n=100):
    # Count total lines in the file
    with open(csv_path) as f:
        total_lines = sum(1 for _ in f)

    # Load only the last `n` rows, no header
    df = pd.read_csv(csv_path, header=None, skiprows=range(total_lines - n))

    # Convert result to NumPy array (shape: (6,))
    return df.mean().to_numpy()

## Settings
frac_actualf = 0.5
# frac_actualf = 32/50
f_scale = 100.0

# Argument parser for user input
parser = argparse.ArgumentParser(description='WireStandalone Example')
parser.add_argument('--data_dir', type=str, default='dloimg_data', help='Directory containing dataset with .npy files')
parser.add_argument('--file_id', type=str, default='000000', help='Numpy file with wire positions')
args = parser.parse_args()

# --- Example usage ---
# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .npy file
npy_rel_path = f'../../data/{args.data_dir}/outputs'
npy_dir_path = os.path.normpath(os.path.join(script_dir, npy_rel_path))
file_id = args.file_id + '.npy'
force_folder = '/forces'
pos3d_folder = '/pos3d'
fpos = np.load(npy_dir_path + force_folder + f'/fp_{file_id}')
fvec = np.load(npy_dir_path + force_folder + f'/ef_{file_id}')
wire_pos = np.load(npy_dir_path + pos3d_folder + f'/pos3dsmooth_{file_id}')
wire_pos = wire_pos.reshape((-1,3))
fvec *= f_scale

# Actual force data
actualf_folder = '../actual_f'
actualf_dir = os.path.normpath(os.path.join(npy_dir_path,actualf_folder))
avgvec = average_last_n_rows(actualf_dir + f'/ft_data_{args.file_id}.csv', n=100)[:3]
avgvec *= f_scale
# global xyz - ftsensor x-zy
frame_rot = -np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])
avgvec = frame_rot.dot(avgvec)
avg_pos = get_fractional_point(wire_pos, frac_actualf)
print(f"avgvec = {avgvec}")
print(f"fvec = {fvec}")

fvec_list = [
    np.vstack((fvec[0],fvec[-1])),
    np.array([avgvec]),
    np.array([fvec[1:-1]]),
]
fpos_list = [
    np.vstack((fpos[0],fpos[-1])),
    np.array([avg_pos]),
    np.array([fpos[1:-1]]),
]

plot_path = '../plots'
plot_path = os.path.normpath(os.path.join(npy_dir_path,plot_path))
os.makedirs(plot_path, exist_ok=True)
plot_path = plot_path + f'/plt_{args.file_id}.pdf'

plot_wire_with_forces(wire_pos, fpos_list, fvec_list, plot_path, y_heatmap_range=(300, 370))

f_real = avgvec.copy()/f_scale
fp_real = avg_pos
fvec_spare = fvec[1:-1]
fpos_spare = fpos[1:-1]
f_id = np.argmax(np.linalg.norm(fvec_spare, axis=1))
f_est = fvec_spare[f_id]/f_scale
fp_est = fpos_spare[f_id]

print("f_actual:", np.array2string(f_real, separator=', ', formatter={'float_kind': lambda x: f"{x:.3f}"}))
print("f_est:", np.array2string(f_est, separator=', ', formatter={'float_kind': lambda x: f"{x:.3f}"}))
print("L2 error:", np.linalg.norm(f_est - f_real))
print("Angle (rad):", ang_btwn(f_est, f_real))
# print("Relative error:", np.linalg.norm(f_est - f_real) / (np.linalg.norm(f_real) + 1e-8))
print("Position difference:", np.linalg.norm(fp_est - fp_real))