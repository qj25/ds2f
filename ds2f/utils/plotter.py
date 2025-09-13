import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
import matplotlib.animation as animation
import pandas as pd

plt.rcParams.update({'pdf.fonttype': 42})   # to prevent type 3 fonts in pdflatex
img_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/figs/"
)

def plotter_process_indiv(queue: mp.Queue, num_series=2, save_path="realtime_plot.png"):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Values")
    ax.set_title("Realtime Multi-Series Plot")

    # initialize buffers and line objects
    xdata = []
    ydata = [ [] for _ in range(num_series) ]
    lines = [ax.plot([], [], lw=2, label=f"Series {i+1}")[0] for i in range(num_series)]
    ax.legend()

    def update(_):
        # pull all available data from the queue
        while not queue.empty():
            item = queue.get()
            t = item[0]
            vals = item[1]  # list/tuple of values, length = num_series
            xdata.append(t)
            for i in range(num_series):
                ydata[i].append(vals[i])

        if xdata:
            for i in range(num_series):
                lines[i].set_data(xdata, ydata[i])
            ax.relim()
            ax.autoscale_view()
        return lines

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False)

    try:
        plt.show(block=True)
    finally:
        if xdata:
            fig.savefig(save_path)
            print(f"Plot saved to {save_path}")

def plotter_process_force(queue: mp.Queue, save_path="fsim_plot.png", data_path=None):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.set_title("Realtime Force Plot")
    ax.grid(True)  # enables grid lines

    # List of series names
    series_names = ["$X$", "$Y$", "$Z$", "$X_{est}$", "$Y_{est}$", "$Z_{est}$"]
    colors = ["r", "g", "b"]
    lines = []

    # Initialize line objects
    for i, name in enumerate(series_names):
        color = colors[i % 3]
        linestyle = '-' if i < 3 else ':'
        alpha = 0.3 if i < len(series_names)/2 else 0.7
        line, = ax.plot([], [], lw=2, color=color, alpha=alpha, linestyle=linestyle, label=name)
        lines.append(line)
    ax.legend()

    xdata = []
    ydata = {name: [] for name in series_names}

    def update(_):
        while not queue.empty():
            t, val_dict = queue.get()
            xdata.append(t)
            for i, name in enumerate(series_names):
                val = val_dict.get(name)
                if val is not None:
                    ydata[name].append(val)
                else:
                    # If None, append last value or NaN to keep array lengths consistent
                    ydata[name].append(np.nan)  

        if xdata:
            for i, name in enumerate(series_names):
                lines[i].set_data(xdata, ydata[name])
            ax.relim()
            ax.autoscale_view()
        return lines

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False)

    plt.show(block=True)

    if xdata:
        # Save plot
        if save_path is not None:
            fig.savefig(save_path)
            print(f"Plot saved to {save_path}")
        # Save data
        if data_path is not None:
            df = pd.DataFrame({"Time": xdata})
            for name in series_names:
                df[name] = ydata[name]
            df.to_csv(data_path, index=False)
            print(f"Data saved to {data_path}")

def plotter_process_fp(queue: mp.Queue, save_path="fpsim_plot.png", data_path=None):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Realtime Position Plot")
    ax.grid(True)  # enables grid lines

    # List of series names
    series_names = ["$S$", "$S_{est}$"]
    colors = ["k"]
    lines = []

    # Initialize line objects
    for i, name in enumerate(series_names):
        color = colors[int(i % len(series_names)/2)]
        linestyle = '-' if i < len(series_names)/2 else ':'
        alpha = 0.5 if i < len(series_names)/2 else 0.7
        line, = ax.plot([], [], lw=2, color=color, alpha=alpha, linestyle=linestyle, label=name)
        lines.append(line)
    ax.legend()

    xdata = []
    ydata = {name: [] for name in series_names}

    def update(_):
        while not queue.empty():
            t, val_dict = queue.get()
            xdata.append(t)
            for i, name in enumerate(series_names):
                val = val_dict.get(name)
                if val is not None:
                    ydata[name].append(val)
                else:
                    # If None, append last value or NaN to keep array lengths consistent
                    ydata[name].append(np.nan)  

        if xdata:
            for i, name in enumerate(series_names):
                lines[i].set_data(xdata, ydata[name])
            ax.relim()
            ax.autoscale_view()
        return lines

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False)

    plt.show(block=True)

    if xdata:
        # Save plot
        if save_path is not None:
            fig.savefig(save_path)
            print(f"Plot saved to {save_path}")
        # Save data
        if data_path is not None:
            df = pd.DataFrame({"Time": xdata})
            for name in series_names:
                df[name] = ydata[name]
            df.to_csv(data_path, index=False)
            print(f"Data saved to {data_path}")

def plot3d(nodes,nodes2=None):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points as a scatter plot
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='r', marker='o')
    
    # Optionally, connect the nodes with lines (to visualize the path)
    ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='b')
    if nodes2 is not None:
        nodes2 = np.array(nodes2)
        if nodes2.ndim < 3:
            ax.scatter(nodes2[:, 0], nodes2[:, 1], nodes2[:, 2], c='y', marker='o')
            ax.plot(nodes2[:, 0], nodes2[:, 1], nodes2[:, 2], c='g')
        else:
            for i in range(len(nodes2)):
                ax.scatter(nodes2[i,:, 0], nodes2[i,:, 1], nodes2[i,:, 2], c='y', marker='o')
                ax.plot(nodes2[i,:, 0], nodes2[i,:, 1], nodes2[i,:, 2], c='g',alpha=(i+1)/len(nodes2))

    # Labels for clarity
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Set title
    ax.set_title('3D Plot of Node Points')

    # Ensure equal scaling of the axes by setting the limits
    max_range = np.array([nodes[:, 0].max() - nodes[:, 0].min(),
                        nodes[:, 1].max() - nodes[:, 1].min(),
                        nodes[:, 2].max() - nodes[:, 2].min()]).max()
    
    # Centering the plot (so that the axes are symmetric)
    mid_x = (nodes[:, 0].max() + nodes[:, 0].min()) / 2
    mid_y = (nodes[:, 1].max() + nodes[:, 1].min()) / 2
    mid_z = (nodes[:, 2].max() + nodes[:, 2].min()) / 2
    
    # Setting the limits of all axes based on the max range
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_bars(error, add_markers=False):
    index = np.arange(len(error[0,0]))+1
    bar_width = 0.1
    midbarsep = 0.03
    # colors_model1 = ['lightblue', 'lightgreen', 'lightcoral']
    rope_positions = [1,2,3,4]
    wire_names = ['white wire','black wire','red wire']
    wire_colors = ['grey','black','red']
    model_types = ['adapted model', 'native model']
    # model_edgecolors = ['lightgreen', 'lightpink']
    model_hatch = [None, '\\\\']

    bar_adapt = []
    bar_native = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, rope in enumerate(wire_colors):
        bar_adapt.append(ax.bar(index + (2*(i-1) * (bar_width+midbarsep)) - bar_width/2, error[0,i,], bar_width-0.01,
            color=wire_colors[i], capsize=5, alpha=0.7, hatch=model_hatch[0]))
        bar_native.append(ax.bar(index + (2*(i-1) * (bar_width+midbarsep)) + bar_width/2, error[1,i,], bar_width-0.01,
            color=wire_colors[i], capsize=5, alpha=0.7, hatch=model_hatch[1]))
        ax.bar(index + (2*i * bar_width), np.zeros_like(error[0,i,]), bar_width-0.04,
            label=f'{wire_names[i]}', color=wire_colors[i], capsize=5, alpha=0.7, edgecolor='none')
        # for j in range(len(rope_positions)):
        #     # bar_adapt[-1][j].set_color('lightgreen')
        #     # bar_adapt[-1][j].set_edgecolor(wire_colors[i])
        #     bar_adapt[-1][j].set_edgecolor(model_edgecolors[0])
        #     bar_adapt[-1][j].set_linewidth(5)
        #     # bar_native[-1][j].set_color('lightpink')
        #     # bar_native[-1][j].set_edgecolor(wire_colors[i])
        #     bar_native[-1][j].set_edgecolor(model_edgecolors[1])
        #     bar_native[-1][j].set_linewidth(5)

    # Make hatching visible on black bars by setting edgecolor to white (very thin)
    for bars, color in zip(bar_native, wire_colors):
        if color == 'black':
            for bar in bars:
                bar.set_edgecolor('white')
                bar.set_linewidth(0.7)  # Thinner outline for hatch visibility

    for i in range(len(model_types)):
        t1 = ax.bar(index + (2*i * bar_width), np.zeros_like(error[0,i,]), bar_width-0.04,
            label=f'{model_types[i]}', color='white', capsize=5, alpha=0.7, hatch=model_hatch[i], zorder = 10, edgecolor='none')
        t1[0].set_edgecolor('black')
        # t1[0].set_linewidth(5)

    ax.set_xlabel('Robot Pose', fontsize=14, labelpad=20)
    ax.tick_params(axis='x', pad=15)
    ax.set_ylabel('Normalized Position Error',fontsize=14)
    # ax.set_title('Comparison of Simulation Models and Real-world Data with Error Bars')
    ax.set_xticks(index)
    ax.set_xticklabels(rope_positions)
    # ax.legend(title='')
    # Remove the top and right borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.tick_params(axis='both', which='major', labelsize=14)
    # Position the legend at the center top
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=14)

    # Add a custom marker: two vertical lines at x=0.75 and x=1.5, connected by a horizontal line
    def draw_marker(ax, x0, x1, label=None, marker_lw=2, cmark = 'grey', marker_alpha=1.0, y_base=None):
        offset = 0.001  # how far below x-axis to draw
        if y_base is None:
            y_base = -offset  # below x-axis
        marker_height = 0.001  # height of the vertical lines
        # Draw left vertical line
        ax.plot([x0, x0], [y_base, y_base + marker_height], color=cmark, lw=marker_lw, zorder=20, alpha=marker_alpha)
        # Draw right vertical line
        ax.plot([x1, x1], [y_base, y_base + marker_height], color=cmark, lw=marker_lw, zorder=20, alpha=marker_alpha)
        # Draw horizontal line connecting the tops
        ax.plot([x0, x1], [y_base, y_base], color=cmark, lw=marker_lw, zorder=20, alpha=marker_alpha)
        # Optionally, add a label above the marker
        # if label is not None:
        #     ax.text((x0 + x1) / 2, y_base + marker_height * 1.2, label, ha='center', va='bottom', fontsize=13)

    disp_marker = 0.07
    base_loc0 = 1 + (-2 * (bar_width+midbarsep)) - bar_width/2
    base_loc1 = 1 + (2 * (bar_width+midbarsep)) + bar_width/2
    for i in range(4):
        draw_marker(ax, i+base_loc0-disp_marker, i+base_loc1+disp_marker, label='Group', cmark='lightgrey', marker_lw=5, marker_alpha=1.0)

    if add_markers:
        ybase_addmark = -0.002
        base_loc0 = 1 + (0 * (bar_width+midbarsep)) - bar_width/2
        base_loc1 = 1 + (0 * (bar_width+midbarsep)) + bar_width/2
        draw_marker(
            ax, 1+base_loc0-disp_marker, 1+base_loc1+disp_marker,
            y_base=ybase_addmark,
            cmark='#444444', label='Group', marker_lw=5, marker_alpha=1.0
        )

        base_loc0 = 1 + (2 * (bar_width+midbarsep)) - bar_width/2
        base_loc1 = 1 + (2 * (bar_width+midbarsep)) + bar_width/2
        draw_marker(
            ax, 3+base_loc0-disp_marker, 3+base_loc1+disp_marker,
            y_base=ybase_addmark,
            cmark='#ff9999', label='Group', marker_lw=5, marker_alpha=1.0
        )

    # After plotting bars and markers, expand y-limits to show markers below x-axis
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(-0.004,0.1)
    ax.set_yticks(np.arange(0, 0.12, 0.02))
    ax.spines['bottom'].set_position('zero')

    plt.tight_layout()
    plt.savefig(img_path + "valid_bars.pdf",bbox_inches='tight')
    plt.show()

def plot_bars_more(error, model_types, add_markers=False):
    index = np.arange(len(error[0,0]))+1
    bar_width = 0.075
    midbarsep = 0.018
    # colors_model1 = ['lightblue', 'lightgreen', 'lightcoral']
    rope_positions = [1,2,3,4]
    wire_names = ['white wire','black wire','red wire']
    wire_colors = ['grey','black','red']
    # model_edgecolors = ['lightgreen', 'lightpink']
    model_hatch = [None, 'xxx', '\\\\']

    bar_models = [[],[],[]]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, rope in enumerate(wire_colors):
        for j in range(len(model_types)):
            # print(index+ (3*(i-1) * (bar_width+midbarsep)))
            print(index + (3*(i-1) * (bar_width+midbarsep)) + (j-1)*bar_width)
            bar_models[j].append(ax.bar(index + (3*(i-1) * (bar_width+midbarsep)) + (j-1)*bar_width, error[j,i,], bar_width-0.01,
                color=wire_colors[i], capsize=5, alpha=0.7, hatch=model_hatch[j], edgecolor='none'))
        ax.bar(index + (2*i * bar_width), np.zeros_like(error[0,i,]), bar_width-0.04,
            label=f'{wire_names[i]}', color=wire_colors[i], capsize=5, alpha=0.7, edgecolor='none')
        # for j in range(len(rope_positions)):
        #     # bar_adapt[-1][j].set_color('lightgreen')
        #     # bar_adapt[-1][j].set_edgecolor(wire_colors[i])
        #     bar_adapt[-1][j].set_edgecolor(model_edgecolors[0])
        #     bar_adapt[-1][j].set_linewidth(5)
        #     # bar_native[-1][j].set_color('lightpink')
        #     # bar_native[-1][j].set_edgecolor(wire_colors[i])
        #     bar_native[-1][j].set_edgecolor(model_edgecolors[1])
        #     bar_native[-1][j].set_linewidth(5)

    # Make hatching visible on black bars by setting edgecolor to white
    for j in bar_models:
        for bars, color in zip(j, wire_colors):
            if color == 'black':
                for bar in bars:
                    bar.set_edgecolor('white')
                    bar.set_linewidth(1.5)  # Optional: make hatch lines thicker

    for i in range(len(model_types)):
        t1 = ax.bar(index + (2*i * bar_width), np.zeros_like(error[0,i,]), bar_width-0.04,
            label=f'{model_types[i]} model', color='white', capsize=5, alpha=0.7, hatch=model_hatch[i], zorder = 10, edgecolor='none')
        t1[0].set_edgecolor('black')
        # t1[0].set_linewidth(5)

    ax.set_xlabel('Robot Pose',fontsize=14, labelpad=20)
    ax.tick_params(axis='x', pad=15)
    ax.set_ylabel('Normalized Position Error',fontsize=14)
    # ax.set_title('Comparison of Simulation Models and Real-world Data with Error Bars')
    ax.set_xticks(index)
    ax.set_xticklabels(rope_positions)
    # ax.legend(title='')
    # Remove the top and right borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.tick_params(axis='both', which='major', labelsize=14)
    # Position the legend at the center top
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=14)

    # Add a custom marker: two vertical lines at x0 and x1, connected by a horizontal line
    def draw_marker(ax, x0, x1, label=None, marker_lw=2, cmark = '#444444', marker_alpha=1.0, y_base=None):
        offset = 0.001  # how far below x-axis to draw
        if y_base is None:
            y_base = -offset  # below x-axis
        marker_height = 0.001  # height of the vertical lines
        # Draw left vertical line
        ax.plot([x0, x0], [y_base, y_base + marker_height], color=cmark, lw=marker_lw, zorder=20, alpha=marker_alpha)
        # Draw right vertical line
        ax.plot([x1, x1], [y_base, y_base + marker_height], color=cmark, lw=marker_lw, zorder=20, alpha=marker_alpha)
        # Draw horizontal line connecting the tops
        ax.plot([x0, x1], [y_base, y_base], color=cmark, lw=marker_lw, zorder=20, alpha=marker_alpha)
        # Optionally, add a label above the marker
        # if label is not None:
        #     ax.text((x0 + x1) / 2, y_base + marker_height * 1.2, label, ha='center', va='bottom', fontsize=13)

    # Example marker usage (customize as needed):
    disp_marker = 0.07
    base_loc0 = 1 + (-3 * (bar_width+midbarsep)) - bar_width
    base_loc1 = 1 + (3 * (bar_width+midbarsep)) + bar_width
    for i in range(4):
        draw_marker(ax, i+base_loc0-disp_marker, i+base_loc1+disp_marker, label='Group', cmark='lightgrey', marker_lw=5, marker_alpha=1.0)

    if add_markers:
        # Example: add colored markers as in plot_bars
        ybase_addmark = -0.002
        base_loc0 = 1 + (0 * (bar_width+midbarsep)) - bar_width
        base_loc1 = 1 + (0 * (bar_width+midbarsep)) + bar_width
        draw_marker(
            ax, 1+base_loc0-disp_marker, 1+base_loc1+disp_marker,
            y_base=ybase_addmark,
            cmark='#444444', label='Group', marker_lw=5, marker_alpha=1.0
        )

        base_loc0 = 0 + (3 * (bar_width+midbarsep)) - bar_width/2
        base_loc1 = 0 + (3 * (bar_width+midbarsep)) + bar_width/2
        draw_marker(
            ax, 3+base_loc0-disp_marker, 3+base_loc1+disp_marker,
            y_base=ybase_addmark,
            cmark='#ff9999', label='Group', marker_lw=5, marker_alpha=1.0
        )

    # Set y-limits and y-ticks as in plot_bars
    ax.set_ylim(-0.004,0.1)
    ax.set_yticks(np.arange(0, 0.12, 0.02))
    ax.spines['bottom'].set_position('zero')

    plt.tight_layout()
    plt.savefig(img_path + "plgn/valid_bars_more.pdf",bbox_inches='tight')
    plt.show()

def plot_computetime(pieces_list, data_list):
    plt.style.use('seaborn-v0_8')
    # Sample data for 4 different results
    x = np.array(pieces_list)  # Number of discrete pieces simulated (from 1 to 100)
    y = np.array(data_list)

    # compute percentage difference
    y_percent = (y[:] - y[0]) / y[0] * 100.0

    # Create the plot
    fig, twin1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.75)
    ax = twin1.twinx()
    twin1.set_facecolor('white')
    twin1.grid(color='#DDDDDD', linewidth=0.8, zorder=0)
    ax.grid(False)
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    twin1.yaxis.tick_right()
    twin1.yaxis.set_label_position("right")

    ax.spines['top'].set_visible(False)
    twin1.spines['top'].set_visible(False)
    for spine_str in [
        'bottom',
        'left',
        'right'    
    ]:
        ax.spines[spine_str].set_linewidth(1)
        ax.spines[spine_str].set_edgecolor('k')

    # Plot each line with a label
    ax.plot(x, y[0], label='plain', alpha=0.7, color='k', linewidth=2,zorder=3)
    ax.plot(x, y[3], label='adapted', alpha=0.7, linewidth=2)
    ax.plot(x, y[2], label='direct', alpha=0.7, linewidth=2)
    ax.plot(x, y[1], label='native', alpha=0.7, linewidth=2)
    twin1.plot(0,0, label='raw time', alpha=1.0, color='k', linewidth=2)
    twin1.plot(0,0, label='percent increase', alpha=1.0, color='k', linewidth=2, linestyle='--')
    # twin1.plot(x, y_percent[0], alpha=0.7, color='k', linewidth=2,linestyle='--')
    twin1.plot(x, y_percent[3], alpha=0.7, linewidth=2,linestyle='--')
    twin1.plot(x, y_percent[2], alpha=0.7, linewidth=2,linestyle='--')
    twin1.plot(x, y_percent[1], alpha=0.7, linewidth=2,linestyle='--')

    ax.set_ylim(0, 50)
    ax.set_xlim(40, 180)
    twin1.set_ylim(-5, 20)
    ax.set_yticks(np.linspace(0., 50., 6))
    ax.set_xticks(x)
    twin1.set_yticks(np.linspace(-5, 20., 6))
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', **tkw)
    twin1.tick_params(axis='y', **tkw)

    # Add labels and title
    ax.set_ylabel('Computational Time to Simulate 1s (seconds)', fontsize=14)
    twin1.set_xlabel('Number of Discrete Pieces', fontsize=14)
    twin1.set_ylabel("Percentage Increase from plain", fontsize=14)
    # plt.title('Speed test', fontsize=14)

    # Add a legend
    ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.21, 0.98),ncol=1)
    twin1.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.39, 0.98),ncol=1)

    # Grid for better readability
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    twin1.tick_params(axis='both', which='major', labelsize=12)
    # # Set x and y axis limits for better visualization
    # plt.xlim([0, 100])
    # plt.ylim([0, 120])

    plt.tight_layout()
    # Save the plot if needed
    plt.savefig(img_path + "compute_time.pdf",bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_computetime_all(pieces_list, data_list, plot_labels=None):
    plt.style.use('seaborn-v0_8')
    # Sample data for 4 different results
    x = np.array(pieces_list)  # Number of discrete pieces simulated (from 1 to 100)
    y = np.array(data_list)

    # compute percentage difference
    y_percent = (y[:] - y[0]) / y[0] * 100.0

    # Create the plot
    fig, twin1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.75)
    ax = twin1.twinx()
    twin1.set_facecolor('white')
    twin1.grid(color='#DDDDDD', linewidth=0.8, zorder=0)
    ax.grid(False)
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    twin1.yaxis.tick_right()
    twin1.yaxis.set_label_position("right")

    ax.spines['top'].set_visible(False)
    twin1.spines['top'].set_visible(False)
    for spine_str in [
        'bottom',
        'left',
        'right'    
    ]:
        ax.spines[spine_str].set_linewidth(1)
        ax.spines[spine_str].set_edgecolor('k')

    if plot_labels is None:
        plot_labels = ['plain', 'native', 'direct', 'adapted']

    # Plot each line with a label
    ax.plot(x, y[0], label='plain', alpha=0.5, color='k', linewidth=2,zorder=3)
    twin1.plot(0,0, label='raw time', alpha=1.0, color='k', linewidth=2)
    twin1.plot(0,0, label='percent increase', alpha=1.0, color='k', linewidth=2, linestyle='--')
    for i in range(len(plot_labels)-1,0,-1):
        ax.plot(x, y[i], label=plot_labels[i], alpha=0.7, linewidth=2)
        twin1.plot(x, y_percent[i], alpha=0.7, linewidth=2,linestyle='--')

    ax.set_ylim(0, 50)
    ax.set_xlim(40, 180)
    twin1.set_ylim(-5, 20)
    ax.set_yticks(np.linspace(0., 50., 6))
    ax.set_xticks(x)
    twin1.set_yticks(np.linspace(-5, 20., 6))
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', **tkw)
    twin1.tick_params(axis='y', **tkw)

    # Add labels and title
    ax.set_ylabel('Computational Time to Simulate 1s (seconds)', fontsize=14)
    twin1.set_xlabel('Number of Discrete Pieces', fontsize=14)
    twin1.set_ylabel("Percentage Increase from plain", fontsize=14)
    # plt.title('Speed test', fontsize=14)

    # Add a legend
    ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.19, 0.98),ncol=1)
    twin1.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.43, 0.98),ncol=1)

    # Grid for better readability
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    twin1.tick_params(axis='both', which='major', labelsize=12)
    # # Set x and y axis limits for better visualization
    # plt.xlim([0, 100])
    # plt.ylim([0, 120])

    plt.tight_layout()
    # Save the plot if needed
    plt.savefig(img_path + "plgn/compute_time.pdf",bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    import time
    queue = mp.Queue()
    p = mp.Process(target=plotter_process, args=(queue, "dict_plot.png"))
    p.start()

    start = time.time()
    try:
        while True:
            t = time.time() - start
            val_dict = {
                "X": np.sin(t),
                "Y": np.cos(t),
                "Z": None,          # will be skipped in plot
                "X_dash": np.sin(t)+0.1,
                "Y_dash": None,     # skipped
                "Z_dash": 0.5*np.sin(t)+0.1
            }
            queue.put((t, val_dict))
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopping...")
        p.terminate()