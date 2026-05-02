import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_trajectory(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    print(f"Loading trajectory from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Depending on the file format (groundtruth.csv has tx, ty, tz)
    if 'tx' in df.columns:
        x, y, z = df['tx'].values, df['ty'].values, df['tz'].values
    elif 'x' in df.columns:
        x, y, z = df['x'].values, df['y'].values, df['z'].values
    else:
        print("Error: CSV does not contain expected position columns (tx, ty, tz) or (x, y, z).")
        return

    timestamps = df['timestamp'].values

    print("Generating 3D trajectory plot with time-mapped trail...")
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Clean, minimalist formatting for beautiful 3D graphs
    ax.view_init(elev=30, azim=-60)
    ax.set_title(f"Camera Trajectory", fontsize=16, pad=15)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_zlabel("Height Z (m)", fontsize=12)
    
    # Hide the gray pane fills
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.5)

    # Plot the dense trajectory trail, colormapped to elapsed time
    scatter = ax.scatter(x, y, z, c=timestamps, cmap='viridis', s=5.0, depthshade=False, edgecolor='none')
    
    # Prominent start and end markers
    ax.plot([x[0]], [y[0]], [z[0]], marker='o', color='green', markersize=10, zorder=5, label='Start')
    ax.plot([x[-1]], [y[-1]], [z[-1]], marker='X', color='red', markersize=10, zorder=5, label='End')
    
    # Add a nice colorbar for time scaling
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1, aspect=15)
    cbar.set_label("Time (s)", rotation=270, labelpad=20, fontsize=12)
    
    ax.legend(loc='upper right', frameon=False, fontsize=12)
    plt.tight_layout()
    
    # Save the plot in the same directory as the CSV
    out_dir = os.path.dirname(csv_path)
    if out_dir == "":
        out_dir = "."
    
    plot_path = os.path.join(out_dir, "trajectory_plot_3d.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory plot to: {plot_path}")
    
    # Also show it interactively
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Trajectory from groundtruth.csv")
    parser.add_argument(
        '--csv', 
        type=str, 
        default='output/seq_001/groundtruth.csv', 
        help='Path to the groundtruth.csv or relative_poses.csv file'
    )
    args = parser.parse_args()
    
    visualize_trajectory(args.csv)
