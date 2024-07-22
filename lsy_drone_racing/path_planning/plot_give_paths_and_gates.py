import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import ast
import json
from lsy_drone_racing.path_planning.calc_path_through_gates import (
    create_gate,
    create_cylinder,
    OBSTACLE_DIMENSIONS,
)


def plot_gates_and_obstacles(csv_file):
    df = pd.read_csv(csv_file)

    obstacles = []
    paths = []
    gates_list = []
    path_labels = []
    gate_labels = []
    waypoints = []

    # parse waypoints
    if "Last Waypoints" in df.columns:
        waypoints = ast.literal_eval(df["Last Waypoints"][0])

    # Parse obstacles
    if "Obstacles" in df.columns:
        obstacles = ast.literal_eval(df["Obstacles"][0])

    # Parse paths and gates
    path_columns = [col for col in df.columns if col.startswith("Path")]
    gate_columns = [col for col in df.columns if col.startswith("Gates")]

    # Extract paths
    for i in range(len(path_columns) // 3):
        path_x = ast.literal_eval(df[f"Path_{i*2 + 1}_x"].to_list()[0])
        path_y = ast.literal_eval(df[f"Path_{i*2 + 1}_y"].to_list()[0])
        path_z = ast.literal_eval(df[f"Path_{i*2 + 1}_z"].to_list()[0])
        paths.append((path_x, path_y, path_z))
        path_labels.append(f"Path {i}")

    # Extract gates
    for i, gate_col in enumerate(gate_columns):
        gates_list.append(ast.literal_eval(df[gate_col][0]))
        gate_labels.append(f"Gates for Path {i}")

    print(gates_list)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    all_points = []

    # Define a color map for different gate sets
    gate_colors = cm.rainbow(np.linspace(0, 1, len(gates_list)))

    for gate_set, color, label in zip(gates_list, gate_colors, gate_labels):
        for i, gate in enumerate(gate_set):
            x, y, z_center, _, _, yaw, gate_type = gate
            points = create_gate(x=x, y=y, yaw=yaw, gate_type=gate_type)
            all_points.append(points)
            verts = [list(zip(points[:, 0], points[:, 1], points[:, 2]))]
            poly = Poly3DCollection(verts, alpha=0.5, linewidths=1, edgecolors=color)
            poly.set_facecolor(color)
            ax.add_collection3d(poly)
            ax.text(x, y, z_center, str(i), color="black")

        # Add a legend entry for the gate set
        ax.plot([], [], [], color=color, label=label)

    for cylinder in obstacles:
        x, y, z, roll, pitch, yaw = cylinder
        x_grid, y_grid, z_grid = create_cylinder(
            x, y, z, OBSTACLE_DIMENSIONS["height"], OBSTACLE_DIMENSIONS["radius"]
        )
        ax.plot_surface(x_grid, y_grid, z_grid, color="b", alpha=0.5)
        all_points.append(np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten())))

    for path, color, label in zip(paths, gate_colors, path_labels):
        ax.plot(path[0], path[1], path[2], color=color, label=label)

    data_actual = pd.read_csv("/home/michael/autoDroneRepos/lsy_drone_racing/target_vs_control.csv")

    ax.plot(
        data_actual["Actual position X"],
        data_actual["Actual position Y"],
        data_actual["Actual position Z"],
        "-",
        label="Actual Position",
        color="black",
    )

    # Convert all_points to a single numpy array for easy limit calculation
    if all_points:
        all_points = np.concatenate(all_points, axis=0)

        ax.set_xlim([all_points[:, 0].min() - 1, all_points[:, 0].max() + 1])
        ax.set_ylim([all_points[:, 1].min() - 1, all_points[:, 1].max() + 1])
        ax.set_zlim([all_points[:, 2].min() - 1, all_points[:, 2].max() + 1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.legend()
    for waypoint in waypoints:
        ax.plot(waypoint[0], waypoint[1], waypoint[2], "o", label="Waypoints", color="red")
    plt.show()


if __name__ == "__main__":
    plot_gates_and_obstacles("/home/michael/autoDroneRepos/lsy_drone_racing/paths_gates.csv")
