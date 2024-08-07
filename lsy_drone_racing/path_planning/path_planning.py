import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev


OBSTACLE_DIMENSIONS = {"shape": "cylinder", "height": 1.05, "radius": 0.05}


class PathPlanning:

    def __init__(self, gates, obstacles, start_point, t, plot=True) -> None:
        self.gates = gates
        self.obstacles = obstacles
        self.start_point = start_point
        self.t = t
        self.plot = plot
        self.before_after_points = []
        self.go_around_points = []
        self.intersection_points = []
        self.waypoint_gate_indices = [0]
        self.buffer = 0.2
        self.avoidance_distance = 0.4
        self.waypoint_obstacle_buffer = 0.27
        self.path_collision_buffer = 0.25
        self.adjust_path_buffer = 0.45
        self.gate_frame_buffer = 0.1
        self.edge_length = 0.45

    def create_gate(self, x, y, yaw, gate_type):
        if gate_type == 0:
            height = 1.0
        else:
            height = 0.525

        z_bottom = height - self.edge_length / 2
        z_top = height + self.edge_length / 2

        # Define the square in the XZ plane
        points = np.array(
            [
                [-self.edge_length / 2, 0, z_bottom],
                [self.edge_length / 2, 0, z_bottom],
                [self.edge_length / 2, 0, z_top],
                [-self.edge_length / 2, 0, z_top],
            ]
        )

        # Apply yaw rotation
        yaw_matrix = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        points[:, :2] = np.dot(points[:, :2], yaw_matrix[:2, :2].T)

        # Adjust position to account for x, y
        points[:, 0] += x
        points[:, 1] += y

        return points

    def create_cylinder(self, x, y, z, height, radius):
        z_values = np.linspace(0, height, 30)
        theta = np.linspace(0, 2 * np.pi, 30)
        theta_grid, z_grid = np.meshgrid(theta, z_values)
        x_grid = radius * np.cos(theta_grid) + x
        y_grid = radius * np.sin(theta_grid) + y
        z_grid += z
        return x_grid, y_grid, z_grid

    def create_waypoints(self):
        self.waypoints = [self.start_point]

        for i, gate in enumerate(self.gates):
            self.generate_gate_waypoints(i, gate)
        self.waypoints = np.array(self.waypoints)

    def generate_gate_waypoints(self, i, gate):
        x, y, _, _, _, yaw, gate_type = gate
        if gate_type == 0:
            z = 1.0
        else:
            z = 0.525

        tmp_wp_1 = [x - self.buffer * np.sin(yaw), y + self.buffer * np.cos(yaw), z]
        tmp_wp_2 = [x + self.buffer * np.sin(yaw), y - self.buffer * np.cos(yaw), z]

        prev_wp = self.waypoints[-1]
        dist_tmp_wp_1 = np.linalg.norm([prev_wp[0] - tmp_wp_1[0], prev_wp[1] - tmp_wp_1[1]])
        dist_tmp_wp_2 = np.linalg.norm([prev_wp[0] - tmp_wp_2[0], prev_wp[1] - tmp_wp_2[1]])

        factor_for_2nd_before_point = 1
        if dist_tmp_wp_2 < dist_tmp_wp_1:
            after_wp = tmp_wp_1
            # double the buffer for the before_wp
            # before_wp = [x + 2 * buffer * np.sin(yaw), y - 2 * buffer * np.cos(yaw), z]
            factor_for_2nd_before_point = 0.8
            before_wp = tmp_wp_2
        else:
            # double the buffer for the before_wp
            # before_wp = [x - 2 * buffer * np.sin(yaw), y + 2 * buffer * np.cos(yaw), z]
            factor_for_2nd_before_point = -0.8
            before_wp = tmp_wp_1
            after_wp = tmp_wp_2

        # check if last before_wp has distance of more then 2 to the last waypoint then put a waypoint in beweteen at half distance
        if len(self.waypoints) >= 1:
            last_wp = self.waypoints[-1]
            dist_last_wp_before_wp = np.linalg.norm(
                [last_wp[0] - before_wp[0], last_wp[1] - before_wp[1]]
            )
            if dist_last_wp_before_wp > 1.4:

                self.waypoints.append(
                    [
                        x + factor_for_2nd_before_point * np.sin(yaw),
                        y - factor_for_2nd_before_point * np.cos(yaw),
                        (last_wp[2] + before_wp[2]) / 2,
                    ]
                )
                self.waypoint_gate_indices.append(i)

        self.waypoints.append(before_wp)
        self.waypoints.append([x, y, z])
        self.waypoints.append(after_wp)

        # waypoint_gate_indices append i 3 times
        self.waypoint_gate_indices.append(i)
        self.waypoint_gate_indices.append(i)
        self.waypoint_gate_indices.append(i)

        self.before_after_points.append((before_wp, after_wp))

        # Check if the line to the next gate goes through the current gate
        if i < len(self.gates) - 1:
            self.check_and_avoid_gate_intersect(i, gate, z, after_wp, factor_for_2nd_before_point)
        else:
            self.go_around_points.append(([]))

    def check_and_avoid_gate_intersect(
        self, i, gate, z_center, after_wp, factor_for_2nd_before_point
    ):
        next_gate = self.gates[i + 1]
        x, y, _, _, _, yaw, gate_type = gate
        next_x, next_y, _, _, _, _, next_gate_type = next_gate
        if next_gate_type == 0:
            next_z_center = 1.0
        else:
            next_z_center = 0.525
        line_start = np.array([after_wp[0], after_wp[1], after_wp[2]])
        line_end = np.array([next_x, next_y, next_z_center])
        plane_point = np.array([x, y, z_center])
        gate_points = self.create_gate(x, x, yaw, gate_type)
        plane_normal = np.cross(gate_points[1] - gate_points[0], gate_points[2] - gate_points[0])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        intersects, intersection_point = self.line_intersects_plane(
            line_start, line_end, plane_point, plane_normal
        )

        if intersects:
            self.intersection_points.append(intersection_point)

        if intersects and self.point_in_gate(intersection_point, np.array([x, y, z_center]), yaw):
            print(
                f"Gate {i} intersects with line to gate {i + 1} at {intersection_point} for next gate {next_x, next_y, next_z_center}"
            )
            # Calculate direction to next gate
            # Calculate direction from gate center to intersection point
            intersection_vector = intersection_point - np.array([x, y, z_center])
            intersection_vector /= np.linalg.norm(intersection_vector)

            # 1. Variant to Calculate the avoidance point (more dynamic)
            # avoidance_wp = plane_point + intersection_vector * (0.45/sqrt(2) + self.avoidance_distance)

            # 2nd variant to calculate avoidance points (more safe)
            # make another avoidance waypoint parallel to the gate with the buffer and z = avaoidance_wp[2]
            factor_for_2nd_avoidancy_point = (
                -1 / factor_for_2nd_before_point * factor_for_2nd_before_point
            )
            self.waypoints[-1] = [
                x + factor_for_2nd_avoidancy_point * (self.buffer + 0.1) * np.sin(yaw),
                y - factor_for_2nd_avoidancy_point * (self.buffer + 0.1) * np.cos(yaw),
                z_center - 0.05,
            ]
            avoidance_wp2 = [
                x + factor_for_2nd_avoidancy_point * (self.buffer + 0.1) * np.sin(yaw),
                y - factor_for_2nd_avoidancy_point * (self.buffer + 0.1) * np.cos(yaw),
                z_center + 0.3,
            ]
            test_avoidance_wp = [x, y, z_center + self.edge_length]

            self.waypoints.append(avoidance_wp2)
            # waypoints.append(avoidance_wp)
            self.waypoints.append(test_avoidance_wp)
            self.waypoint_gate_indices.append(i)
            self.waypoint_gate_indices.append(i)
            self.go_around_points.append((avoidance_wp2, test_avoidance_wp))

            # self.waypoints.append(avoidance_wp)
            # self.waypoint_gate_indices.append(i)
            # self.go_around_points.append((avoidance_wp))
        else:
            self.go_around_points.append(([]))

    def point_in_gate(self, point, gate_center, yaw):
        gate_width = self.edge_length + self.buffer + self.gate_frame_buffer
        gate_height = self.edge_length + self.buffer + self.gate_frame_buffer
        rel_point = point - gate_center

        yaw_matrix = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw), np.cos(-yaw), 0],
                [0, 0, 1],
            ]
        )

        local_point = np.dot(yaw_matrix, rel_point)

        return -gate_width / 2 <= local_point[0] <= gate_width / 2 and (
            gate_center[2] - gate_height / 2
        ) <= point[2] <= (gate_center[2] + gate_height / 2)

    def line_intersects_plane(self, p1, p2, plane_point, plane_normal):
        line_vec = p2 - p1
        plane_point_vec = plane_point - p1
        dot_product = np.dot(plane_normal, line_vec)

        if np.abs(dot_product) < 1e-6:
            return False, None

        t = np.dot(plane_normal, plane_point_vec) / dot_product

        if 0 <= t <= 1:
            intersection_point = p1 + t * line_vec
            return True, intersection_point

        return False, None

    def check_path_collision(self, path):
        height = OBSTACLE_DIMENSIONS["height"]
        radius = OBSTACLE_DIMENSIONS["radius"]
        radius = radius + self.path_collision_buffer
        for cylinder in self.obstacles:
            x_c, y_c, z_c, _, _, _ = cylinder
            for p in path.T:
                x, y, z = p
                if (x - x_c) ** 2 + (y - y_c) ** 2 <= radius**2 and 0 <= z <= height:
                    # print(f"Collision at {x}, {y}, {z}")
                    return True
        return False

    def adjust_waypoints(self) -> np.ndarray:
        adjusted_waypoints = []
        height = OBSTACLE_DIMENSIONS["height"]
        radius = OBSTACLE_DIMENSIONS["radius"]
        radius = radius + self.waypoint_obstacle_buffer
        for i, waypoint in enumerate(self.waypoints):
            x, y, z = waypoint
            collision = False
            for cylinder in self.obstacles:
                x_c, y_c, z_c, _, _, _ = cylinder
                if (x - x_c) ** 2 + (y - y_c) ** 2 <= radius**2 and 0 <= z <= height:
                    collision = True
                    # print(f"Collision at {x}, {y}, {z} for waypoint {waypoint} at gate {self.waypoint_gate_indices[i]} with index {i}")
                    # Adjust the waypoint by moving it away from the cylinder so that buffer is incorporated but parallel to the gate
                    gate = self.gates[self.waypoint_gate_indices[i]]
                    _, _, _, _, _, yaw, _ = gate
                    adjusted_waypoint = self.find_intersection_with_buffer(waypoint, x_c, y_c, yaw)
                    x, y, z = adjusted_waypoint.round(4)
                    # print(f"Adjusted to {x}, {y}, {z}")
            adjusted_waypoints.append([x, y, z])
        return np.array(adjusted_waypoints)

    def find_intersection_with_buffer(self, waypoint, x_c, y_c, yaw):
        # Line parameters
        x1, y1, z = waypoint
        m = np.tan(yaw)
        b = y1 - m * x1

        # Circle parameters
        r = self.waypoint_obstacle_buffer

        # Solve for intersection points
        A = 1 + m**2
        B = 2 * (m * b - m * y_c - x_c)
        C = x_c**2 + y_c**2 + b**2 - 2 * b * y_c - r**2

        discriminant = B**2 - 4 * A * C

        if discriminant < 0:
            # No intersection
            return waypoint

        # Calculate the two intersection points
        sqrt_discriminant = np.sqrt(discriminant)
        x_inter1 = (-B + sqrt_discriminant) / (2 * A)
        x_inter2 = (-B - sqrt_discriminant) / (2 * A)

        y_inter1 = m * x_inter1 + b
        y_inter2 = m * x_inter2 + b

        inter_point1 = np.array([x_inter1, y_inter1, z])
        inter_point2 = np.array([x_inter2, y_inter2, z])

        # Choose the intersection point that is on the buffer
        distance1 = np.linalg.norm(inter_point1 - waypoint)
        distance2 = np.linalg.norm(inter_point2 - waypoint)

        if distance1 < distance2:
            return inter_point1
        else:
            return inter_point2

    def adjust_path(self, path) -> np.ndarray:
        extra_buffer = 0.3
        adjusted_path = [[], [], []]
        height = OBSTACLE_DIMENSIONS["height"]
        radius = OBSTACLE_DIMENSIONS["radius"]
        radius = radius + self.adjust_path_buffer
        r_squared = radius**2
        for x, y, z in zip(*path):
            collision = False
            if z < 0.1:
                z = 0.1
            for cylinder in self.obstacles:
                x_c, y_c, z_c, _, _, _ = cylinder
                distance_calc = (x - x_c) ** 2 + (y - y_c) ** 2
                if distance_calc <= r_squared and 0 <= z <= height:
                    collision = True
                    # Adjust the point by moving it away from the cylinder
                    angle = np.arctan2(y - y_c, x - x_c)
                    distance = np.sqrt(distance_calc)
                    # Apply a quadratic adjustment factor
                    adjustment_factor = np.clip(((radius - distance) / radius) ** 2, 0, 1)
                    x += np.cos(angle) * extra_buffer * adjustment_factor
                    y += np.sin(angle) * extra_buffer * adjustment_factor
            adjusted_path[0].append(x)
            adjusted_path[1].append(y)
            adjusted_path[2].append(z)

        return np.array(adjusted_path)

    def calc_best_path(self):

        self.create_waypoints()
        self.waypoints = self.adjust_waypoints()
        tck, u = splprep(self.waypoints.T, s=0)

        path1 = splev(self.t, tck)
        path = path1
        if self.check_path_collision(np.array(path)):
            print("Path collides with obstacles, adjusting waypoints...")
            path = self.adjust_path(path)
            # recompute splprep and path with splev
            tck, u = splprep(path, s=0)
            path3 = splev(self.t, tck)
            path = path3

        if self.plot:
            self.plot_gates_and_obstacles(path1, path)

        return path, self.waypoints

    def plot_gates_and_obstacles(
        self, path_before_obstacle_avoidance, path_after_obstacle_avoidance
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        all_points = []

        for i, gate in enumerate(self.gates):
            x, y, z_center, _, _, yaw, gate_type = gate
            points = self.create_gate(x, y, yaw, gate_type)
            all_points.append(points)
            verts = [list(zip(points[:, 0], points[:, 1], points[:, 2]))]
            poly = Poly3DCollection(verts, alpha=0.5, linewidths=1, edgecolors="r")
            poly.set_facecolor([0.5, 0.5, 1])
            ax.add_collection3d(poly)
            ax.text(x, y, z_center, str(i), color="black")

            # Plot before and after waypoints
            before_wp, after_wp = self.before_after_points[i]
            ax.scatter(*before_wp, color="blue", s=50)
            ax.scatter(*after_wp, color="red", s=50)

            go_around_wp1 = self.go_around_points[i]
            if len(go_around_wp1) > 0:
                ax.scatter(*go_around_wp1, color="green", s=50, label="Go Around Points")
            # ax.scatter(*go_around_wp2, color='green', s=50)

        for intersection_point in self.intersection_points:
            if intersection_point is not None:
                p = intersection_point.tolist()
                ax.scatter(
                    [round(p[0], 2)],
                    [round(p[1], 2)],
                    [round(p[2], 2)],
                    color="orange",
                    s=50,
                )

        # ax.scatter(*[-0.12161042,0.56976572,0.525], color='orange', s=60)

        for cylinder in self.obstacles:
            x, y, z, roll, pitch, yaw = cylinder
            height = OBSTACLE_DIMENSIONS["height"]
            radius = OBSTACLE_DIMENSIONS["radius"]
            x_grid, y_grid, z_grid = self.create_cylinder(x, y, z, height, radius)
            ax.plot_surface(x_grid, y_grid, z_grid, color="b", alpha=0.5)
            all_points.append(
                np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))
            )

        # Convert all_points to a single numpy array for easy limit calculation
        all_points = np.concatenate(all_points, axis=0)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set limits
        ax.set_xlim([all_points[:, 0].min() - 1, all_points[:, 0].max() + 1])
        ax.set_ylim([all_points[:, 1].min() - 1, all_points[:, 1].max() + 1])
        ax.set_zlim([all_points[:, 2].min() - 1, all_points[:, 2].max() + 1])

        # Plot the path
        ax.plot(
            path_before_obstacle_avoidance[0],
            path_before_obstacle_avoidance[1],
            path_before_obstacle_avoidance[2],
            "g",
            label="Original Path",
        )
        ax.plot(
            path_after_obstacle_avoidance[0],
            path_after_obstacle_avoidance[1],
            path_after_obstacle_avoidance[2],
            "r",
            label="Adjusted Path",
        )

        # Plot waypoints
        for wp in self.waypoints:
            ax.scatter(*wp, color="black", s=50)

        # where some data has already been plotted to ax
        handles, _ = ax.get_legend_handles_labels()

        # manually define a new patch
        patch = Line2D(
            [0],
            [0],
            label="Waypoints",
            color="black",
            marker="o",
            markersize=6,
            linestyle="",
        )
        patch2 = Line2D(
            [0],
            [0],
            label="Original Before Gate Waypoints",
            color="blue",
            marker="o",
            markersize=6,
            linestyle="",
        )
        patch3 = Line2D(
            [0],
            [0],
            label="Original After Gate Waypoints",
            color="red",
            marker="o",
            markersize=6,
            linestyle="",
        )
        patch4 = Line2D(
            [0],
            [0],
            label="Gate Intersections",
            color="orange",
            marker="o",
            markersize=6,
            linestyle="",
        )
        # handles is a list, so append manual patch
        handles.append(patch)
        handles.append(patch2)
        handles.append(patch3)
        handles.append(patch4)

        plt.legend(handles=handles)
        plt.show()


LEVEL_0_GATES = [
    # Example gate, z_center adjusted for low obstacle
    [0.45, -1.0, 0, 0, 0, 2.35, 1],
    [1.0, -1.55, 0, 0, 0, -0.78, 0],
    [0.0, 0.5, 0, 0, 0, 0, 1],
    [-0.5, -0.5, 0, 0, 0, 3.14, 0],
]

LEVEL_2_GATES = [
    [0.5643489615604683, -1.1340759352922771, 0.525, 0.0, 0.0, 2.2416243338411697, 1],
    [0.9356331072064012, -1.551774864669796, 1.0, 0.0, 0.0, -0.6778159130838787, 0],
    [
        -0.11679737660112775,
        0.36982364329250744,
        0.525,
        0.0,
        0.0,
        0.02406755585949466,
        1,
    ],
    [-0.5, -0.5, 0, 0, 0, 3.14, 0],
]

OBSTACLES = [
    [1.0, -0.5, 0, 0, 0, 0],
    [0.5, -1.5, 0, 0, 0, 0],
    [-0.5, 0, 0, 0, 0, 0],
    [0, 1.0, 0, 0, 0, 0],
]
duration = 10
CTRL_FREQ = 30
T = np.linspace(0, 1, int(duration * CTRL_FREQ))

START_POINT = [0.9339007658017658, 0.9934538571454128, 0.05]

if __name__ == "__main__":
    path_planner = PathPlanning(LEVEL_0_GATES, OBSTACLES, START_POINT, T, plot=True)
    path_planner.calc_best_path()
