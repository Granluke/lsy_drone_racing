import numpy as np
import csv


class PIDController:
    def __init__(self):
        self.Kp = np.array([0.3, 0.3, 0])
        self.Ki = np.array([0.0, 0.0, 0.0])
        self.Kd = np.array([0.01, 0.01, 0])
        self.dt = 0.01  # Time step
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.csv_filename = "target_vs_control.csv"

        with open(self.csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Target X",
                    "Target Y",
                    "Target Z",
                    "Control X",
                    "Control Y",
                    "Control Z",
                    "Actual position X",
                    "Actual position Y",
                    "Actual position Z",
                ]
            )

    def compute_control(self, target_pos, obs):
        # Extract the current position from obs
        current_pos = np.array([obs[0], obs[2], obs[4]])

        # Compute the error
        error = target_pos - current_pos

        # Proportional term
        P = self.Kp * error

        # Integral term
        self.integral += error * self.dt
        I = self.Ki * self.integral

        # Derivative term
        derivative = (error - self.previous_error) / self.dt
        D = self.Kd * derivative

        # Update previous error
        self.previous_error = error

        # Compute control output
        control_output = P + I + D

        new_target_pos = target_pos + control_output
        # check if z is < 0.1 and set to 0.1
        if new_target_pos[2] < 0.15:
            new_target_pos[2] = 0.15

        # Append target position and control output to CSV
        self.append_to_csv(target_pos, new_target_pos, current_pos)

        return new_target_pos

    def append_to_csv(self, target_pos, control_output, current_pos):
        with open(self.csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            row = list(target_pos) + list(control_output) + list(current_pos)
            writer.writerow(row)
