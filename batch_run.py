import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import csv

# Number of times to run the simulation
num_runs = 50

# Lists to store the results
completion_times = []
termination_reasons = []

# Function to run the simulation and capture the output
def run_simulation():
    command = ["python", "scripts/sim.py", "--config", "config/level3.yaml", "--controller", "examples/own_path_controller.py"]
    result = subprocess.run(command,  capture_output=True, text=False)
    return result.stdout.decode()

# Function to parse the output
def parse_output(output):
    time_match = re.findall(r"\d+\.\d+", output)
    if time_match:
        time = float(time_match[-1])
        return time
    return None

# Run the simulation multiple times
for _ in range(num_runs):
    output = run_simulation()
    time = parse_output(output)
    reason = "TASK COMPLETION"
    if reason and time:
        termination_reasons.append(reason)
        completion_times.append(time)

#save termination_reasons & completion_times to csv
with open('simulation_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Run', 'Reason for termination', 'Completion time'])
    for i, (reason, time) in enumerate(zip(termination_reasons, completion_times), start=1):
        writer.writerow([i, reason, time])


# Convert completion times to a numpy array for statistical analysis
completion_times = np.array(completion_times)

# Calculate statistics
mean_time = np.mean(completion_times)
std_dev_time = np.std(completion_times)

print(f"Mean completion time: {mean_time}")
print(f"Standard deviation of completion time: {std_dev_time}")

# Plot the completion times
plt.figure(figsize=(10, 5))
plt.plot(completion_times, marker='o', linestyle='-', color='b')
plt.xlabel('Run number')
plt.ylabel('Completion time')
plt.title('Completion times for 50 runs')
plt.grid(True)
plt.show()
