import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Simulation settings
# -----------------------------
time_steps = 250
dt = 0.1

# -----------------------------
# System parameters
# -----------------------------
setpoint = 10.0
initial_water_level = 5.0
outflow_rate = 0.05
kp = 0.8

# -----------------------------
# MITM attack settings
# -----------------------------
attack_start = 80
attack_end = 160
attack_offset = -3.0


def run_simulation(attack_enabled=False):
    """
    Runs the water tank simulation.

    If attack_enabled is True, a MITM attack manipulates
    the sensor value sent to the controller.
    """

    water_level = initial_water_level

    time_data = []
    real_level_data = []
    measured_level_data = []
    pump_data = []
    error_data = []

    for t in range(time_steps):
        current_time = t * dt

        # Sensor reads the real water level
        measured_level = water_level

        # MITM attack manipulates the sensor value
        if attack_enabled and attack_start <= t <= attack_end:
            measured_level = measured_level + attack_offset

        # Controller calculates the error based on measured value
        controller_error = setpoint - measured_level

        # Simple proportional controller
        pump_inflow = kp * controller_error

        # Pump cannot have negative flow
        if pump_inflow < 0:
            pump_inflow = 0

        # Water tank dynamics
        water_level = water_level + (pump_inflow - outflow_rate * water_level) * dt

        # Store simulation data
        time_data.append(current_time)
        real_level_data.append(water_level)
        measured_level_data.append(measured_level)
        pump_data.append(pump_inflow)
        error_data.append(setpoint - water_level)

    return {
        "time": np.array(time_data),
        "real_level": np.array(real_level_data),
        "measured_level": np.array(measured_level_data),
        "pump": np.array(pump_data),
        "error": np.array(error_data)
    }


def calculate_metrics(results):
    """
    Calculates measurable performance metrics.

    These metrics are used to evaluate how the MITM attack
    affects the physical behavior of the simulated system.
    """

    real_level = results["real_level"]

    # Overshoot: how much the water level exceeds the setpoint
    overshoot = max(real_level) - setpoint
    if overshoot < 0:
        overshoot = 0

    # Maximum deviation over the whole simulation
    max_deviation = max(abs(real_level - setpoint))

    # Steady-state error at the end of the simulation
    steady_state_error = abs(setpoint - real_level[-1])

    # Deviation during the attack period only
    attack_period_level = real_level[attack_start:attack_end + 1]
    attack_period_deviation = abs(attack_period_level - setpoint)

    max_attack_deviation = max(attack_period_deviation)
    mean_attack_deviation = np.mean(attack_period_deviation)

    # Output boundedness check
    lower_bound = 0
    upper_bound = 15
    output_bounded = np.all((real_level >= lower_bound) & (real_level <= upper_bound))

    return {
        "Overshoot": overshoot,
        "Maximum deviation from setpoint": max_deviation,
        "Max deviation during attack": max_attack_deviation,
        "Mean deviation during attack": mean_attack_deviation,
        "Steady-state error": steady_state_error,
        "Output bounded": output_bounded
    }


# -----------------------------
# Run simulations
# -----------------------------
normal_results = run_simulation(attack_enabled=False)
attack_results = run_simulation(attack_enabled=True)

normal_metrics = calculate_metrics(normal_results)
attack_metrics = calculate_metrics(attack_results)


# -----------------------------
# Print comparison table
# -----------------------------
print("\nPerformance Metrics Comparison")
print("--------------------------------------------------------------------------")
print(f"{'Metric':40s} {'Normal':15s} {'MITM Attack':15s}")
print("--------------------------------------------------------------------------")

for metric in normal_metrics:
    normal_value = normal_metrics[metric]
    attack_value = attack_metrics[metric]

    if isinstance(normal_value, (bool, np.bool_)):
        normal_text = str(normal_value)
        attack_text = str(attack_value)
    else:
        normal_text = f"{normal_value:.2f}"
        attack_text = f"{attack_value:.2f}"

    print(f"{metric:40s} {normal_text:15s} {attack_text:15s}")

print("--------------------------------------------------------------------------")


# -----------------------------
# Plot 1: Normal vs MITM water level
# -----------------------------
plt.figure()
plt.plot(normal_results["time"], normal_results["real_level"], label="Normal operation")
plt.plot(attack_results["time"], attack_results["real_level"], label="MITM attack")
plt.axhline(y=setpoint, linestyle="--", label="Setpoint")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="Attack period")
plt.xlabel("Time (s)")
plt.ylabel("Water level")
plt.title("Comparison of Water Level: Normal Operation vs MITM Attack")
plt.legend()
plt.grid(True)
plt.savefig("figure_1_normal_vs_mitm_water_level.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# Plot 2: Sensor manipulation under MITM attack
# -----------------------------
plt.figure()
plt.plot(attack_results["time"], attack_results["real_level"], label="Real water level")
plt.plot(attack_results["time"], attack_results["measured_level"], label="Measured level received by controller")
plt.axhline(y=setpoint, linestyle="--", label="Setpoint")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="Attack period")
plt.xlabel("Time (s)")
plt.ylabel("Water level")
plt.title("Sensor Manipulation During MITM Attack")
plt.legend()
plt.grid(True)
plt.savefig("figure_2_sensor_manipulation.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# Plot 3: Pump signal comparison
# -----------------------------
plt.figure()
plt.plot(normal_results["time"], normal_results["pump"], label="Normal operation")
plt.plot(attack_results["time"], attack_results["pump"], label="MITM attack")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="Attack period")
plt.xlabel("Time (s)")
plt.ylabel("Pump inflow")
plt.title("Controller Output: Normal Operation vs MITM Attack")
plt.legend()
plt.grid(True)
plt.savefig("figure_3_pump_signal_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# Plot 4: Error comparison
# -----------------------------
plt.figure()
plt.plot(normal_results["time"], normal_results["error"], label="Normal operation")
plt.plot(attack_results["time"], attack_results["error"], label="MITM attack")
plt.axhline(y=0, linestyle="--")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="Attack period")
plt.xlabel("Time (s)")
plt.ylabel("Deviation from setpoint")
plt.title("Physical Impact: Deviation from Desired System State")
plt.legend()
plt.grid(True)
plt.savefig("figure_4_error_comparison.png", dpi=300, bbox_inches="tight")
plt.show()