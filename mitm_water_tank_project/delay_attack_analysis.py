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
# Delay attack settings
# -----------------------------
attack_start = 30
attack_end = 130
delay_steps = 40   # 12 steps * 0.1 s = 1.2 seconds delay


def run_simulation(delay_attack_enabled=False):
    """
    Runs the water tank simulation.
    If delay_attack_enabled is True, the controller receives delayed sensor values.
    """

    water_level = initial_water_level

    time_data = []
    real_level_data = []
    measured_level_data = []
    pump_data = []
    error_data = []

    # Buffer for delayed sensor values
    sensor_buffer = [initial_water_level] * delay_steps

    for t in range(time_steps):
        current_time = t * dt

        # Current sensor reading
        current_sensor_value = water_level

        # Delay attack: send an older sensor value instead of the current one
        if delay_attack_enabled and attack_start <= t <= attack_end:
            measured_level = sensor_buffer.pop(0)
            sensor_buffer.append(current_sensor_value)
        else:
            measured_level = current_sensor_value

            # Keep buffer updated even outside attack period
            sensor_buffer.pop(0)
            sensor_buffer.append(current_sensor_value)

        # Controller calculates error using measured value
        controller_error = setpoint - measured_level

        # Proportional controller
        pump_inflow = kp * controller_error

        # Pump cannot have negative flow
        if pump_inflow < 0:
            pump_inflow = 0

        # Tank dynamics
        water_level = water_level + (pump_inflow - outflow_rate * water_level) * dt

        # Store data
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
    """

    real_level = results["real_level"]

    # Overshoot
    overshoot = max(real_level) - setpoint
    if overshoot < 0:
        overshoot = 0

    # Maximum deviation over the whole simulation
    max_deviation = max(abs(real_level - setpoint))

    # Steady-state error at the end
    steady_state_error = abs(setpoint - real_level[-1])

    # Attack-period deviation only
    attack_period_level = real_level[attack_start:attack_end + 1]
    attack_period_deviation = abs(attack_period_level - setpoint)

    max_attack_deviation = max(attack_period_deviation)
    mean_attack_deviation = np.mean(attack_period_deviation)

    # Output boundedness
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
normal_results = run_simulation(delay_attack_enabled=False)
delay_results = run_simulation(delay_attack_enabled=True)

normal_metrics = calculate_metrics(normal_results)
delay_metrics = calculate_metrics(delay_results)


# -----------------------------
# Print comparison table
# -----------------------------
print("\nPerformance Metrics Comparison: Normal vs Delay Attack")
print("--------------------------------------------------------------------------")
print(f"{'Metric':40s} {'Normal':15s} {'Delay Attack':15s}")
print("--------------------------------------------------------------------------")

for metric in normal_metrics:
    normal_value = normal_metrics[metric]
    delay_value = delay_metrics[metric]

    if isinstance(normal_value, (bool, np.bool_)):
        normal_text = str(normal_value)
        delay_text = str(delay_value)
    else:
        normal_text = f"{normal_value:.2f}"
        delay_text = f"{delay_value:.2f}"

    print(f"{metric:40s} {normal_text:15s} {delay_text:15s}")

print("--------------------------------------------------------------------------")


# -----------------------------
# Plot 1: Normal vs Delay water level
# -----------------------------
plt.figure()
plt.plot(normal_results["time"], normal_results["real_level"], label="Normal operation")
plt.plot(delay_results["time"], delay_results["real_level"], label="Delay attack")
plt.axhline(y=setpoint, linestyle="--", label="Setpoint")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="Attack period")
plt.xlabel("Time (s)")
plt.ylabel("Water level")
plt.title("Comparison of Water Level: Normal Operation vs Delay Attack")
plt.legend()
plt.grid(True)
plt.savefig("delay_figure_1_normal_vs_delay_water_level.png", dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# Plot 2: Measured value under Delay attack
# -----------------------------
plt.figure()
plt.plot(delay_results["time"], delay_results["real_level"], label="Real water level")
plt.plot(delay_results["time"], delay_results["measured_level"], label="Delayed measured level")
plt.axhline(y=setpoint, linestyle="--", label="Setpoint")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="Attack period")
plt.xlabel("Time (s)")
plt.ylabel("Water level")
plt.title("Delayed Sensor Values During Attack")
plt.legend()
plt.grid(True)
plt.savefig("delay_figure_2_delayed_sensor_values.png", dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# Plot 3: Pump signal comparison
# -----------------------------
plt.figure()
plt.plot(normal_results["time"], normal_results["pump"], label="Normal operation")
plt.plot(delay_results["time"], delay_results["pump"], label="Delay attack")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="Attack period")
plt.xlabel("Time (s)")
plt.ylabel("Pump inflow")
plt.title("Controller Output: Normal Operation vs Delay Attack")
plt.legend()
plt.grid(True)
plt.savefig("delay_figure_3_pump_signal_comparison.png", dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# Plot 4: Error comparison
# -----------------------------
plt.figure()
plt.plot(normal_results["time"], normal_results["error"], label="Normal operation")
plt.plot(delay_results["time"], delay_results["error"], label="Delay attack")
plt.axhline(y=0, linestyle="--")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="Attack period")
plt.xlabel("Time (s)")
plt.ylabel("Deviation from setpoint")
plt.title("Physical Impact: Deviation from Desired State Under Delay Attack")
plt.legend()
plt.grid(True)
plt.savefig("delay_figure_4_error_comparison.png", dpi=300, bbox_inches="tight")
plt.close()


print("\nPlots saved successfully.")