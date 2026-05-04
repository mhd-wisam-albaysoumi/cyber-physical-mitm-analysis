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
# Attack settings
# -----------------------------
attack_start = 80
attack_end = 160

# Sensor manipulation attack
attack_offset = -3.0

# Delay attack
delay_steps = 40

# False injection attack
false_sensor_value = 6.0


def run_simulation(attack_type="normal"):
    """
    Runs the water tank simulation.

    attack_type can be:
    - "normal"
    - "sensor_manipulation"
    - "delay"
    - "false_injection"
    """

    water_level = initial_water_level

    time_data = []
    real_level_data = []
    measured_level_data = []
    pump_data = []
    error_data = []

    # Buffer used for delay attack
    sensor_buffer = [initial_water_level] * delay_steps

    for t in range(time_steps):
        current_time = t * dt

        # Real sensor reading
        real_sensor_value = water_level
        measured_level = real_sensor_value

        attack_active = attack_start <= t <= attack_end

        # -----------------------------
        # Attack logic
        # -----------------------------
        if attack_type == "sensor_manipulation" and attack_active:
            measured_level = real_sensor_value + attack_offset

        elif attack_type == "delay" and attack_active:
            measured_level = sensor_buffer.pop(0)
            sensor_buffer.append(real_sensor_value)

        elif attack_type == "false_injection" and attack_active:
            measured_level = false_sensor_value

        else:
            measured_level = real_sensor_value

            # Keep delay buffer updated even outside delay attack
            sensor_buffer.pop(0)
            sensor_buffer.append(real_sensor_value)

        # Controller calculates error using measured value
        controller_error = setpoint - measured_level

        # Proportional controller
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
    Calculates performance metrics for physical impact analysis.
    """

    real_level = results["real_level"]

    # Overshoot
    overshoot = max(real_level) - setpoint
    if overshoot < 0:
        overshoot = 0

    # Maximum deviation during whole simulation
    max_deviation = max(abs(real_level - setpoint))

    # Attack-period deviation
    attack_period_level = real_level[attack_start:attack_end + 1]
    attack_period_deviation = abs(attack_period_level - setpoint)

    max_attack_deviation = max(attack_period_deviation)
    mean_attack_deviation = np.mean(attack_period_deviation)

    # Steady-state error
    steady_state_error = abs(setpoint - real_level[-1])

    # Output boundedness
    lower_bound = 0
    upper_bound = 15
    output_bounded = np.all((real_level >= lower_bound) & (real_level <= upper_bound))

    return {
        "Overshoot": overshoot,
        "Maximum deviation": max_deviation,
        "Max deviation during attack": max_attack_deviation,
        "Mean deviation during attack": mean_attack_deviation,
        "Steady-state error": steady_state_error,
        "Output bounded": output_bounded
    }


# -----------------------------
# Run all simulations
# -----------------------------
normal_results = run_simulation("normal")
sensor_results = run_simulation("sensor_manipulation")
delay_results = run_simulation("delay")
injection_results = run_simulation("false_injection")

normal_metrics = calculate_metrics(normal_results)
sensor_metrics = calculate_metrics(sensor_results)
delay_metrics = calculate_metrics(delay_results)
injection_metrics = calculate_metrics(injection_results)


# -----------------------------
# Print combined comparison table
# -----------------------------
print("\nCombined Performance Metrics Comparison")
print("---------------------------------------------------------------------------------------------------------")
print(f"{'Metric':35s} {'Normal':15s} {'Sensor Manip.':15s} {'Delay Attack':15s} {'False Injection':15s}")
print("---------------------------------------------------------------------------------------------------------")

for metric in normal_metrics:
    values = [
        normal_metrics[metric],
        sensor_metrics[metric],
        delay_metrics[metric],
        injection_metrics[metric]
    ]

    formatted_values = []

    for value in values:
        if isinstance(value, (bool, np.bool_)):
            formatted_values.append(str(value))
        else:
            formatted_values.append(f"{value:.2f}")

    print(
        f"{metric:35s} "
        f"{formatted_values[0]:15s} "
        f"{formatted_values[1]:15s} "
        f"{formatted_values[2]:15s} "
        f"{formatted_values[3]:15s}"
    )

print("---------------------------------------------------------------------------------------------------------")


# -----------------------------
# Plot 1: Water level comparison
# -----------------------------
plt.figure()
plt.plot(normal_results["time"], normal_results["real_level"], label="Normal operation")
plt.plot(sensor_results["time"], sensor_results["real_level"], label="Sensor manipulation")
plt.plot(delay_results["time"], delay_results["real_level"], label="Delay attack")
plt.plot(injection_results["time"], injection_results["real_level"], label="False injection")
plt.axhline(y=setpoint, linestyle="--", label="Setpoint")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="Attack period")
plt.xlabel("Time (s)")
plt.ylabel("Water level")
plt.title("Water Level Comparison Across Attack Scenarios")
plt.legend()
plt.grid(True)
plt.savefig("combined_figure_1_water_level_all_attacks.png", dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# Plot 2: Pump signal comparison
# -----------------------------
plt.figure()
plt.plot(normal_results["time"], normal_results["pump"], label="Normal operation")
plt.plot(sensor_results["time"], sensor_results["pump"], label="Sensor manipulation")
plt.plot(delay_results["time"], delay_results["pump"], label="Delay attack")
plt.plot(injection_results["time"], injection_results["pump"], label="False injection")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="Attack period")
plt.xlabel("Time (s)")
plt.ylabel("Pump inflow")
plt.title("Controller Output Comparison Across Attack Scenarios")
plt.legend()
plt.grid(True)
plt.savefig("combined_figure_2_pump_signal_all_attacks.png", dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# Plot 3: Error comparison
# -----------------------------
plt.figure()
plt.plot(normal_results["time"], normal_results["error"], label="Normal operation")
plt.plot(sensor_results["time"], sensor_results["error"], label="Sensor manipulation")
plt.plot(delay_results["time"], delay_results["error"], label="Delay attack")
plt.plot(injection_results["time"], injection_results["error"], label="False injection")
plt.axhline(y=0, linestyle="--")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="Attack period")
plt.xlabel("Time (s)")
plt.ylabel("Deviation from setpoint")
plt.title("Physical Impact Comparison: Deviation from Desired State")
plt.legend()
plt.grid(True)
plt.savefig("combined_figure_3_error_all_attacks.png", dpi=300, bbox_inches="tight")
plt.close()


print("\nCombined plots saved successfully.")