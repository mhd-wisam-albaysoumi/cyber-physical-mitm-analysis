import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Simulation settings
# -----------------------------
time_steps = 250
dt = 0.1

# -----------------------------
# Water tank parameters
# -----------------------------
setpoint = 10.0          # Desired water level
water_level = 5.0        # Initial water level
outflow_rate = 0.05      # Natural outflow from tank

# -----------------------------
# Controller parameter
# -----------------------------
kp = 0.8                 # Proportional controller gain

# -----------------------------
# MITM attack settings
# -----------------------------
attack_start = 80
attack_end = 160
attack_offset = -3.0     # Attacker makes sensor value look lower


# -----------------------------
# Data storage
# -----------------------------
time_data = []
real_level_data = []
measured_level_data = []
pump_data = []
error_data = []


# -----------------------------
# Simulation loop
# -----------------------------
for t in range(time_steps):
    current_time = t * dt

    # Sensor reads the real water level
    measured_level = water_level

    # MITM attack manipulates the sensor value
    attack_active = attack_start <= t <= attack_end

    if attack_active:
        measured_level = measured_level + attack_offset

    # Controller calculates error based on measured value
    controller_error = setpoint - measured_level

    # Simple proportional controller
    pump_inflow = kp * controller_error

    # Pump cannot have negative flow
    if pump_inflow < 0:
        pump_inflow = 0

    # Water tank dynamics
    water_level = water_level + (pump_inflow - outflow_rate * water_level) * dt

    # Store results
    time_data.append(current_time)
    real_level_data.append(water_level)
    measured_level_data.append(measured_level)
    pump_data.append(pump_inflow)
    error_data.append(setpoint - water_level)


# -----------------------------
# Convert lists to numpy arrays
# -----------------------------
time_data = np.array(time_data)
real_level_data = np.array(real_level_data)
measured_level_data = np.array(measured_level_data)
pump_data = np.array(pump_data)
error_data = np.array(error_data)


# -----------------------------
# Performance metrics
# -----------------------------
overshoot = max(real_level_data) - setpoint
max_deviation = max(abs(real_level_data - setpoint))
steady_state_error = abs(setpoint - real_level_data[-1])

print("Performance Metrics")
print("-------------------")
print(f"Overshoot: {overshoot:.2f}")
print(f"Maximum deviation from setpoint: {max_deviation:.2f}")
print(f"Steady-state error: {steady_state_error:.2f}")


# -----------------------------
# Plot 1: Water level
# -----------------------------
plt.figure()
plt.plot(time_data, real_level_data, label="Real water level")
plt.plot(time_data, measured_level_data, label="Measured level received by controller")
plt.axhline(y=setpoint, linestyle="--", label="Setpoint")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="MITM attack period")
plt.xlabel("Time (s)")
plt.ylabel("Water level")
plt.title("Water Tank Level Under MITM Attack")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# Plot 2: Pump signal
# -----------------------------
plt.figure()
plt.plot(time_data, pump_data, label="Pump inflow")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="MITM attack period")
plt.xlabel("Time (s)")
plt.ylabel("Pump inflow")
plt.title("Controller Output During MITM Attack")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# Plot 3: Error from setpoint
# -----------------------------
plt.figure()
plt.plot(time_data, error_data, label="Deviation from setpoint")
plt.axhline(y=0, linestyle="--")
plt.axvspan(attack_start * dt, attack_end * dt, alpha=0.2, label="MITM attack period")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.title("Physical Impact: Deviation from Desired State")
plt.legend()
plt.grid(True)
plt.show()