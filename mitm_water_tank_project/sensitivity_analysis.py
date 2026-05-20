import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("sensitivity_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

BASE = {
    "time_steps": 250,
    "dt": 0.1,
    "setpoint": 10.0,
    "initial_water_level": 5.0,
    "outflow_rate": 0.05,
    "kp": 0.8,
    "attack_start": 80,
    "attack_end": 160,
    "attack_offset": -3.0,
    "delay_steps": 40,
    "false_sensor_value": 6.0,
    "lower_bound": 0.0,
    "upper_bound": 15.0,
}


def run_simulation(
    attack_type="normal",
    time_steps=BASE["time_steps"],
    dt=BASE["dt"],
    setpoint=BASE["setpoint"],
    initial_water_level=BASE["initial_water_level"],
    outflow_rate=BASE["outflow_rate"],
    kp=BASE["kp"],
    attack_start=BASE["attack_start"],
    attack_end=BASE["attack_end"],
    attack_offset=BASE["attack_offset"],
    delay_steps=BASE["delay_steps"],
    false_sensor_value=BASE["false_sensor_value"],
):
    water_level = initial_water_level

    time_data = []
    real_level_data = []
    measured_level_data = []
    pump_data = []
    process_error_data = []
    controller_error_data = []

    delay_steps = max(1, int(delay_steps))
    sensor_buffer = [initial_water_level] * delay_steps

    for k in range(time_steps):
        current_time = k * dt

        real_sensor_value = water_level
        attack_active = attack_start <= k <= attack_end

        if attack_type == "sensor_manipulation" and attack_active:
            measured_level = real_sensor_value + attack_offset

        elif attack_type == "delay" and attack_active:
            measured_level = sensor_buffer.pop(0)
            sensor_buffer.append(real_sensor_value)

        elif attack_type == "false_injection" and attack_active:
            measured_level = false_sensor_value

        else:
            measured_level = real_sensor_value
            sensor_buffer.pop(0)
            sensor_buffer.append(real_sensor_value)

        controller_error = setpoint - measured_level
        pump_inflow = kp * controller_error
        pump_inflow = max(0.0, pump_inflow)

        water_level = water_level + (pump_inflow - outflow_rate * water_level) * dt

        time_data.append(current_time)
        real_level_data.append(water_level)
        measured_level_data.append(measured_level)
        pump_data.append(pump_inflow)
        process_error_data.append(setpoint - water_level)
        controller_error_data.append(controller_error)

    return {
        "time": np.array(time_data),
        "real_level": np.array(real_level_data),
        "measured_level": np.array(measured_level_data),
        "pump": np.array(pump_data),
        "process_error": np.array(process_error_data),
        "controller_error": np.array(controller_error_data),
    }


def calculate_metrics(
    results,
    setpoint=BASE["setpoint"],
    attack_start=BASE["attack_start"],
    attack_end=BASE["attack_end"],
    lower_bound=BASE["lower_bound"],
    upper_bound=BASE["upper_bound"],
):
    h = results["real_level"]
    attack_h = h[attack_start:attack_end + 1]
    attack_dev = np.abs(attack_h - setpoint)

    return {
        "Overshoot": max(0.0, float(np.max(h) - setpoint)),
        "Maximum deviation": float(np.max(np.abs(h - setpoint))),
        "Max deviation during attack": float(np.max(attack_dev)),
        "Mean deviation during attack": float(np.mean(attack_dev)),
        "Steady-state error": float(abs(h[-1] - setpoint)),
        "Output bounded": bool(np.all((h >= lower_bound) & (h <= upper_bound))),
    }


def run_base_comparison():
    scenarios = {
        "Normal": "normal",
        "Sensor manipulation": "sensor_manipulation",
        "Delay attack": "delay",
        "False injection": "false_injection",
    }

    rows = []
    results_by_name = {}

    for name, attack_type in scenarios.items():
        results = run_simulation(attack_type=attack_type)
        results_by_name[name] = results
        metrics = calculate_metrics(results)
        rows.append({"Scenario": name, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "base_comparison_metrics.csv", index=False)

    plot_base_figures(results_by_name)
    return df


def plot_base_figures(results_by_name):
    attack_start_time = BASE["attack_start"] * BASE["dt"]
    attack_end_time = BASE["attack_end"] * BASE["dt"]

    plt.figure(figsize=(10, 5))
    for name, results in results_by_name.items():
        plt.plot(results["time"], results["real_level"], label=name)

    plt.axhline(y=BASE["setpoint"], linestyle="--", label="Setpoint")
    plt.axvspan(attack_start_time, attack_end_time, alpha=0.2, label="Attack interval")
    plt.xlabel("Time (s)")
    plt.ylabel("Water level (units)")
    plt.title("Water Level Under Normal Operation and MITM Attack Scenarios")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "figure_1_water_level_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    for name, results in results_by_name.items():
        plt.plot(results["time"], results["pump"], label=name)

    plt.axvspan(attack_start_time, attack_end_time, alpha=0.2, label="Attack interval")
    plt.xlabel("Time (s)")
    plt.ylabel("Pump inflow (units/s)")
    plt.title("Controller Output / Pump Inflow")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "figure_2_pump_inflow_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    for name, results in results_by_name.items():
        deviation = results["real_level"] - BASE["setpoint"]
        plt.plot(results["time"], deviation, label=name)

    plt.axhline(y=0, linestyle="--", label="Zero deviation")
    plt.axvspan(attack_start_time, attack_end_time, alpha=0.2, label="Attack interval")
    plt.xlabel("Time (s)")
    plt.ylabel("Deviation from setpoint (units)")
    plt.title("Physical Deviation from Desired Water Level")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "figure_3_deviation_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def sensitivity_table(parameter_name, values, attack_type, label):
    rows = []

    for value in values:
        params = dict(BASE)

        if parameter_name == "attack_offset":
            params["attack_offset"] = value

        elif parameter_name == "delay_steps":
            params["delay_steps"] = value

        elif parameter_name == "false_sensor_value":
            params["false_sensor_value"] = value

        elif parameter_name == "attack_duration":
            params["attack_end"] = params["attack_start"] + int(value)

        elif parameter_name == "kp":
            params["kp"] = value

        elif parameter_name == "attack_start":
            duration = BASE["attack_end"] - BASE["attack_start"]
            params["attack_start"] = int(value)
            params["attack_end"] = int(value) + duration

        else:
            raise ValueError("Unknown parameter")

        results = run_simulation(
            attack_type=attack_type,
            time_steps=params["time_steps"],
            dt=params["dt"],
            setpoint=params["setpoint"],
            initial_water_level=params["initial_water_level"],
            outflow_rate=params["outflow_rate"],
            kp=params["kp"],
            attack_start=params["attack_start"],
            attack_end=params["attack_end"],
            attack_offset=params["attack_offset"],
            delay_steps=params["delay_steps"],
            false_sensor_value=params["false_sensor_value"],
        )

        metrics = calculate_metrics(
            results,
            setpoint=params["setpoint"],
            attack_start=params["attack_start"],
            attack_end=params["attack_end"],
            lower_bound=params["lower_bound"],
            upper_bound=params["upper_bound"],
        )

        rows.append({
            "Parameter": parameter_name,
            "Value": value,
            "Attack type": attack_type,
            **metrics
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / f"sensitivity_{label}.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["Value"], df["Max deviation during attack"], marker="o", label="Max deviation during attack")
    plt.plot(df["Value"], df["Steady-state error"], marker="o", label="Steady-state error")
    plt.xlabel(parameter_name)
    plt.ylabel("Metric value (units)")
    plt.title(f"Sensitivity Analysis: {label}")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / f"sensitivity_{label}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return df


def run_all_sensitivity():
    all_tables = []

    all_tables.append(sensitivity_table(
        parameter_name="attack_offset",
        values=[-1.0, -2.0, -3.0, -4.0, -5.0],
        attack_type="sensor_manipulation",
        label="sensor_offset"
    ))

    all_tables.append(sensitivity_table(
        parameter_name="delay_steps",
        values=[5, 10, 20, 40, 60],
        attack_type="delay",
        label="delay_length"
    ))

    all_tables.append(sensitivity_table(
        parameter_name="false_sensor_value",
        values=[2.0, 4.0, 6.0, 8.0, 12.0],
        attack_type="false_injection",
        label="false_injection_value"
    ))

    all_tables.append(sensitivity_table(
        parameter_name="attack_duration",
        values=[20, 40, 80, 120],
        attack_type="false_injection",
        label="attack_duration_false_injection"
    ))

    all_tables.append(sensitivity_table(
        parameter_name="kp",
        values=[0.2, 0.4, 0.6, 0.8, 1.0],
        attack_type="sensor_manipulation",
        label="controller_gain_sensor_attack"
    ))

    all_tables.append(sensitivity_table(
        parameter_name="attack_start",
        values=[20, 50, 80, 120],
        attack_type="false_injection",
        label="attack_timing_false_injection"
    ))

    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "all_sensitivity_results.csv", index=False)

    return combined


if __name__ == "__main__":
    base_df = run_base_comparison()
    sensitivity_df = run_all_sensitivity()

    print("\nBase comparison metrics:")
    print(base_df.to_string(index=False))

    print("\nSensitivity analysis saved in:")
    print(OUTPUT_DIR.resolve())

    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.iterdir()):
        print("-", file.name)