# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 07:30:20 2026

@author: khali
"""

#%% library part

# Standard library
import os
import pickle
from collections import deque
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd

# Local project modules
from eplus_gym.envs.env import AmphitheaterEnv
from eplus_gym.envs.energyplus import (
    _resolve_output_dir,
    _find_project_root,
)
from eplus_gym.agents.ddqn_bilstm.dqn_agent import DDQN_BiLSTM


# -----------------------------------------------------------------------------
# Helper utilities                                                            |
# -----------------------------------------------------------------------------

def _ensure_dir(path: str):
    """Create parent directory if it does not yet exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

# -------- training -----------------------------------------------------------

def save_training_metrics(filepath: str, scores, eps_history, steps_array):
    """Persist training arrays to *filepath* (Pickle)."""
    _ensure_dir(filepath)
    with open(filepath, "wb") as f:
        pickle.dump({
            "scores": scores,
            "eps_history": eps_history,
            "steps_array": steps_array,
        }, f)

# 1. Get the absolute path to the main directory 
#    This will be /.../energyplus-gym/    
output_dir = Path(_resolve_output_dir(None))
# 2. recreate gitkeep
output_root = Path(_resolve_output_dir(None))  # your resolver
(output_root / ".gitkeep").touch(exist_ok=True)

checkpoint_dir = _resolve_output_dir(None, default_name="runs/checkpoints")





      


    



#%% Generalisation test (DDQN_BiLSTM)

n_episodes = 366


env = AmphitheaterEnv(
    env_config={"output": str(output_dir), "location": "USA"}, new_begin_month=1, new_end_month=12, new_begin_day=1, new_end_day=31,
    train=False, w_t=80, w_co2=7, w_dh=1, w_elc=4)
env.runner_config.csv = True



# Create and load your DQN agent
agent = DDQN_BiLSTM(
    gamma=0.99,
    epsilon=0,
    lr=0.001,
    input_dims=(env.observation_space.shape),
    n_actions=env.action_space.n,
    mem_size=40000,
    eps_min=0.05,
    batch_size=128,
    replace=384,
    eps_dec=0.9 * n_episodes * 366,
    chkpt_dir=checkpoint_dir,
    algo='DDQN_BiLSTM',
    env_name='energyplus'
)

agent.load_models()
agent.q_eval.eval()

scores = []
action_sequence = []
action_sequence_flowrate = []

# --------------- Run the environment once, collect rewards ---------------
state_window = deque(maxlen=96)
done = False
observation = env.reset()[0]
score = 0

while not done:
    if len(state_window) < 96:
        action_idx = np.random.choice(agent.action_space)
    else:
        obs_seq = np.array(state_window, dtype=np.float32)
        action_idx = agent.choose_action_test(obs_seq)
    
    observation_, reward, terminated, truncated, _ = env.step(action_idx)
    action = env.valid_actions[action_idx]  # discrete -> continuous

    score += reward
    done = terminated or truncated
    observation = observation_
    state_window.append(observation)
    # Convert to real-world scale (optional if you need it)
    action_sequence.append(
        env._rescale(action[0] * action[2], range1=(0, 4-1), range2=[15, 30])
    )
    action_sequence_flowrate.append(
        env._rescale(action[1] * action[2], range1=(0, 4-1), range2=[0.3, 5])
    )
    scores.append(reward)

print("Score (entire run): ", score)
env.close()

# --------------------------- Global Accumulators ---------------------------
total_score = 0.0
co2_violation_count = 0
iat_violation_high_count = 0
iat_violation_low_count = 0
Indoor_violation_count=0
Heating_Distict_total = 0.0
Electricity_Hvac_total = 0.0
Electricity_Plant_total = 0.0
Heating_Coil_total = 0.0

# If you want total CO2 ppm over all steps (not typical, but shown for completeness)
CO2_Concentration_total = 0.0

# Each episode/day is 96 timesteps
first_timestep = -96
last_timestep = 0

# ---------------- Main loop over 366 days/chunks --------------------------
for i in range(n_episodes):
    first_timestep += 96
    last_timestep += 96

    # 1) Compute chunk score
    chunk_score = float(
    pd.DataFrame(scores)
    .head(last_timestep)
    .tail(last_timestep - first_timestep)
    .sum()
    .iloc[0]  # <-- Add this
    )
    total_score += chunk_score

    # 2) Read CSV and slice the data for this chunk
    csv_path = (_find_project_root() / "eplus_outputs" / "eplusout.csv")
    data = pd.read_csv(csv_path)

    # Fix date/time
    data['Date/Time'] = '2020/' + data['Date/Time'].str.strip()
    data['Date/Time'] = data['Date/Time'].str.replace(r'\s+', ' ', regex=True)
    data['Date/Time'] = data['Date/Time'].str.replace('24:00:00', '23:59:59')
    data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%Y/%m/%d %H:%M:%S')

    # Slice chunk
    chunk_data = data.head(last_timestep).tail(last_timestep - first_timestep).copy()

    # Extract needed columns
    temp_zone = chunk_data['TZ_AMPHITHEATER:Zone Mean Air Temperature [C](TimeStep)']
    occupancy = chunk_data['MAISON DU SAVOIR AUDITORIUM OCC:Schedule Value [](TimeStep)']
    co2_conc = chunk_data['TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)']
    
    # 2.5) Count indoor Violations (only occupant>0)
    Indoor_violation_chunk = (((co2_conc > 1000) | (temp_zone > 24) | (temp_zone < 21) ) & (occupancy > 0)).sum()
    Indoor_violation_count += Indoor_violation_chunk
    
    # 3) Count Violations (only occupant>0)
    co2_violation_chunk = ((co2_conc > 1000) & (occupancy > 0)).sum()
    co2_violation_count += co2_violation_chunk

    iat_high_chunk = ((temp_zone > 24) & (occupancy > 0)).sum()
    iat_low_chunk = ((temp_zone < 21) & (occupancy > 0)).sum()
    iat_violation_high_count += iat_high_chunk
    iat_violation_low_count += iat_low_chunk

    # 4) Energy in Joules
    electricity_hvac = chunk_data['Electricity:HVAC [J](TimeStep)']
    heating_district = chunk_data['Heating:DistrictHeatingWater [J](TimeStep)']
    electricity_plant = chunk_data['Electricity:Plant [J](TimeStep)']
    heating_coil = chunk_data['HeatingCoils:EnergyTransfer [J](TimeStep)']

    # Sum them for this chunk
    Heating_Distict_total += heating_district.sum()
    Electricity_Hvac_total += electricity_hvac.sum()
    Electricity_Plant_total += electricity_plant.sum()
    Heating_Coil_total += heating_coil.sum()

    # (Optional) sum of CO2 concentration over this chunk
    CO2_Concentration_total += co2_conc.sum()

# -------------------------- Final Stats -------------------------------------
average_score = total_score / n_episodes
iat_violation_count = iat_violation_high_count + iat_violation_low_count

# Convert Joules -> MJ by dividing by 1e6, then average per day (div by 366)
avg_heating_district_MJ = (Heating_Distict_total / n_episodes) / 1e6
avg_elec_hvac_MJ = (Electricity_Hvac_total / n_episodes) / 1e6
avg_elec_plant_MJ = (Electricity_Plant_total / n_episodes) / 1e6
avg_heating_coil_MJ = (Heating_Coil_total / n_episodes) / 1e6
avg_ene=avg_heating_district_MJ+avg_elec_hvac_MJ
print("\n============ FINAL DEPLOYMENT STATS (No Plots) ============")
print(f"Average Score (over {n_episodes} episodes): {average_score:.3f}")
print(f"Total Indoor Violations (occupant>0): {Indoor_violation_count}")
print(f"Total CO2 Violations (CO2>1000, occupant>0): {co2_violation_count}")
print(f"Total IAT Violations (occupant>0): {iat_violation_count}")
print(f"  --> High IAT Violations (IAT>24): {iat_violation_high_count}")
print(f"  --> Low IAT Violations (IAT<21): {iat_violation_low_count}")
print(f"Average Daily Energy consumption [MJ]: {avg_ene:.3f}")
print(f"Average Daily District Heating [MJ]: {avg_heating_district_MJ:.3f}")
print(f"Average Daily HVAC Electricity [MJ]: {avg_elec_hvac_MJ:.3f}")
print(f"Average Daily Plant Electricity [MJ]: {avg_elec_plant_MJ:.3f}")
print(f"Average Daily Heating Coil [MJ]: {avg_heating_coil_MJ:.3f}")

print("===========================================================")           

