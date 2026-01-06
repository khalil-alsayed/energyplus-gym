# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 01:06:21 2024

@author: kalsayed
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
import matplotlib.pyplot as plt

# Local project modules
from eplus_gym.envs.env import AmphitheaterEnv
from eplus_gym.envs.energyplus import (
    _resolve_output_dir,
    _find_project_root,
)
from eplus_gym.agents.Q_transformer.dqn_agent import Q_transformer

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

#%% Adaptation test (Q-transformer)--------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    
    
    env = AmphitheaterEnv(
        env_config={"output": str(output_dir),"location": "luxembourg"}, new_begin_month=1, new_end_month=1, new_begin_day=1, new_end_day=7,
        train=True)
    env.runner_config.csv = False

    best_score = -np.inf
    load_checkpoint = False
    total_deployment_load_checkpoint = False
    n_episodes = 3
    


    agent = Q_transformer(
        gamma=0.99,
        epsilon=1,
        lr=0.001,
        input_dims=(env.observation_space.shape),
        n_actions=env.action_space.n,
        mem_size=250000,
        eps_min=0.05,
        batch_size=128,
        replace=384,
        eps_dec=0.9 * n_episodes,
        chkpt_dir=checkpoint_dir,
        algo='Q-transformer',
        env_name='energyplus'
    )

    agent.q_eval.train()
    # if load_checkpoint:
    #     agent.load_models()

    n_steps = 0
    Episode = 0
    scores, eps_history, steps_array = [], [], []
    terminated = True
    
    # A rolling buffer to hold up to 96 recent states
    state_window = deque(maxlen=96)

    for i in range(n_episodes):
        Episode += 1
        done = False

        # If the episode just terminated, reset env
        
        # if terminated:  # in case of generalisation test
        if terminated :
            env.close()
            observation = env.reset()[0]  # or env.reset()
            #state_window.clear()
            state_window.append(observation)

        score = 0

        while not done:
            # If we don't yet have 96 states, fallback to random action (or old approach)
            if len(state_window) < 96 :
                action = np.random.choice(agent.action_space)
            else:
                # Convert the rolling window to a numpy array of shape (96, n_features)
                obs_seq = np.array(state_window, dtype=np.float32)
                # Use the agent's method that expects a 96-step sequence
                action = agent.choose_action_test(obs_seq)

            # Step the environment
            observation_, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated

            # Store the transition (single-step) in replay buffer
            # The replay buffer will internally assemble 96-step sequences for training
            
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()

            # Move on
            observation = observation_
            state_window.append(observation)
            n_steps += 1

        # End of episode
        agent.decrement_epsilon()
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('Episode:', Episode,
              'Score:', score,
              'Average score: %.1f' % avg_score,
              'Best score: %.2f' % best_score,
              'Epsilon: %.2f' % agent.epsilon,
              'Steps:', n_steps)

        if avg_score > best_score:
            if not total_deployment_load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    env.close()
    agent.save_models()
    
    #metrics_file = "C:/Users/khali/Desktop/code_phd_13_8_2024/Agent3/conference_paper/Q-tansformer/adaptation_test/nodropout/training_metrics.pkl"
    #save_training_metrics(metrics_file, scores, eps_history, steps_array)
    # -----------------------------------------------------------------
    # Plotting as in your original script
    # -----------------------------------------------------------------
    x = [i+1 for i in range(len(scores))]
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color=color)
    ax1.plot(x, scores, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Moving Average (Mean of every n_episodes)
    means = [np.mean(scores[i:i+n_episodes])
             for i in range(0, len(scores), n_episodes)]
    x_means = [i+n_episodes for i in range(0, len(scores), n_episodes)]
    ax1.plot(x_means, means, label='Mean Score per chunk',
             color='black', linestyle=':')
    ax1.legend(loc='upper left')


    # Cumulative Average
    cumulative_avg = np.cumsum(scores) / np.arange(1, len(scores)+1)
    ax1.plot(x, cumulative_avg, color='red',
             label='Cumulative Avg', linestyle='--')
    ax1.legend(loc='upper left')

    # Create a second y-axis for epsilon
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(x, eps_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Scores and Epsilon History Over Episodes')
    fig.tight_layout()
    plt.show()
    
    
#%%Deployment Q-transformer (Adaptation test)

env = AmphitheaterEnv(
    env_config={"output": str(output_dir),"location": "luxembourg"}, new_begin_month=1, new_end_month=1, new_begin_day=1, new_end_day=10,
    train=False)

env.runner_config.csv = True

# Create and load your DQN agent
agent = Q_transformer(
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
    algo='Q-transformer',
    env_name='energyplus'
)

agent.load_models()
agent.q_eval.eval()

scores = []
action_temp = []
action_flow = []

# --------------- Run the environment once, collect rewards ---------------
state_window = deque(maxlen=96)
done = False
observation = env.reset()[0]
score = 0

while not done:
    if len(state_window) < 96:
        action_idx = np.random.choice(agent.action_space)
        score=0
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
    action_temp.append(
        env._rescale(action[0] * action[2], range1=(0, 8-1), range2=[15, 30])
    )
    action_flow.append(
        env._rescale(action[1] * action[2], range1=(0, 8-1), range2=[0.3, 5])
    )
    scores.append(reward)

print("Score (entire run): ", score)
env.close()
    


# -------------------------------------------------------------
#  CONFIGURATION  – edit here if you want different colours etc.
# -------------------------------------------------------------
CSV_PATH = Path(output_dir) / "eplusout.csv"

DATE_SLICE = slice(96, 96*17)             # one week → 672 rows
#DATE_SLICE = slice(192, 288)             # one day

SMOOTH_WIN = 15                          # Savitzky–Golay window (odd)
SMOOTH_POLY= 3                           # Savitzky–Golay poly order
FIGSIZE    = (50, 7)

# matplotlib global style
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams.update({"font.size": 12, "axes.titleweight": "bold",
                     # NEW — bigger tick labels, axis labels and legend text
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.labelsize":  20,
    "legend.fontsize": 20,})
plt.rcParams["xtick.labelsize"] = 20   # bigger numbers on x-axes
plt.rcParams["ytick.labelsize"] = 20   # bigger numbers on y-axes
# -------------------------------------------------------------
#  LOAD & PREP DATA
# -------------------------------------------------------------

data = pd.read_csv(CSV_PATH)
# Fix datetime column
data["Date/Time"] = (
    "2020/" + data["Date/Time"].str.strip()          # add dummy year
                .str.replace(r"\s+", " ", regex=True)
                .str.replace("24:00:00", "23:59:59")
)
data["Date/Time"] = pd.to_datetime(
    data["Date/Time"], format="%Y/%m/%d %H:%M:%S"
).rename("datetime")
data.set_index("Date/Time", inplace=True)

# Slice one week
week = data.iloc[DATE_SLICE]

# Extract columns ------------------------------------------------
zoneT = week["TZ_AMPHITHEATER:Zone Mean Air Temperature [C](TimeStep)"]
htgThr= week["HTG HVAC 1 ADJUSTED BY 1.1 F:Schedule Value [](TimeStep)"]
clgThr= week["CLG HVAC 1 ADJUSTED BY 0 F:Schedule Value [](TimeStep)"]
oat   = week["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"]
occ   = week["MAISON DU SAVOIR AUDITORIUM OCC:Schedule Value [](TimeStep)"]
ahu   = week["AHUS ONOFF:Schedule Value [](TimeStep)"]
temp_sp = action_temp[DATE_SLICE]
flow_sp = action_flow[DATE_SLICE]

# -------------------------------------------------------------
#  1.  TEMPERATURE & SET-POINTS
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(zoneT, label="Zone T", lw=2)
ax.plot(oat,   label="Outdoor T", lw=2, color="tab:green")
ax.plot(htgThr, label="Heating threshold", ls="--", color="tab:red")
ax.plot(clgThr, label="Cooling threshold", ls=":",  color="tab:red")
ax.step(zoneT.index, temp_sp, label="Supply-air T set-point", color="tab:orange")
# --- bold “DDQN Bi-LSTM” label – choose the spot yourself ----
ax.text(
    0.3, 0.3,                # (x, y) in axes fraction: 0–1 left→right / bottom→top
    "Q-Transformer",
    transform=ax.transAxes,    # interpret coords in axes space
    fontsize=25,
    fontweight="bold",
)
ax.set(
    ylabel="Temperature (°C)",
)

ax.xaxis.set_major_formatter(
    plt.matplotlib.dates.DateFormatter("%b %d  %H:%M")
)
plt.xticks(rotation=0)

# Twin axes for occupancy & AHU
ax_occ = ax.twinx()
ax_occ.step(zoneT.index, occ*399, label="Occupancy", lw=1.8, color="k")
ax_occ.set_ylabel("Occupants")
ax_occ.set_ylim(0, 400)

ax_ahu = ax.twinx()
ax_ahu.spines.right.set_position(("axes", 1.08))
ax_ahu.step(zoneT.index, ahu, label="AHU state", ls="--", color="grey")
ax_ahu.set_yticks([0, 1]); ax_ahu.set_ylabel("AHU")

ax_ahu.spines["right"].set_visible(True)
ax_ahu.spines["right"].set_color("black")     # darker line
ax_ahu.spines["right"].set_linewidth(1.2)     # a bit thicker

ax_flow = ax.twinx()
ax_flow.spines.right.set_position(("axes", 1.04))
ax_flow.step(zoneT.index, flow_sp, label="OAC mass flow", lw=1.6, color="tab:cyan")
ax_flow.set_ylabel("Outdoor-air flow rate (kg/s)")

ax_flow.spines["right"].set_visible(True)
ax_flow.spines["right"].set_color("black")
ax_flow.spines["right"].set_linewidth(1.2)
# Collect legends from all axes
lines, labels = [], []
for a in (ax, ax_occ, ax_ahu, ax_flow):
    L, lab = a.get_legend_handles_labels()
    lines += L; labels += lab
ax.legend(lines, labels, loc="upper left")

# 1) remove the white grid lines completely
for a in (ax_occ, ax_ahu, ax_flow):
    a.grid(False)

# 2) shrink right margin so no wasted white area
fig.subplots_adjust(right=0.82)


LEGEND_X = 0.35   # 0 = left edge of axes, 1 = right edge, >1 = outside right
LEGEND_Y = 0.95   # 0 = bottom edge,   1 = top edge,   >1 = above axes
# merge legends
lines, labels = [], []
for a in (ax, ax_occ, ax_ahu, ax_flow):
    L, lab = a.get_legend_handles_labels()
    lines += L; labels += lab

# 8 items → 4 columns ⇒ 2 rows
ax.legend(
    lines, labels,
    loc="upper center",            # anchor at centre-top …
    bbox_to_anchor=(LEGEND_X, LEGEND_Y),    # … just above the axes
    ncol=2,                        # 4 columns → 2 rows
    fontsize=22,
    frameon=False                  # optional: no box edge
)
# ... legend code ...

# -------------------------------------------------------------
#  Add rose-tinted background for the deployment period
# -------------------------------------------------------------
split_dt = pd.Timestamp("2020-01-08 00:00:00")   # start of 8 Jan

ax.axvspan(split_dt, zoneT.index[-1],           # ← from split to end
           facecolor="#e6f7e6", alpha=1,     # ← light rose
           zorder=-5)                           # ← behind every artist



#--------------------------------------------------------------------------


ax.set_xlim(zoneT.index[0], zoneT.index[-1])
ax.margins(x=0)
plt.text(0.86, 0.99, f'Total Score: {score:.2f}', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')

plt.show()



# -------------------------------------------------------------
#  2.  ENERGY CONSUMPTION
# -------------------------------------------------------------
elec = week["Electricity:HVAC [J](TimeStep)"]
heat = week["Heating:DistrictHeatingWater [J](TimeStep)"]

fig, ax1 = plt.subplots(figsize=FIGSIZE)
ax1.step(elec.index, elec, color="tab:blue", label="Electricity: HVAC [J]")
ax1.set_ylabel("Electricity [J]")
ax1.xaxis.set_major_formatter(
    plt.matplotlib.dates.DateFormatter("%b %d  %H:%M")
)
plt.xticks(rotation=0)

ax2 = ax1.twinx()
ax2.step(heat.index, heat, color="tab:orange", label="District Heating [J]")
ax2.set_ylabel("District Heating [J]")


lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines+lines2, labels+labels2, loc="upper left",fontsize=14)
# 1) remove the white grid lines completely
for a in (ax,ax2):
    a.grid(False)
# Annotate totals
tot_elec = elec.sum(); tot_heat = heat.sum()
ax1.text(0.86, 0.95, f"∑ Elec: {tot_elec/1e9:.2f} GJ",
         transform=ax1.transAxes,fontsize=20)
ax1.text(0.86, 0.88, f"∑ Heat: {tot_heat/1e9:.2f} GJ",
         transform=ax1.transAxes,fontsize=20)
ax1.set_xlim(elec.index[0], elec.index[-1])
ax1.margins(x=0)
ax1.text(
    0.3, 0.3,            # ← move left↔right / down↕up as you wish
    "Q-Transformer",
    transform=ax1.transAxes,
    fontsize=25,
    fontweight="bold",
)
# merge legends
lines, labels   = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines + lines2, labels + labels2,
           loc="upper left",            # anchor-point = upper-left corner
           bbox_to_anchor=(0.3, LEGEND_Y),# move 8 % of axes-width to the LEFT
           fontsize=22, frameon=False)  # no box edge if you prefer
# -------------------------------------------------------------
#  Add rose-tinted background for the deployment period
# -------------------------------------------------------------
split_dt = pd.Timestamp("2020-01-08 00:00:00")   # start of 8 Jan

ax1.axvspan(split_dt, zoneT.index[-1],           # ← from split to end
           facecolor="#e6f7e6", alpha=1,     # ← light rose
           zorder=-5)                           # ← behind every artist
# -------------------------------------------------------------
#  3.  CO₂ CONCENTRATION
# -------------------------------------------------------------
co2 = week["TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)"]

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(co2.index, co2, color="tab:purple")
ax.set(
    ylabel="CO2 [ppm]",
)
ax.xaxis.set_major_formatter(
    plt.matplotlib.dates.DateFormatter("%b %d  %H:%M")
)
ax.text(
    0.3, 0.3,            # same idea—adjust to suit this plot
    "Q-Transformer",
    transform=ax.transAxes,
    fontsize=25,
    fontweight="bold",
)
plt.xticks(rotation=0)

tot_co2 = co2.sum()
# -------------------------------------------------------------
#  Add rose-tinted background for the deployment period
# -------------------------------------------------------------
split_dt = pd.Timestamp("2020-01-08 00:00:00")   # start of 8 Jan

ax.axvspan(split_dt, zoneT.index[-1],           # ← from split to end
           facecolor="#e6f7e6", alpha=1,     # ← light rose
           zorder=-5)                           # ← behind every artist
ax.set_xlim(co2.index[0], co2.index[-1])
ax.margins(x=0)
plt.show()    
   


#%% Generalisation test

n_episodes = 366


env = AmphitheaterEnv(
    env_config={"output": str(output_dir), "location": "USA"}, new_begin_month=1, new_end_month=12, new_begin_day=1, new_end_day=31,
    train=False, w_t=80, w_co2=7, w_dh=1, w_elc=4)
env.runner_config.csv = True



# Create and load your DQN agent
agent = Q_transformer(
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
    algo='Q-transformer',
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

