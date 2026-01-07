# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 07:21:53 2026

@author: khali
"""
#%% library part

# Standard library
import os
import pickle
from pathlib import Path

# Third-party
import pandas as pd
import matplotlib.pyplot as plt

# Local project modules
from eplus_gym.envs.env import AmphitheaterEnv
from eplus_gym.envs.energyplus import (
    _resolve_output_dir
)
from eplus_gym.agents.ddqn_mlp.dqn_agent import DDQN_MLP


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



#%%Deployment DDQN_MLP (Adaptation test)
n_episodes=50
env = AmphitheaterEnv(
    env_config={"output": str(output_dir),"location": "luxembourg"}, new_begin_month=1, new_end_month=1, new_begin_day=1, new_end_day=10,
    train=False)

env.runner_config.csv = True

# Create and load your DQN agent
agent = DDQN_MLP(
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
    algo='DDQN_MLP',
    env_name='energyplus'
)

agent.load_models()
agent.q_eval.eval()

scores = []
action_temp = []
action_flow = []

# --------------- Run the environment once, collect rewards ---------------
done = False
observation = env.reset()[0]
score = 0

while not done:
    
    action_idx = agent.choose_action_test(observation)
    
    observation_, reward, terminated, truncated, _ = env.step(action_idx)
    action = env.valid_actions[action_idx]  # discrete -> continuous

    score += reward
    done = terminated or truncated
    observation = observation_
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
   