# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 01:06:21 2024

@author: kalsayed
"""



#%% library part
from rleplus.examples.amphitheater.env import AmphitheaterEnv
from rleplus.env.energyplus import RunnerConfig
import random
from unittest.mock import patch
from pathlib import Path
import unittest
from gym.wrappers.record_video import RecordVideo
import numpy as np
import gym
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from deep_q_network import DeepQNetwork
import pickle

import torch.optim as optim
import torch as T
import warnings
from datetime import timedelta
# Suppress the VisibleDeprecationWarning
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#T.autograd.set_detect_anomaly(True)
#T.autograd.set_detect_anomaly(True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
np.set_printoptions(precision=2, suppress=True)
#from Record_video import RecordVide

#import inspect
#print(inspect.getsource(gym.wrappers.RecordVideo))

def modify_runperiod_dates(file_path, new_begin_month, new_end_month, 
                           new_begin_day, new_end_day):
    """
    Updates the Begin/End Month and Begin/End Day of Month fields
    in the RunPeriod object of an IDF file. Only lines within the
    RunPeriod object are modified; other objects remain unchanged.

    :param file_path: Path to the IDF file
    :param new_begin_month: (int) New Begin Month
    :param new_end_month:   (int) New End Month
    :param new_begin_day:   (int) New Begin Day of Month
    :param new_end_day:     (int) New End Day of Month
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_run_period = False

    for i, line in enumerate(lines):
        # 1. Detect the start of the RunPeriod object
        if line.strip().startswith("RunPeriod,"):
            in_run_period = True

        if in_run_period:
            # 2. BEGIN MONTH
            if '!- Begin Month' in line:
                parts = line.split(',')
                if len(parts) > 0:
                    idx = parts[0].rfind(' ')
                    if idx != -1:
                        parts[0] = parts[0][:idx+1] + str(new_begin_month)
                    else:
                        parts[0] = str(new_begin_month)
                    lines[i] = ','.join(parts)

            # 3. END MONTH
            elif '!- End Month' in line:
                parts = line.split(',')
                if len(parts) > 0:
                    idx = parts[0].rfind(' ')
                    if idx != -1:
                        parts[0] = parts[0][:idx+1] + str(new_end_month)
                    else:
                        parts[0] = str(new_end_month)
                    lines[i] = ','.join(parts)

            # 4. BEGIN DAY OF MONTH
            elif '!- Begin Day of Month' in line:
                parts = line.split(',')
                if len(parts) > 0:
                    idx = parts[0].rfind(' ')
                    if idx != -1:
                        parts[0] = parts[0][:idx+1] + str(new_begin_day)
                    else:
                        parts[0] = str(new_begin_day)
                    lines[i] = ','.join(parts)

            # 5. END DAY OF MONTH
            elif '!- End Day of Month' in line:
                parts = line.split(',')
                if len(parts) > 0:
                    idx = parts[0].rfind(' ')
                    if idx != -1:
                        parts[0] = parts[0][:idx+1] + str(new_end_day)
                    else:
                        parts[0] = str(new_end_day)
                    lines[i] = ','.join(parts)

            # 6. Detect the end of the RunPeriod object 
            #    (when we hit the semicolon)
            if ';' in line:
                in_run_period = False

    # 7. Write the modifications back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)



from rleplus.examples.amphitheater.env import AmphitheaterEnv
from rleplus.env.energyplus import RunnerConfig
import random
from unittest.mock import patch
from pathlib import Path
import unittest
from gym.wrappers.record_video import RecordVideo
import numpy as np
import gym
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from deep_q_network import DeepQNetwork
import torch.optim as optim
import torch as T
from dqn_agent import DQNAgent
from collections import deque
#T.autograd.set_detect_anomaly(True)
#T.autograd.set_detect_anomaly(True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
np.set_printoptions(precision=2, suppress=True)
#from Record_video import RecordVide

#import inspect
#print(inspect.getsource(gym.wrappers.RecordVideo))




def calculate_percentage_difference(array1, array2):
    """
    Calculate the percentage difference between two arrays of five values.
    The calculation is based on the formula:
    Percentage Difference = ((Value2 - Value1) / Value1) * 100

    Parameters:
    - array1: List of values for the first array (Value2 in the formula)
    - array2: List of values for the second array (Value1 in the formula)

    Returns:
    A dictionary containing the percentage difference for each label.
    """
    if len(array1) != 5 or len(array2) != 5:
        raise ValueError("Both arrays must contain exactly 5 values.")

    labels = [
        "Heating_Distict difference",
        "Electricity_Hvac difference",
        "Electricity_Plant difference",
        "Heating_Coil difference",
        "CO2_Concentration difference"
    ]

    differences = {}
    for i in range(5):
        try:
            percentage_difference = ((array1[i] - array2[i]) / array2[i]) * 100
        except ZeroDivisionError:
            percentage_difference = float('inf')  # Handle division by zero

        differences[labels[i]] = percentage_difference

    return differences

# Example usage:
array1 = [120, 250, 340, 450, 560]
array2 = [100, 200, 300, 400, 500]
result = calculate_percentage_difference(array1, array2)
for label, value in result.items():
    print(f"{label} = {value:.2f}%")


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


def load_training_metrics(filepath: str):
    """Return (scores, eps_history, steps_array) from Pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["scores"], data["eps_history"], data["steps_array"]

def save_deployment_timeseries(filepath: str, dataframe: pd.DataFrame):
    """Persist the deployment‑phase time‑series (CSV)."""
    _ensure_dir(filepath)
    dataframe.to_csv(filepath, index=False)


def load_deployment_timeseries(filepath: str) -> pd.DataFrame:
    """Load the saved deployment‑phase CSV with parsed datetime column."""
    return pd.read_csv(filepath, parse_dates=["Date/Time"])


#%% DDQN agent


file_path = 'C:/Users/kalsayed/rllib-energyplus/rleplus/examples/amphitheater/model.idf'
new_begin_month=1
new_end_month=12

new_begin_day = 1 # The new value for the Begin Day of Month
new_end_day = 31   # The new value for the End Day of Month

modify_runperiod_dates(file_path, new_begin_month, new_end_month, new_begin_day, new_end_day)
env = AmphitheaterEnv({"output": "/tmp/tests_output"},new_begin_month, new_end_month, new_begin_day, new_end_day)
env.runner_config.csv = False


best_score = -np.inf
total_deployment_load_checkpoint = False
n_episodes = 2



agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.001,
                         input_dims=(env.observation_space.shape),
                         n_actions=env.action_space.n, mem_size=10000, eps_min=0.05,
                         batch_size=1000, replace=384, eps_dec=0.9*n_episodes,
                         chkpt_dir='/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus', algo='DQNAgent',
                         env_name='energyplus')
        



n_steps = 0
Episode = 0
scores, eps_history, steps_array = [], [], []

for i in range(n_episodes):
     Episode += 1
     done = False
     observation = env.reset()[0]
     score = 0
     while not done:
         action = agent.choose_action_test(observation)
         #action = agent.choose_action_distillation(observation)
         observation_, reward, terminated, truncated, _ = env.step(
             action)
         score += reward
         done = terminated or truncated
         if not total_deployment_load_checkpoint:
             agent.store_transition(observation, action,
                                    reward, observation_, done)
             agent.learn()
         observation = observation_
         n_steps += 1
         '''
         #normal buffer b----------------------------------------------------------------------
         # actual state
         m=agent.memory.state_memory[:300]
         # next state
         m_next=agent.memory.new_state_memory[:300]
         #action
         m_action=agent.memory.action_memory[:300]
         #reward
         m_reward=agent.memory.reward_memory[:300]
         #done message
         m_done=agent.memory.terminal_memory[:300]
         #Transformer buffer B--------------------------------------------------------------------
         # actual state
         t=agent.memory.transformer_state_memory[:300]
         # next state
         t_next=agent.memory.transformer_next_state_memory[:300]
         #action
         t_action=agent.memory.transformer_action_memory[:300]
         #reward
         t_reward=agent.memory.transformer_reward_memory[:300]
         #done message
         t_done=agent.memory.transformer_done_memory[:300]
         '''
     agent.decrement_epsilon()
     scores.append(score)
     steps_array.append(n_steps)

     avg_score = np.mean(scores[-100:])
     print('Episode:', Episode, 'Score:', score,
           ' Average score: %.1f' % avg_score, 'Best score: %.2f' % best_score,
           'Epsilon: %.2f' % agent.epsilon, 'Steps:', n_steps)

     if avg_score > best_score:
         if not total_deployment_load_checkpoint:
             agent.save_models()
         best_score = avg_score

     eps_history.append(agent.epsilon)

env.close()



#training graphe------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------

x = [i+1 for i in range(len(scores))]
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Score', color=color)
ax1.plot(x, scores, color=color)
ax1.tick_params(axis='y', labelcolor=color)
# Moving Average (Mean of every 100 episodes)
means = [np.mean(scores[i:i+n_episodes])
         for i in range(0, len(scores), n_episodes)]
# X-axis points for the means
x_means = [i+n_episodes for i in range(0, len(scores), n_episodes)]
# Plot the rolling average data
ax1.plot(x_means, means, label='Mean Score per 100 Episodes',
         color='black', linestyle=':')
ax1.legend(loc='upper left')
# Cumulative Average
cumulative_avg = np.cumsum(scores) / np.arange(1, len(scores)+1)
# Plot the rolling average data
ax1.plot(x, cumulative_avg, color='red',
         label='Cumulative Avg', linestyle='--')
ax1.legend(loc='upper left')
# Create a second y-axis for the epsilon values
ax2 = ax1.twinx()
color = 'tab:green'
# we already handled the x-label with ax1
ax2.set_ylabel('Epsilon', color=color)
ax2.plot(x, eps_history, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Optional: Add a title and a tight layout
plt.title('Scores and Epsilon History Over Episodes')
fig.tight_layout()  # To ensure there's no overlap in the layout

# Save the plot to a file
#plt.savefig(figure_file)  # Saves the plot to the path specified in your script

# Show the plot
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------
#deployment--------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
file_path = 'C:/Users/kalsayed/rllib-energyplus/rleplus/examples/amphitheater/model.idf'
new_begin_day = days  # The new value for the Begin Day of Month
new_end_day = days   # The new value for the End Day of Month
modify_idf_file_in_place(file_path, new_begin_day, new_end_day)

env = AmphitheaterEnv(
    {"output": "/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus/tests_output"}, days)
env.runner_config.csv = True
#agent.epsilon=0



agent = DQNAgent(gamma=0.99, epsilon=0, lr=0.001,
                         input_dims=(env.observation_space.shape),
                         n_actions=env.action_space.n, mem_size=10000, eps_min=0.05,
                         batch_size=1000, replace=384, eps_dec=0.9*n_episodes,
                         chkpt_dir='/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus', algo='DQNAgent',
                         env_name='energyplus')

agent.load_models()

#agent.load_models_transfer_learning()
#agent.q_eval.load_cumulative_updates()
Scores = []

action_sequence = []
action_sequence_flowrate = []
action_sequence.append(15)
action_sequence_flowrate.append(0.3)
done = False
observation = env.reset()[0]

score = 0
while not done:
    
    #action = agent.choose_action_distillation(observation)  # new agent
    action = agent.choose_action_test(observation)
    #action=(0,10000,1)
    #action= (random.uniform(0, 10000),random.uniform(0, 10000),1)
    observation_, reward, terminated, truncated, _ = env.step(
        action)
    action = env.valid_actions[action]
    score += reward
    done = terminated or truncated
    observation = observation_
    action_sequence.append(env._rescale(action[0]*action[2], range1=(
        0, 99-1), range2=[15, 30]))
    action_sequence_flowrate.append(env._rescale(action[1]*action[2], range1=(
        0, 99-1), range2=[0.3, 5]))

Scores.append(score)
action_sequence.pop()
action_sequence_flowrate.pop()
print('Score: ', score)

env.close()


'''
totla_score_unexpected+=score_total_unexpected
t_heating_district+=total_heating_district
t_electricity_hvac+=total_electricity_hvac
t_electricity_plant+=total_electricity_plant
t_heating_coil+=total_heating_coil
t_CO2_concentration+=total_CO2_concentration
'''

#-----------------------------------------------------------------------------------------------------------------------------
#plot for temperatures variations
#---------------------------------------------------------------------------------------------------------------------------

# Step 1: Read the CSV file
# Update this to the path of your CSV file
file_path = 'C:/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus/tests_output/episode-0-0/eplusout.csv'
data = pd.read_csv(file_path)
# Add a default year to the 'Date/Time' column
data['Date/Time'] = '2023/' + data['Date/Time'].str.strip()

# Remove any extra spaces between the date and time
data['Date/Time'] = data['Date/Time'].str.replace(r'\s+', ' ', regex=True)

data['Date/Time'] = data['Date/Time'].str.replace('24:00:00', '23:59:59')
data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%Y/%m/%d %H:%M:%S')
 
 
# Step 2: Extract Necessary Data
datetime = data['Date/Time']
temp_zone = data['TZ_AMPHITHEATER:Zone Mean Air Temperature [C](TimeStep)']
htg_threshold = data['HTG HVAC 1 ADJUSTED BY 1.1 F:Schedule Value [](TimeStep)']
clg_threshold = data['CLG HVAC 1 ADJUSTED BY 0 F:Schedule Value [](TimeStep)']
outdoor_air_temp = data['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
AHU_ON_OFF = data['AHUS ONOFF:Schedule Value [](TimeStep)']
action_temp_set = action_sequence  # Assuming this is the column name for your action temperature set
  
# New variable for occupancy
occupancy = data['MAISON DU SAVOIR AUDITORIUM OCC:Schedule Value [](TimeStep)']

# Step 3: Plot the Data
fig, ax1 = plt.subplots(figsize=(12, 7))
 
# Increase font size
plt.rcParams.update({'font.size': 14})


# Plotting the Zone Mean Air Temperature
ax1.plot(datetime, temp_zone, label='Zone Mean Air Temperature', color='blue')

# Plotting the thresholds
ax1.plot(datetime, htg_threshold, label='Heating Threshold', color='red', linestyle='--')
ax1.plot(datetime, clg_threshold, label='Cooling Threshold', color='red', linestyle=':')

# Plotting the Outdoor Air Temperature in green
ax1.plot(datetime, outdoor_air_temp, label='Outdoor Air Temperature', color='green', linewidth=2)

# Plotting the Action Temperature Set in orange
ax1.step(datetime, action_temp_set, label='Supply Air Temperature Setpoint', color='orange', linewidth=2)

# Enhancing the plot
ax1.set_xlabel('Date/Time')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Zone Temperature and HVAC Thresholds')
ax1.legend(loc='upper left')
# Create x-axis labels to show every other datetime
# Automatically format the x-axis to show time first and then the date
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
# Set x-limits to avoid wasted space
ax1.set_xlim([datetime.min(), datetime.max()])
# Rotate the x-axis labels for better readability
plt.xticks(rotation=25,fontsize=12)

# Create a second y-axis to plot the number of occupants
ax2 = ax1.twinx()
ax2.step(datetime, occupancy * 399, label='Number of Occupants', color='black', linewidth=2)
ax2.set_ylabel('Number of Occupants')
ax2.set_ylim(0, 400)

# Create a third y-axis to plot the AHU_ON_OFF
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis outward
ax3.step(datetime, AHU_ON_OFF, label='AHU ON/OFF', color='grey', linewidth=1,linestyle='--',zorder=1)
ax3.set_ylabel('AHU ON/OFF')
ax3.set_ylim(-0.1, 1.1)
ax3.set_yticks([0, 1])  # Show only the values 0 and 1
# Add a vertical dashed line at the first occurrence of AHU_ON_OFF being 1
first_on_time = datetime[AHU_ON_OFF.eq(1)].iloc[0]  # Get the first time AHU_ON_OFF is 1
adjusted_time = first_on_time + timedelta(seconds=700)  # Adjust by 3 seconds to move the line to the right

ax3.axvline(x=adjusted_time, color='grey', linestyle='--', ymin=0.085, ymax=0.915, zorder=1,linewidth=1)

# Add a horizontal line from the y-axis to the vertical line
ax3.hlines(y=1, xmin=datetime.min(), xmax=adjusted_time, color='grey', linestyle='--', linewidth=1, zorder=1)

# Fourth axis for Outdoor Air Controller’s Air Mass Flow Rate
ax4 = ax1.twinx()
ax4.spines['right'].set_position(('outward', 120))
ax4.step(datetime, action_sequence_flowrate, label='Outdoor Air Controller Air Mass Flow Rate', color='skyblue', linewidth=2)
ax4.set_ylabel('OAC Air Mass Flow Rate')



# Combine legends from both y-axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_3, labels_3 = ax3.get_legend_handles_labels()
lines_4, labels_4 = ax4.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2 + lines_3+ lines_4, labels_1 + labels_2 + labels_3+ labels_4, loc='upper left',bbox_to_anchor=(0, 0.9), fontsize=10, framealpha=0)

# Adding score texts
plt.text(0.01, 0.99, f'Score: {score:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
#plt.text(0.01, 0.95, f'Unexpected Score: {score_total_unexpected:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Show plot
plt.tight_layout()
plt.show()


#-------------------------------------------------------------------------------------------------------------------------------------
#plot for energy consumption (version 1)
# Step 2: Extract Necessary Data
datetime = data['Date/Time']
electricity_hvac = data['Electricity:HVAC [J](TimeStep)']
heating_district = data['Heating:DistrictHeatingWater [J](TimeStep)']

# Step 3: Create Plot with Two Y-Axes
fig, ax1 = plt.subplots(figsize=(12, 7))

# Plotting Electricity:HVAC [J] on the first Y-axis
color = 'tab:green'
ax1.set_xlabel('Date/Time')
ax1.set_ylabel('Electricity:HVAC [J](TimeStep)', color=color)
ax1.step(datetime, electricity_hvac,
        label='Electricity:HVAC [J]', color=color)
ax1.tick_params(axis='y', labelcolor=color)
# Create x-axis labels to show every other datetime
# Automatically format the x-axis to show time first and then the date
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
# Set x-limits to avoid wasted space
ax1.set_xlim([datetime.min(), datetime.max()])
# Rotate the x-axis labels for better readability
plt.xticks(rotation=25,fontsize=12)

# Creating a second Y-axis for Heating:DistrictHeatingWater [J]
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:orange'
# we already handled the x-label with ax1
ax2.set_ylabel(
   'Heating:DistrictHeatingWater [J](TimeStep)', color=color)
ax2.step(datetime, heating_district,
        label='Heating:DistrictHeatingWater [J]', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Further customizations
fig.tight_layout()  # to adjust subplot parameters to give specified padding
plt.title('Energy Consumption over Time with Dual Y-Axis')
fig.legend(loc="upper left", bbox_to_anchor=(0.05, 0.9),framealpha=0)

# Calculate the total Heating:DistrictHeatingWater consumption
total_heating_district = heating_district.sum()
total_electricity_hvac=electricity_hvac.sum()
# Adding the total consumption to the top left of the plot
plt.text(0.01, 0.99, f'Total District Heating Water Consumption: {total_heating_district:.2e} J',
        transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.text(0.01, 0.95, f'Total electricity hvac Consumption: {total_electricity_hvac:.2e} J',
        transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Show plot
plt.show() 
        
#--------------------------------------------------------------------------------------------------------------------------------
#plot for energy consumption (version2)
# Step 2: Extract Necessary Data
datetime = data['Date/Time']
electricity_plant = data['Electricity:Plant [J](TimeStep)']
heating_coil = data['HeatingCoils:EnergyTransfer [J](TimeStep)']

# Step 3: Create Plot with Two Y-Axes
fig, ax1 = plt.subplots(figsize=(12, 7))

# Plotting Electricity:HVAC [J] on the first Y-axis
color = 'tab:green'
ax1.set_xlabel('Date/Time')
ax1.set_ylabel('Electricity:Plant [J](TimeStep)', color=color)
ax1.plot(datetime, electricity_plant,
        label='Electricity:Plant [J](TimeStep)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
# Automatically format the x-axis to show time first and then the date
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
# Set x-limits to avoid wasted space
ax1.set_xlim([datetime.min(), datetime.max()])
# Rotate the x-axis labels for better readability
plt.xticks(rotation=25,fontsize=12)

# Creating a second Y-axis for Heating:DistrictHeatingWater [J]
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:orange'
# we already handled the x-label with ax1
ax2.set_ylabel(
   'HeatingCoils:EnergyTransfer [J](TimeStep)', color=color)
ax2.plot(datetime, heating_coil,
        label='HeatingCoils:EnergyTransfer [J](TimeStep)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Further customizations
fig.tight_layout()  # to adjust subplot parameters to give specified padding
plt.title('Energy Consumption over Time with Dual Y-Axis')
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9),framealpha=0)


# Calculate the total Electricity:Plant consumption
total_Electricity_Plant = electricity_plant.sum()
# Adding the total consumption to the top left of the plot
plt.text(0.01, 0.99, f'Total Electricity:Plant consumption: {total_Electricity_Plant:.2e} J',
        transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Calculate the total heating coil consumption
total_heating_coil = heating_coil.sum()
# Adding the total consumption to the top left of the plot
plt.text(0.01, 0.95, f'total heating coil consumption: {total_heating_coil:.2e} J',
        transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


# Show plot
plt.show()
#-------------------------------------------------------------------------------------------------
#plot for CO2 concentration (version2)
# Step 2: Extract Necessary Data
datetime = data['Date/Time']
CO2_concentration = data['TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)']
# Step 3: Create Plot with Two Y-Axes
fig, ax1 = plt.subplots(figsize=(12, 7))

# Plotting Electricity:HVAC [J] on the first Y-axis
color = 'tab:green'
ax1.set_xlabel('Date/Time')
ax1.set_ylabel('TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)', color=color)
ax1.plot(datetime, CO2_concentration,                
        label='TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
# Create x-axis labels to show every other datetime
# Automatically format the x-axis to show time first and then the date
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
# Set x-limits to avoid wasted space
ax1.set_xlim([datetime.min(), datetime.max()])
# Rotate the x-axis labels for better readability
plt.xticks(rotation=25,fontsize=12)

# Further customizations
fig.tight_layout()  # to adjust subplot parameters to give specified padding
plt.title('Zone Air CO2 Concentration')
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9),framealpha=0)

total_CO2_concentration=CO2_concentration.sum()

# Adding the total consumption to the top left of the plot
plt.text(0.01, 0.95, f'total CO2 concentration: {total_CO2_concentration:.2e} ppm',
        transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Show plot
plt.show()





#%% test
total_score=0
n_episodes=100
for days in [3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]:     # november    
#for days in  [1,2, 3, 6,7,8, 9, 10, 13,14,15, 16, 17, 20,21,22, 23, 24, 27,28,29, 30] :       # january
#for days in [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]:  #mars
#for days in [3, 4, 5, 6,7, 10, 11, 12, 13,14, 17, 18, 19, 20,21, 24, 25, 26, 27,28]:    #febriery
#for days in [1,2,3,6,7,8,9,10,13,14,15,16,17,20,21,22,23,24,27,28,29,30]:    #avril

    #---------------------------------------------------------------------------------------------------------------------------------
    #deployment--------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------
    file_path = 'C:/Users/kalsayed/rllib-energyplus/rleplus/examples/amphitheater/model.idf'
    new_begin_day = days  # The new value for the Begin Day of Month
    new_end_day = days   # The new value for the End Day of Month
    modify_idf_file_in_place(file_path, new_begin_day, new_end_day)

    env = AmphitheaterEnv(
        {"output": "/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus/tests_output"}, days)
    env.runner_config.csv = True
    #agent.epsilon=0


    
    agent = DQNAgent(gamma=0.99, epsilon=0, lr=0.001,
                             input_dims=(env.observation_space.shape),
                             n_actions=env.action_space.n, mem_size=10000, eps_min=0.05,
                             batch_size=1000, replace=384, eps_dec=0.9*n_episodes,
                             chkpt_dir='/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus', algo='DQNAgent',
                             env_name='energyplus')

    agent.load_models()

    #agent.load_models_transfer_learning()
    #agent.q_eval.load_cumulative_updates()
    Scores = []

    action_sequence = []
    action_sequence_flowrate = []
    action_sequence.append(15)
    action_sequence_flowrate.append(0.3)
    done = False
    observation = env.reset()[0]

    score = 0
    while not done:
        
        #action = agent.choose_action_distillation(observation)  # new agent
        action = agent.choose_action_test(observation)             # old agent
        #action=(0,10000,1)
        #action= (random.uniform(0, 10000),random.uniform(0, 10000),1)
        observation_, reward, terminated, truncated, _ = env.step(
            action)
        action = env.valid_actions[action]
        
        score += reward
        done = terminated or truncated
        observation = observation_
        action_sequence.append(env._rescale(action[0]*action[2], range1=(
            0, 8-1), range2=[15, 30]))
        action_sequence_flowrate.append(env._rescale(action[1]*action[2], range1=(
            0, 8-1), range2=[0.3, 5]))

    Scores.append(score)
    action_sequence.pop()
    action_sequence_flowrate.pop()
    print('Score: ', score)
    
    env.close()
    
    total_score+=score
    '''
    totla_score_unexpected+=score_total_unexpected
    t_heating_district+=total_heating_district
    t_electricity_hvac+=total_electricity_hvac
    t_electricity_plant+=total_electricity_plant
    t_heating_coil+=total_heating_coil
    t_CO2_concentration+=total_CO2_concentration
    '''
    
    #-----------------------------------------------------------------------------------------------------------------------------
    #plot for temperatures variations
    #---------------------------------------------------------------------------------------------------------------------------
    
    # Step 1: Read the CSV file
    # Update this to the path of your CSV file
    file_path = 'C:/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus/tests_output/episode-0-0/eplusout.csv'
    data = pd.read_csv(file_path)
    # Add a default year to the 'Date/Time' column
    data['Date/Time'] = '2023/' + data['Date/Time'].str.strip()

    # Remove any extra spaces between the date and time
    data['Date/Time'] = data['Date/Time'].str.replace(r'\s+', ' ', regex=True)

    data['Date/Time'] = data['Date/Time'].str.replace('24:00:00', '23:59:59')
    data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%Y/%m/%d %H:%M:%S')
     
     
    # Step 2: Extract Necessary Data
    datetime = data['Date/Time']
    temp_zone = data['TZ_AMPHITHEATER:Zone Mean Air Temperature [C](TimeStep)']
    htg_threshold = data['HTG HVAC 1 ADJUSTED BY 1.1 F:Schedule Value [](TimeStep)']
    clg_threshold = data['CLG HVAC 1 ADJUSTED BY 0 F:Schedule Value [](TimeStep)']
    outdoor_air_temp = data['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
    AHU_ON_OFF = data['AHUS ONOFF:Schedule Value [](TimeStep)']
    action_temp_set = action_sequence  # Assuming this is the column name for your action temperature set
      
    # New variable for occupancy
    occupancy = data['MAISON DU SAVOIR AUDITORIUM OCC:Schedule Value [](TimeStep)']

    # Step 3: Plot the Data
    fig, ax1 = plt.subplots(figsize=(12, 7))
     
    # Increase font size
    plt.rcParams.update({'font.size': 14})


    # Plotting the Zone Mean Air Temperature
    ax1.plot(datetime, temp_zone, label='Zone Mean Air Temperature', color='blue')

    # Plotting the thresholds
    ax1.plot(datetime, htg_threshold, label='Heating Threshold', color='red', linestyle='--')
    ax1.plot(datetime, clg_threshold, label='Cooling Threshold', color='red', linestyle=':')

    # Plotting the Outdoor Air Temperature in green
    ax1.plot(datetime, outdoor_air_temp, label='Outdoor Air Temperature', color='green', linewidth=2)

    # Plotting the Action Temperature Set in orange
    ax1.step(datetime, action_temp_set, label='Supply Air Temperature Setpoint', color='orange', linewidth=2)

    # Enhancing the plot
    ax1.set_xlabel('Date/Time')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Zone Temperature and HVAC Thresholds')
    ax1.legend(loc='upper left')
    # Create x-axis labels to show every other datetime
    # Automatically format the x-axis to show time first and then the date
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
    # Set x-limits to avoid wasted space
    ax1.set_xlim([datetime.min(), datetime.max()])
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=25,fontsize=12)
    
    # Create a second y-axis to plot the number of occupants
    ax2 = ax1.twinx()
    ax2.step(datetime, occupancy * 399, label='Number of Occupants', color='black', linewidth=2)
    ax2.set_ylabel('Number of Occupants')
    ax2.set_ylim(0, 400)
    
    # Create a third y-axis to plot the AHU_ON_OFF
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis outward
    ax3.step(datetime, AHU_ON_OFF, label='AHU ON/OFF', color='grey', linewidth=1,linestyle='--',zorder=1)
    ax3.set_ylabel('AHU ON/OFF')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])  # Show only the values 0 and 1
    # Add a vertical dashed line at the first occurrence of AHU_ON_OFF being 1
    first_on_time = datetime[AHU_ON_OFF.eq(1)].iloc[0]  # Get the first time AHU_ON_OFF is 1
    adjusted_time = first_on_time + timedelta(seconds=700)  # Adjust by 3 seconds to move the line to the right

    ax3.axvline(x=adjusted_time, color='grey', linestyle='--', ymin=0.085, ymax=0.915, zorder=1,linewidth=1)
    
    # Add a horizontal line from the y-axis to the vertical line
    ax3.hlines(y=1, xmin=datetime.min(), xmax=adjusted_time, color='grey', linestyle='--', linewidth=1, zorder=1)

    # Fourth axis for Outdoor Air Controller’s Air Mass Flow Rate
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    ax4.step(datetime, action_sequence_flowrate, label='Outdoor Air Controller Air Mass Flow Rate', color='skyblue', linewidth=2)
    ax4.set_ylabel('OAC Air Mass Flow Rate')
    
    
    
    # Combine legends from both y-axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2 + lines_3+ lines_4, labels_1 + labels_2 + labels_3+ labels_4, loc='upper left',bbox_to_anchor=(0, 0.9), fontsize=10, framealpha=0)
    
    # Adding score texts
    plt.text(0.01, 0.99, f'Score: {score:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    #plt.text(0.01, 0.95, f'Unexpected Score: {score_total_unexpected:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # Show plot
    plt.tight_layout()
    plt.show()

   
    #-------------------------------------------------------------------------------------------------------------------------------------
    #plot for energy consumption (version 1)
    # Step 2: Extract Necessary Data
    datetime = data['Date/Time']
    electricity_hvac = data['Electricity:HVAC [J](TimeStep)']
    heating_district = data['Heating:DistrictHeatingWater [J](TimeStep)']

    # Step 3: Create Plot with Two Y-Axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plotting Electricity:HVAC [J] on the first Y-axis
    color = 'tab:green'
    ax1.set_xlabel('Date/Time')
    ax1.set_ylabel('Electricity:HVAC [J](TimeStep)', color=color)
    ax1.step(datetime, electricity_hvac,
            label='Electricity:HVAC [J]', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # Create x-axis labels to show every other datetime
    # Automatically format the x-axis to show time first and then the date
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
    # Set x-limits to avoid wasted space
    ax1.set_xlim([datetime.min(), datetime.max()])
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=25,fontsize=12)

    # Creating a second Y-axis for Heating:DistrictHeatingWater [J]
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    # we already handled the x-label with ax1
    ax2.set_ylabel(
       'Heating:DistrictHeatingWater [J](TimeStep)', color=color)
    ax2.step(datetime, heating_district,
            label='Heating:DistrictHeatingWater [J]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Further customizations
    fig.tight_layout()  # to adjust subplot parameters to give specified padding
    plt.title('Energy Consumption over Time with Dual Y-Axis')
    fig.legend(loc="upper left", bbox_to_anchor=(0.05, 0.9),framealpha=0)
   
    # Calculate the total Heating:DistrictHeatingWater consumption
    total_heating_district = heating_district.sum()
    total_electricity_hvac=electricity_hvac.sum()
    # Adding the total consumption to the top left of the plot
    plt.text(0.01, 0.99, f'Total District Heating Water Consumption: {total_heating_district:.2e} J',
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.01, 0.95, f'Total electricity hvac Consumption: {total_electricity_hvac:.2e} J',
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
   
    # Show plot
    plt.show() 
            
    #--------------------------------------------------------------------------------------------------------------------------------
    #plot for energy consumption (version2)
    # Step 2: Extract Necessary Data
    datetime = data['Date/Time']
    electricity_plant = data['Electricity:Plant [J](TimeStep)']
    heating_coil = data['HeatingCoils:EnergyTransfer [J](TimeStep)']

    # Step 3: Create Plot with Two Y-Axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plotting Electricity:HVAC [J] on the first Y-axis
    color = 'tab:green'
    ax1.set_xlabel('Date/Time')
    ax1.set_ylabel('Electricity:Plant [J](TimeStep)', color=color)
    ax1.plot(datetime, electricity_plant,
            label='Electricity:Plant [J](TimeStep)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # Automatically format the x-axis to show time first and then the date
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
    # Set x-limits to avoid wasted space
    ax1.set_xlim([datetime.min(), datetime.max()])
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=25,fontsize=12)

    # Creating a second Y-axis for Heating:DistrictHeatingWater [J]
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    # we already handled the x-label with ax1
    ax2.set_ylabel(
       'HeatingCoils:EnergyTransfer [J](TimeStep)', color=color)
    ax2.plot(datetime, heating_coil,
            label='HeatingCoils:EnergyTransfer [J](TimeStep)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Further customizations
    fig.tight_layout()  # to adjust subplot parameters to give specified padding
    plt.title('Energy Consumption over Time with Dual Y-Axis')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9),framealpha=0)
   
   
    # Calculate the total Electricity:Plant consumption
    total_Electricity_Plant = electricity_plant.sum()
    # Adding the total consumption to the top left of the plot
    plt.text(0.01, 0.99, f'Total Electricity:Plant consumption: {total_Electricity_Plant:.2e} J',
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
   
    # Calculate the total heating coil consumption
    total_heating_coil = heating_coil.sum()
    # Adding the total consumption to the top left of the plot
    plt.text(0.01, 0.95, f'total heating coil consumption: {total_heating_coil:.2e} J',
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
   

    # Show plot
    plt.show()
    #-------------------------------------------------------------------------------------------------
    #plot for CO2 concentration (version2)
    # Step 2: Extract Necessary Data
    datetime = data['Date/Time']
    CO2_concentration = data['TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)']
    # Step 3: Create Plot with Two Y-Axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plotting Electricity:HVAC [J] on the first Y-axis
    color = 'tab:green'
    ax1.set_xlabel('Date/Time')
    ax1.set_ylabel('TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)', color=color)
    ax1.plot(datetime, CO2_concentration,                
            label='TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # Create x-axis labels to show every other datetime
    # Automatically format the x-axis to show time first and then the date
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
    # Set x-limits to avoid wasted space
    ax1.set_xlim([datetime.min(), datetime.max()])
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=25,fontsize=12)

    # Further customizations
    fig.tight_layout()  # to adjust subplot parameters to give specified padding
    plt.title('Zone Air CO2 Concentration')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9),framealpha=0)
   
    total_CO2_concentration=CO2_concentration.sum()
   
    # Show plot
    plt.show()


#%% traditional training (entire year)--------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    file_path = 'C:/Users/kalsayed/rllib-energyplus/rleplus/examples/amphitheater/model.idf'
    new_begin_month=1
    new_end_month=1

    new_begin_day = 1 # The new value for the Begin Day of Month
    new_end_day = 7   # The new value for the End Day of Month

    modify_runperiod_dates(file_path, new_begin_month, new_end_month, new_begin_day, new_end_day)
    env = AmphitheaterEnv({"output": "/tmp/tests_output"},new_begin_month, new_end_month, new_begin_day, new_end_day,True,80,7,1,4)
    env.runner_config.csv = False

    best_score = -np.inf
    load_checkpoint = False
    total_deployment_load_checkpoint = False
    n_episodes = 1

    agent = DQNAgent(
        gamma=0.99,
        epsilon=1,
        lr=0.001,
        input_dims=(env.observation_space.shape),
        n_actions=env.action_space.n,
        mem_size=250000,
        eps_min=0.05,
        batch_size=64,
        replace=384,
        eps_dec=0.9 * n_episodes * 50,
        chkpt_dir='/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus',
        algo='DQNAgent',
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

    for i in range(n_episodes*50):
        Episode += 1
        done = False

        # If the episode just terminated, reset env
        if terminated:
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
            '''
            #normal buffer b----------------------------------------------------------------------
            m=agent.memory.state_memory[:300]
            # next state
            m_next=agent.memory.new_state_memory[:300]
            #action
            m_action=agent.memory.action_memory[:300]
            #reward
            m_reward=agent.memory.reward_memory[:300]
            #done message
            m_done=agent.memory.terminal_memory[:300]
            #Transformer buffer B--------------------------------------------------------------------
            # actual state
            t=agent.memory.transformer_state_memory[:300]
            # next state
            t_next=agent.memory.transformer_next_state_memory[:300]
            #action
            t_action=agent.memory.transformer_action_memory[:300]
            #reward
            t_reward=agent.memory.transformer_reward_memory[:300]
            #done message
            t_done=agent.memory.transformer_done_memory[:300]
            '''

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
    '''
    metrics_file = "C:/Users/kalsayed/Desktop/code_phd_13_8_2024/Agent3/conference_paper/Bi-LSTM/adaptation_test/nodropout/training_metrics.pkl"
    save_training_metrics(metrics_file, scores, eps_history, steps_array)
    '''
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
    
    
    
'''
data_to_save = {
    'x': x,
    'scores': scores,
    'x_means': x_means,
    'means': means,
    'cumulative_avg': cumulative_avg,
    'epsilon' : eps_history
    
}

# Define the file path, including the filename and extension
save_file_path = 'C:/Users/kalsayed/Desktop/code_phd_13_8_2024/Agents/Q-transformer/data_graphe_training/data.pkl'

# Save the dictionary to a pickle file
with open(save_file_path, 'wb') as f:
    pickle.dump(data_to_save, f)
'''   
   
#%% teste avec le l'année complet
#---------------------------------------------------------------------------------------------------------------------------------
#deployment--------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
  

file_path = 'C:/Users/kalsayed/rllib-energyplus/rleplus/examples/amphitheater/model.idf'
new_begin_month=1
new_end_month=1

new_begin_day = 1 # The new value for the Begin Day of Month
new_end_day = 30   # The new value for the End Day of Month
modify_runperiod_dates(file_path, new_begin_month, new_end_month, new_begin_day, new_end_day)

env = AmphitheaterEnv(
    {"output": "/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus/tests_output"}, new_begin_month, new_end_month, new_begin_day, new_end_day,False,80,7,1,4)
env.runner_config.csv = True


agent = DQNAgent(
    gamma=0.99,
    epsilon=0,
    lr=0.001,
    input_dims=(env.observation_space.shape),
    n_actions=env.action_space.n,
    mem_size=400000,
    eps_min=0.05,
    batch_size=128,
    replace=384,
    eps_dec=0.9 * n_episodes * 366,
    chkpt_dir='/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus',
    algo='DQNAgent',
    env_name='energyplus'
)


agent.load_models()

agent.q_eval.eval()

scores = []
action_sequence = []
action_sequence_flowrate = []
# A rolling buffer to hold up to 96 recent states
state_window = deque(maxlen=96)
done = False
observation = env.reset()[0]
score = 0
while not done:
    if len(state_window) < 96:
        action = np.random.choice(agent.action_space)
    else:
        obs_seq = np.array(state_window, dtype=np.float32)
        action = agent.choose_action_test(obs_seq)
    #action = agent.choose_action_deployment(observation)               #eval_network
    #action=0
    observation_, reward, terminated, truncated, _ = env.step(
        action)
    
    
    score += reward
    #score += reward
    done = terminated or truncated
   
    action = env.valid_actions[action]
    observation = observation_
    state_window.append(observation)
    action_sequence.append(env._rescale(action[0]*action[2], range1=(
        0, 8-1), range2=[15, 30]))
    action_sequence_flowrate.append(env._rescale(action[1]*action[2], range1=(
        0, 8-1), range2=[0.3, 5]))
    scores.append(reward)
    #normal buffer b----------------------------------------------------------------------
    # actual state
    '''
    m=agent.memory.state_memory[:300]
    # next state
    m_next=agent.memory.new_state_memory[:300]
    #action
    m_action=agent.memory.action_memory[:300]
    #reward
    m_reward=agent.memory.reward_memory[:300]
    #done message
    m_done=agent.memory.terminal_memory[:300]
    #Transformer buffer B--------------------------------------------------------------------
    # actual state
    t=agent.memory.transformer_state_memory[:300]
    # next state
    t_next=agent.memory.transformer_next_state_memory[:300]
    #action
    t_action=agent.memory.transformer_action_memory[:300]
    #reward
    t_reward=agent.memory.transformer_reward_memory[:300]
    #done message
    t_done=agent.memory.transformer_done_memory[:300]
    '''

#scores.append(score)
print('Score: ', score)
env.close()    

action_sequence.insert(0, 0)
action_sequence_flowrate.insert(0, 0)

#---------------------------------------------------------------------------------------------------------------------------------
#deployment--------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
Heating_Distict=0
Electricity_Hvac=0
Electricity_Plant=0
Heating_Coil=0
CO2_Concentration=0
  
first_timstep=-96

last_timestep=0


for i in range(0,30):
    
    first_timstep=first_timstep+96
    last_timestep=last_timestep+96

    
    score = float(pd.DataFrame(scores).head(last_timestep).tail(last_timestep-first_timstep).sum() )   
    #-----------------------------------------------------------------------------------------------------------------------------
    #plot for temperatures variations
    # Step 1: Read the CSV file
    # Update this to the path of your CSV file
    file_path = 'C:/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus/tests_output/episode-0-0/eplusout.csv'
    data = pd.read_csv(file_path)
    # Add a default year to the 'Date/Time' column
    data['Date/Time'] = '2020/' + data['Date/Time'].str.strip()

    # Remove any extra spaces between the date and time
    data['Date/Time'] = data['Date/Time'].str.replace(r'\s+', ' ', regex=True)

    data['Date/Time'] = data['Date/Time'].str.replace('24:00:00', '23:59:59')
    data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%Y/%m/%d %H:%M:%S')

    # Step 2: Extract Necessary Data
    datetime = data['Date/Time'].head(last_timestep).tail(last_timestep-first_timstep)
    temp_zone = data['TZ_AMPHITHEATER:Zone Mean Air Temperature [C](TimeStep)'].head(last_timestep).tail(last_timestep-first_timstep)
    htg_threshold = data['HTG HVAC 1 ADJUSTED BY 1.1 F:Schedule Value [](TimeStep)'].head(last_timestep).tail(last_timestep-first_timstep)
    clg_threshold = data['CLG HVAC 1 ADJUSTED BY 0 F:Schedule Value [](TimeStep)'].head(last_timestep).tail(last_timestep-first_timstep)
    # New variable
    outdoor_air_temp = data[
        'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].head(last_timestep).tail(last_timestep-first_timstep)
    AHU_ON_OFF = data['AHUS ONOFF:Schedule Value [](TimeStep)'].head(last_timestep).tail(last_timestep-first_timstep)
    
    action_temp_set = pd.DataFrame(action_sequence).head(last_timestep).tail(last_timestep-first_timstep)
    action_flowrate_set = pd.DataFrame(action_sequence_flowrate).head(last_timestep).tail(last_timestep-first_timstep)
  
    # New variable for occupancy
    occupancy = data['MAISON DU SAVOIR AUDITORIUM OCC:Schedule Value [](TimeStep)'].head(last_timestep).tail(last_timestep-first_timstep)

    # Step 3: Plot the Data
    fig, ax1 = plt.subplots(figsize=(12, 7))
     
    # Increase font size
    plt.rcParams.update({'font.size': 14})


    # Plotting the Zone Mean Air Temperature
    ax1.plot(datetime, temp_zone, label='Zone Mean Air Temperature', color='blue')

    # Plotting the thresholds
    ax1.plot(datetime, htg_threshold, label='Heating Threshold', color='red', linestyle='--')
    ax1.plot(datetime, clg_threshold, label='Cooling Threshold', color='red', linestyle=':')

    # Plotting the Outdoor Air Temperature in green
    ax1.plot(datetime, outdoor_air_temp, label='Outdoor Air Temperature', color='green', linewidth=2)

    # Plotting the Action Temperature Set in orange
    ax1.step(datetime, action_temp_set, label='Supply Air Temperature Setpoint', color='orange', linewidth=2)

    # Enhancing the plot
    ax1.set_xlabel('Date/Time')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Zone Temperature and HVAC Thresholds')
    ax1.legend(loc='upper left')
    # Create x-axis labels to show every other datetime
    # Automatically format the x-axis to show time first and then the date
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
    # Set x-limits to avoid wasted space
    ax1.set_xlim([datetime.min(), datetime.max()])
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=25,fontsize=12)
    
    # Create a second y-axis to plot the number of occupants
    ax2 = ax1.twinx()
    ax2.step(datetime, occupancy * 399, label='Number of Occupants', color='black', linewidth=2)
    ax2.set_ylabel('Number of Occupants')
    ax2.set_ylim(0, 400)
    
    # Create a third y-axis to plot the AHU_ON_OFF
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis outward
    ax3.step(datetime, AHU_ON_OFF, label='AHU ON/OFF', color='grey', linewidth=1,linestyle='--',zorder=1)
    ax3.set_ylabel('AHU ON/OFF')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])  # Show only the values 0 and 1
    # Add a vertical dashed line at the first occurrence of AHU_ON_OFF being 1
    #first_on_time = datetime[AHU_ON_OFF.eq(1)].iloc[0]  # Get the first time AHU_ON_OFF is 1
    #adjusted_time = first_on_time + timedelta(seconds=700)  # Adjust by 3 seconds to move the line to the right

    #ax3.axvline(x=adjusted_time, color='grey', linestyle='--', ymin=0.085, ymax=0.915, zorder=1,linewidth=1)
    
    # Add a horizontal line from the y-axis to the vertical line
    #ax3.hlines(y=1, xmin=datetime.min(), xmax=adjusted_time, color='grey', linestyle='--', linewidth=1, zorder=1)
    # Fourth axis for Outdoor Air Controller’s Air Mass Flow Rate
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    ax4.step(datetime, action_flowrate_set, label='Outdoor Air Controller Air Mass Flow Rate', color='skyblue', linewidth=2)
    ax4.set_ylabel('OAC Air Mass Flow Rate')
    
    # Combine legends from both y-axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2 + lines_3+ lines_4, labels_1 + labels_2 + labels_3+ labels_4, loc='upper left',bbox_to_anchor=(0, 0.9), fontsize=10, framealpha=0)
    
    # Adding score texts
    plt.text(0.01, 0.99, f'Score: {score:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    #plt.text(0.01, 0.95, f'Unexpected Score: {score_total_unexpected:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # Show plot
    plt.tight_layout()
    plt.show()
    
    
    #-------------------------------------------------------------------------------------------------------------------------------------
    #plot for energy consumption (version 1)
    # Step 2: Extract Necessary Data
    
    electricity_hvac = data['Electricity:HVAC [J](TimeStep)'].head(last_timestep).tail(last_timestep-first_timstep)
    heating_district = data['Heating:DistrictHeatingWater [J](TimeStep)'].head(last_timestep).tail(last_timestep-first_timstep)

    # Step 3: Create Plot with Two Y-Axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plotting Electricity:HVAC [J] on the first Y-axis
    color = 'tab:green'
    ax1.set_xlabel('Date/Time')
    ax1.set_ylabel('Electricity:HVAC [J](TimeStep)', color=color)
    ax1.step(datetime, electricity_hvac,
            label='Electricity:HVAC [J]', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # Create x-axis labels to show every other datetime
    # Automatically format the x-axis to show time first and then the date
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
    # Set x-limits to avoid wasted space
    ax1.set_xlim([datetime.min(), datetime.max()])
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=25,fontsize=12)

    # Creating a second Y-axis for Heating:DistrictHeatingWater [J]
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    # we already handled the x-label with ax1
    ax2.set_ylabel(
       'Heating:DistrictHeatingWater [J](TimeStep)', color=color)
    ax2.step(datetime, heating_district,
            label='Heating:DistrictHeatingWater [J]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Further customizations
    fig.tight_layout()  # to adjust subplot parameters to give specified padding
    plt.title('Energy Consumption over Time with Dual Y-Axis')
    fig.legend(loc="upper left", bbox_to_anchor=(0.05, 0.9),framealpha=0)

    # Calculate the total Heating:DistrictHeatingWater consumption
    total_heating_district = heating_district.sum()
    total_electricity_hvac=electricity_hvac.sum()
    # Adding the total consumption to the top left of the plot
    plt.text(0.01, 0.99, f'Total District Heating Water Consumption: {total_heating_district:.2e} J',
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.01, 0.95, f'Total electricity hvac Consumption: {total_electricity_hvac:.2e} J',
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # Show plot
    plt.show() 
    Heating_Distict+=total_heating_district
    Electricity_Hvac+=total_electricity_hvac
    #--------------------------------------------------------------------------------------------------------------------------------
    #plot for energy consumption (version2)
    # Step 2: Extract Necessary Data
  
    electricity_plant = data['Electricity:Plant [J](TimeStep)'].head(last_timestep).tail(last_timestep-first_timstep)
    heating_coil = data['HeatingCoils:EnergyTransfer [J](TimeStep)'].head(last_timestep).tail(last_timestep-first_timstep)

    # Step 3: Create Plot with Two Y-Axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plotting Electricity:HVAC [J] on the first Y-axis
    color = 'tab:green'
    ax1.set_xlabel('Date/Time')
    ax1.set_ylabel('Electricity:Plant [J](TimeStep)', color=color)
    ax1.plot(datetime, electricity_plant,
            label='Electricity:Plant [J](TimeStep)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # Automatically format the x-axis to show time first and then the date
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
    # Set x-limits to avoid wasted space
    ax1.set_xlim([datetime.min(), datetime.max()])
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=25,fontsize=12)

    # Creating a second Y-axis for Heating:DistrictHeatingWater [J]
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    # we already handled the x-label with ax1
    ax2.set_ylabel(
       'HeatingCoils:EnergyTransfer [J](TimeStep)', color=color)
    ax2.plot(datetime, heating_coil,
            label='HeatingCoils:EnergyTransfer [J](TimeStep)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Further customizations
    fig.tight_layout()  # to adjust subplot parameters to give specified padding
    plt.title('Energy Consumption over Time with Dual Y-Axis')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9),framealpha=0)


    # Calculate the total Electricity:Plant consumption
    total_Electricity_Plant = electricity_plant.sum()
    # Adding the total consumption to the top left of the plot
    plt.text(0.01, 0.99, f'Total Electricity:Plant consumption: {total_Electricity_Plant:.2e} J',
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # Calculate the total heating coil consumption
    total_heating_coil = heating_coil.sum()
    # Adding the total consumption to the top left of the plot
    plt.text(0.01, 0.95, f'total heating coil consumption: {total_heating_coil:.2e} J',
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    
    # Show plot
    plt.show()
    Electricity_Plant+=total_Electricity_Plant
    Heating_Coil+=total_heating_coil
    #-------------------------------------------------------------------------------------------------
    #plot for CO2 concentration (version2)
    # Step 2: Extract Necessary Data
    
    CO2_concentration = data['TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)'].head(last_timestep).tail(last_timestep-first_timstep)
    # Step 3: Create Plot with Two Y-Axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plotting Electricity:HVAC [J] on the first Y-axis
    color = 'tab:green'
    ax1.set_xlabel('Date/Time')
    ax1.set_ylabel('TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)', color=color)
    ax1.plot(datetime, CO2_concentration,                
            label='TZ_AMPHITHEATER:Zone Air CO2 Concentration [ppm](TimeStep)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # Create x-axis labels to show every other datetime
    # Automatically format the x-axis to show time first and then the date
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S %m-%d'))
    # Set x-limits to avoid wasted space
    ax1.set_xlim([datetime.min(), datetime.max()])
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=25,fontsize=12)

    # Further customizations
    fig.tight_layout()  # to adjust subplot parameters to give specified padding
    plt.title('Zone Air CO2 Concentration')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9),framealpha=0)

    total_CO2_concentration=CO2_concentration.sum()

    # Show plot
    plt.show()
    CO2_Concentration+=total_CO2_concentration

#%%
# Define number of episodes (days)
n_episodes = 366

# Create the environment
file_path = 'C:/Users/kalsayed/rllib-energyplus/rleplus/examples/amphitheater/model.idf'
new_begin_month=1
new_end_month=12

new_begin_day = 1 # The new value for the Begin Day of Month
new_end_day = 31   # The new value for the End Day of Month
modify_runperiod_dates(file_path, new_begin_month, new_end_month, new_begin_day, new_end_day)

env = AmphitheaterEnv(
    {"output": "/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus/tests_output"}, new_begin_month, new_end_month, new_begin_day, new_end_day,False,80,7,1,4)
env.runner_config.csv = True

# Create and load your DQN agent
agent = DQNAgent(
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
    chkpt_dir='/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus',
    algo='DQNAgent',
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
    )
    total_score += chunk_score

    # 2) Read CSV and slice the data for this chunk
    csv_path = 'C:/Users/kalsayed/Desktop/TD3_hafchetah_code/DQN/energyplus/tests_output/episode-0-0/eplusout.csv'
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