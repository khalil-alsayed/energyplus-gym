# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 07:19:51 2026

@author: khali
"""

#%% library part

# Standard library
import os
import pickle
from pathlib import Path

# Third-party
import numpy as np
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


    





#%% Adaptation test (DDQN_MLP)--------------------------------------------------------------------------------------------------------------------------



if __name__ == '__main__':
    
    
    
    env = AmphitheaterEnv(
        env_config={"output": str(output_dir),"location": "luxembourg"}, new_begin_month=1, new_end_month=1, new_begin_day=1, new_end_day=7,
        train=True)
    env.runner_config.csv = False

    best_score = -np.inf
    load_checkpoint = False
    total_deployment_load_checkpoint = False
    n_episodes = 50
    


    agent = DDQN_MLP(
        gamma=0.99,
        epsilon=1,
        lr=0.001,
        input_dims=(env.observation_space.shape),
        n_actions=env.action_space.n,
        mem_size=250000,
        eps_min=0.05,
        batch_size=64,
        replace=384,
        eps_dec=0.9 * n_episodes,
        chkpt_dir=checkpoint_dir,
        algo='DDQN_MLP',
        env_name='energyplus'
    )

    agent.q_eval.train()
    # if load_checkpoint:
    #     agent.load_models()

    n_steps = 0
    Episode = 0
    scores, eps_history, steps_array = [], [], []
    terminated = True
    
  

    for i in range(n_episodes):
        Episode += 1
        done = False

        # If the episode just terminated, reset env
        
        # if terminated:  # in case of generalisation test
        if terminated :
            env.close()
            observation = env.reset()[0]  # or env.reset()
            #state_window.clear()

        score = 0

        while not done:
            # If we don't yet have 96 states, fallback to random action (or old approach)
            
            action = agent.choose_action_test(observation)

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

