import numpy as np
import gym
import random
import time
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import pandas as pd
import json
from gym import Env, spaces
from scipy.stats import lognorm
from road_charging2 import RoadCharging, ConstrainAction


def display_policy(fleet_size, max_time_steps, trajectories, save_path):

    for i in range(fleet_size):
        print(f"Showing trajectory of agent {i} ......")
        
        # Extract agent's trajectory data
        ride_times = trajectories['states'][i, 0, 1:]
        fractions_of_cap = trajectories['states'][i, 2, 1:]  # range [0, 1]
        actions = trajectories['actions'][i, :]

        # Create a figure with two subplots, share the x-axis
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))

        # --- First plot: Ride times ---
        ax1.step(range(max_time_steps), ride_times[:max_time_steps], color='royalblue', linestyle='-', linewidth=2, label='Ride Time')
        ax1.set_ylabel('Remaining Ride Time Steps', color='black', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer ticks
        ax1.set_title(f"Agent {i} - Ride Time and State of Charge", fontsize=14, pad=20)
        ax1.legend(loc="upper right", fontsize=10)
        
        # --- Second plot: State of charge ---
        ax2.step(range(max_time_steps), fractions_of_cap[:max_time_steps], color='darkgreen', linestyle='-.', linewidth=2, label='State of Charge')
        ax2.set_ylabel('State of Charge', color='black', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.legend(loc="upper left", fontsize=10)
        
        # Create a secondary y-axis for actions
        ax2_secondary = ax2.twinx()
        ax2_secondary.step(range(max_time_steps), actions[:max_time_steps], color='firebrick', linestyle='-', linewidth=2, label='Actions')
        ax2_secondary.set_ylabel('Actions', color='firebrick', fontsize=12)
        ax2_secondary.tick_params(axis='y', labelcolor='firebrick')
        ax2_secondary.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer ticks
        ax2_secondary.legend(loc="upper right", fontsize=10)

        # Common x-axis labeling
        ax2.set_xlabel('Time Step', fontsize=12)

        # Adjust layout to avoid overlapping elements
        plt.tight_layout()

        # Save the plot to a file
        filename = f"{save_path}agent_{i}.png"  # Example: "figures/agent_trajectory_agent_0.png"
        plt.savefig(filename)
        print(f"Saved figure for agent {i} to {filename}")
        
        # Show the plot
        # plt.show()
        plt.close()


def base_policy(env, state):
    alpha = state["RideTime"]
    beta = state["ChargingStatus"]
    theta = state["SoC"]

    action = np.zeros(env.n, dtype=int)
    for i in range(env.n):
        if state["RideTime"][i] >= 1: # if on a ride, not charge
            action[i] = 0
        elif state["SoC"][i] > 1-env.charger_speed[i]: # if full capacity, not charge
            action[i] = 0
        elif state["SoC"][i] <= env.low_battery: # if low capacity has to charge
            action[i] = 1

        total_charging_requests = sum(1 for a, s in zip(action, state["ChargingStatus"]) if s == 0 and a == 1)
        total_continue_charging = sum(1 for a, s in zip(action, state["ChargingStatus"]) if s == 1 and a == 1)
        # released_charger = sum(1 for a, s in zip(action, state["ChargingStatus"]) if s == 1 and a == 0)

        if total_charging_requests + total_continue_charging > env.m: # limit charging requests to available charging capacity
            print('Exceed charger capacity!')
            # charging_requests = sum(action)
            # available_capacity = env.m - sum(state["ChargingStatus"])
            continue_agents = [i for i, (a, s) in enumerate(zip(action, state["ChargingStatus"])) if s == 1 and a == 1]
            requesting_agents = [i for i, (a, s) in enumerate(zip(action, state["ChargingStatus"])) if s == 0 and a == 1]

            available_capacity = env.m - total_continue_charging

            if available_capacity <= 0:
                print('No charger available now.')
                # flip all
                to_flip = requesting_agents
                action[to_flip] = 0

            elif available_capacity > 0:

                if np.any(action == 1):
                    # Scheme #1:
                    # Randomly select from the set of agents requesting charging and set their charging actions to 0
                    to_flip = random.sample(requesting_agents, total_charging_requests-available_capacity)
                    # Scheme #2:
                    # sort charging agents based on their SoC from low to high
                    # battery_level = dict()
                    # for i in charging_agents:
                    #     battery_level[i] = env.obs['SoC'][i]

                    # sorted_battery_level = dict(sorted(battery_level.items(), key=lambda item: item[1]))
                    # print('sorted_battery_level:', sorted_battery_level)
                    # to_flip = list(sorted_battery_level.keys())[env.m:]

                    print('Agents requesting charging:', requesting_agents)
                    print('Flip agents:', to_flip)

                    action[to_flip] = 0

    return action



def main():

    # data_file = '/content/gdrive/MyDrive/RoadCharge/data/'
    data_file = "C:\\Users\\zhangling\\OneDrive - Microsoft\\3 Research projects\\2024EV\\codes\\csv_data\\"
    env = RoadCharging(data_file+"case_1.json")

    env.seed(42)

    # Number of agents, states, and actions
    # n_steps = env.k
    n_agents = env.n
    n_states = 3  # 3 possible states per agent
    n_actions = 2  # 2 action options per agent

    print(f"Number of agents {env.n}")
    print(f"Number of time steps {env.k}")

    # Training loop
    n_episodes = 1
    ep_return = []
    for episode in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
        # for _ in range(20):
            # Each agent selects an action using your policy
            action = base_policy(env, state)
            # code goes here

            # Perform joint actions in the environment
            next_state, rewards, done, _ = env.step(action)


            print(f"return up to now is {env.ep_return}")
            ep_return.append(env.ep_return)

            state = next_state

    display_policy(env.n, env.k, env.trajectories, env.save_path)

    print("data path:", env.config["data_fpath"])
    print("save path:", env.config["save_path"])
    # env.summarize_env()

    # plt.plot(ep_return)
    # plt.show()
    # time.sleep(0.02) 
    # plt.close()
    # plt.plot(env.charging_price)
    # plt.title("show charging price")
    # plt.show()

    # plt.plot(env.order_price)
    # plt.title("show order price")
    # plt.show()

    # plt.plot(env.assign_prob)
    # plt.title("show assign prob")
    # plt.show()

    # plt.plot(np.exp(env.RT_mean))
    # plt.title("show mean ride time in minutes")
    # plt.show()

    # plt.plot(np.exp(env.RT_std))
    # plt.title("show std in minutes")
    # plt.show()

    # Example usage:
    # log_means = [3.0, 3.1, 3.2, 3.3]  # Replace with actual log-mean values
    # log_stds = [0.5, 0.6, 0.7, 0.8]  # Replace with actual log-std values
    # plot_ride_time_pdf(hour=1, log_means=env.RT_mean, log_stds=env.RT_std)

    # Combine all results into a dictionary
    results = {
        "number_of_agents": env.n,
        "number_of_steps": env.k,
        "states": env.trajectories["states"].tolist(),
        "rewards": env.trajectories['rewards'],
        "final_return": env.ep_return
    }

    # print(json.dumps(results, indent=4))
    print("Energy consumption rate per time step:", env.charger_speed[0] )
    print("SoC dropped per time step:", env.consume_rate[0] )

    # Write to JSON file
    output_file = "C:\\Users\\zhangling\\OneDrive - Microsoft\\3 Research projects\\2024EV\\codes\\results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)  # Use indent for readability


    # Close the env
    env.close()


if __name__ == "__main__":
    main()
