import json
import pandas as pd
import numpy as np
import os


def config_env(data_path:str, save_path:str):
    config = {
        "fleet_size": 1,
        "n_chargers": 1,
        "time_step_size": 15, # how many minutes in one time step
        "start_minute": 0, # starting from 6:00 to 8:00
        "end_minute": 1440, # how many minutes we are scheduling for
        "max_cap": 72, # EV's battery capacity in kWh
        "connection_fee": 1.5, # fee for connecting to the charging station ($)
        "minute_SoCfrom0to100": 60, # Time in minutes to charge the battery fully
        "minute_SoCfrom100to0": 480, # Time in minutes to fully discharge the battery
        "assign_prob": data_path+"assign_prob.csv", # probability of getting assigned an order if remaining idle
        "trip_fare" : data_path+'trip_fare_24hrs.csv', # unit fare is calculated per minute, multiply by step length to get fare for a step
        "charging_price": data_path+"LMP_24hrs.csv", # $
        "initial_SoC": data_path+"initial_SoC.csv",
        "low_SoC": 0.19,
        "data_fpath": data_path,
        "save_path": save_path,
        "additional": { # provide additional information, for example, if we want to sample different charging prices each round
        "all_chargingPrice_fpath": data_path + '2023LMP_hourly.csv', # File path to access all LMP data for 2024
        "Q1_dates_fpath": data_path+"2023Q1_dates.csv", # list of dates in 1st quarter
        "Q2_dates_fpath": data_path+"2023Q2_dates.csv", # list of dates in 2nd quarter
        "Q3_dates_fpath": data_path+"2023Q3_dates.csv", # list of dates in 3rd quarter
        "Q4_dates_fpath": data_path+"2023Q4_dates.csv" # list of dates in 4th quarter
        }
    }

    return config

    
if __name__ == "__main__":

    data_path = "/home/songlei/zhangling/on-road-charging/csv_data/"
    save_path = "/home/songlei/zhangling/on-road-charging/config_cases/"
    
    config = config_env(data_path, save_path)

    t_0 = int(config["start_minute"]/config["time_step_size"])
    t_T = int(config["end_minute"]/config["time_step_size"])
    max_time_steps = t_T-t_0
    config["max_time_steps"] = max_time_steps

    order_price = pd.read_csv(config["trip_fare"]).iloc[:, 0].tolist()
    charging_price = pd.read_csv(config["charging_price"]).iloc[:, 0].tolist()
    w = np.repeat(order_price, int(60 / config["time_step_size"]))[t_0:t_T] * config["time_step_size"]  # Order price per time step
    p = np.repeat(charging_price, int(60 / config["time_step_size"]))[t_0:t_T]  # Charging price per time step

    config["w"] = w.tolist()
    config["p"] = p.tolist()


    config["d_rate"] = round(1 / (config["minute_SoCfrom100to0"]) * config["time_step_size"], 3)
    config["c_rate"] = round(1 / (config["minute_SoCfrom0to100"]) * config["time_step_size"], 3)
    config["c_r"] = config["c_rate"] * config["max_cap"]
    
    assign_prob = pd.read_csv(config['assign_prob']).iloc[:, 0].tolist()
    rho = np.repeat(assign_prob, int(60 / config["time_step_size"]))[t_0:t_T]
    config["rho"] = rho.tolist()
    
    if config["time_step_size"] == 10:
        ride_time_buckets = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        with open(data_path+"ride_time_pmf_10min.json", "r") as f:
            ride_time_probs = json.load(f)

    elif config["time_step_size"] == 15:
        ride_time_buckets = [15, 30, 45, 60, 75, 90]
        with open(data_path+"ride_time_pmf_15min.json", "r") as f:
            ride_time_probs = json.load(f)

    ride_time_bins = [int(item/config["time_step_size"]) for item in ride_time_buckets]  # Discretized ride times


    assert config["low_SoC"] / config["d_rate"] >= ride_time_bins[-1], (
    f"Low SoC can only support {config['low_SoC'] / config['d_rate']} steps, "
    f"but the maximum ride time bin is {ride_time_bins[-1]}."
    )

    
    
    config["ride_time_bins"] = ride_time_bins
    config["ride_time_probs"] = ride_time_probs
    
    print(f"there are {max_time_steps} steps, starting from hour {t_0} to hour {t_T}")
    print(f"w has length {len(w)}")
    print(f"p has length {len(p)}")
    print(f'SoC increases {config["c_rate"]} for each step')
    print(f'SoC drop {config["d_rate"]} for each step')
    print(f'Charge {config["c_r"]} kWh for each step')
    print(f"order price is {w[0]} $")
    print(f"charging price is {p[0]} $")

    max_cases = 1
    time_steps = list(range(max_time_steps))

    for case in range(max_cases):
        
        initial_SoC = []
        ride_time_samples = []
        
        
        for i in range(config["fleet_size"]):
            
            initial_SoC.append(np.random.uniform(0, 1))
            
            ride_time_trajectory = []
            for step in time_steps:
                if np.random.random() < rho[step]:
                    random_ride_time = random_ride_time = np.random.choice([rt for rt in ride_time_bins if rt > 0],
                                                            p=[prob for prob in ride_time_probs if prob>0] )
                    ride_time_trajectory.append(int(random_ride_time))
                else:
                    ride_time_trajectory.append(0)

            ride_time_samples.append(ride_time_trajectory)

        # print(len(case_samples), len(case_samples[0]))

        config["rt_samples"] = ride_time_samples
        config["initial_SoC"] = initial_SoC

        # Save each case as a JSON file
        with open(save_path + f"/case_{case + 1}.json", "w") as json_file:
            json.dump(config, json_file, indent=4)

    print("JSON configuration file created: case.json")

        
