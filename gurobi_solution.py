from gurobipy import Model, GRB
import numpy as np
import json
import pandas as pd
import csv

# Initialize model
model = Model("MultiAgentMDP")

M = 1000        # A large constant for Big-M formulation

# read config file
with open("/home/songlei/zhangling/on-road-charging/config_cases/"+"case_1.json", "r") as file:
    config = json.load(file)
    

n_agents = config['fleet_size']  # Number of EVs in the fleet
n_chargers = config['n_chargers']  # Total number of chargers

T = config["max_time_steps"] 
    
max_cap = config['max_cap']  # Max battery capacity (kWh)
h = config['connection_fee']  # $ per connection session
d_rate = config['d_rate']  # Battery consumption rate per time step
c_rate = config['c_rate']   # Charger speed per time step
c_r = config['c_r']
initial_SoC = config["initial_SoC"]
low_SoC = config["low_SoC"]
rt_samples = config["rt_samples"]
w = config["w"]
p = config["p"]

print(f"rt_samples has length {len(rt_samples[0])}")
for i in range(n_agents):
  rt_samples[i] = [rt+1 for rt in rt_samples[i]]

print(f"initial_SoC has length {len(initial_SoC)}")
print(f"ride_time_sample has length {len(rt_samples[0])}")
print(f"w has length {len(w)}")
print(f"p has length {len(p)}")


print(f"SoC increases {c_rate} for each step")
print(f"SoC drop {d_rate} for each step")
print(f"charge {c_r} kWh for each step")
print(f"connection fee is {h} $")
print(f"order price is {w[0]} $")
print(f"charging price is {p[0]} $")


# Variables
a = model.addVars(n_agents, T, vtype=GRB.BINARY, name="a")  # Action
z_order = model.addVars(n_agents, T, vtype=GRB.BINARY, name="take_order")  # Indicator
z_connect = model.addVars(n_agents, T, vtype=GRB.BINARY, name="start_charge")  # Indicator
z_charge = model.addVars(n_agents, T, vtype=GRB.BINARY, name="cont_charge")  # Indicator
z_ride = model.addVars(n_agents, T, vtype=GRB.BINARY, name="on_ride")  # On a ride
rt = model.addVars(n_agents, T+1, vtype=GRB.INTEGER, name="rt")  # Ride time
ct = model.addVars(n_agents, T+1, vtype=GRB.BINARY, name="ct")  # Charging time
SoC = model.addVars(n_agents, T+1, vtype=GRB.CONTINUOUS, name="SoC")  # State of Charge
assigned_rt = model.addVars(n_agents, T, vtype=GRB.INTEGER, name="rt")  # Ride time

z_okSoC = model.addVars(n_agents, T, vtype=GRB.BINARY, name="ok_SoC")  # On a ride
z_okRange = model.addVars(n_agents, T, vtype=GRB.BINARY, name="z_okRange")  # On a ride



epsilon = 0.0001

# Initialize values for t=0 for all agents
for i in range(n_agents):
    
    model.addConstr(rt[i, 0] == 0, "Initial rt for agent %s at t=0" % i)
    model.addConstr(ct[i, 0] == 0, "Initial ct for agent %s at t=0" % i)
    model.addConstr(SoC[i, 0] == initial_SoC[i], "Initial SoC for agent %s at t=0" % i)
    # model.addConstr(assigned_rt[i, 0] == 0, "Initial assigned rt for agent %s at t=0" % i)

# Constraints
objective = 0
for i in range(n_agents):
    for t in range(T): # 0, 1, ..., T-1
      
        # Transition dynamics for 'rt'
        model.addGenConstrIndicator(z_ride[i, t], True, rt[i, t+1] == rt[i, t] - 1)
        model.addGenConstrIndicator(z_order[i, t], True, rt[i, t+1] == assigned_rt[i, t])
        model.addGenConstrIndicator(z_connect[i, t], True, rt[i, t+1] ==0)
        model.addGenConstrIndicator(z_charge[i, t], True, rt[i, t+1] == 0)
    
        model.addGenConstrIndicator(z_order[i, t], True, assigned_rt[i, t] <= rt_samples[i][t])
        model.addGenConstrIndicator(z_order[i, t], True, assigned_rt[i, t] <= SoC[i, t]/d_rate)
        model.addGenConstrIndicator(z_order[i, t], True, assigned_rt[i, t] <= z_okSoC[i,t]*M)
        # model.addGenConstrIndicator(z_okSoC[i,t], True, assigned_rt[i, t] <= SoC[i, t]/d_rate)
        # model.addGenConstrIndicator(z_okSoC[i,t], True, assigned_rt[i, t] == rt_samples[i][t])
        # model.addGenConstrIndicator(z_okSoC[i,t], True, assigned_rt[i, t] <= SoC[i, t]/d_rate)
        model.addGenConstrIndicator(z_okSoC[i,t], False, assigned_rt[i, t] == 0)
        # model.addGenConstrIndicator(z_okSoC[i,t], True, assigned_rt[i, t] == rt_samples[i][t]*z_okRange[i,t])
        # model.addGenConstrIndicator(z_order[i, t], True, assigned_rt[i, t] <= rt_samples[i][t] + M * (1 - z_okSoC[i, t]))
        # model.addGenConstrIndicator(z_order[i, t], True, assigned_rt[i, t] >= rt_samples[i][t] - M * (1 - z_okSoC[i, t]))
        # model.addGenConstrIndicator(z_order[i, t], True, assigned_rt[i, t] <= M * z_okSoC[i, t])
        # model.addGenConstrIndicator(z_order[i, t], True, assigned_rt[i, t] >= 0)
        
        # model.addConstr(SoC[i, t]/d_rate >= rt_samples[i][t] - M * (1 - z_okRange[i,t]) )
        # model.addConstr(SoC[i, t]/d_rate <= rt_samples[i][t] + M * z_okRange[i,t] - epsilon)
        
        model.addConstr(SoC[i, t] >= low_SoC - M * (1 - z_okSoC[i,t]) )
        model.addConstr(SoC[i, t] <= low_SoC + M * z_okSoC[i,t] - epsilon)
        
        model.addConstr(SoC[i, t+1] <= 1.0)
        model.addConstr(SoC[i, t+1] >= 0.)
        
        model.addConstr(rt[i, t+1] >=0)
        model.addConstr(assigned_rt[i, t] >=0)

        # Transition dynamics for 'ct'
        model.addConstr(ct[i, t+1] == a[i, t])

        # SoC transition constraint
        model.addConstr(SoC[i, t+1] == SoC[i, t] + ct[i, t] * c_rate - (1 - ct[i, t]) * d_rate)

        # Constraints for indicator variables
        model.addConstr(z_ride[i, t] == 1 - z_order[i, t] - z_connect[i, t] - z_charge[i, t])
        model.addConstr(z_order[i, t] + z_connect[i, t] + z_charge[i, t] <= 1)

        # Constraints for z_connect (first-time connection to a charger)
        model.addConstr(z_connect[i, t] <= a[i, t])
        model.addConstr(z_connect[i, t] <= 1 - ct[i, t])
        model.addConstr(z_connect[i, t] >= a[i, t] - ct[i, t])

        # Constraints for z_charge (continue charging)
        model.addConstr(z_charge[i, t] <= a[i, t])
        model.addConstr(z_charge[i, t] <= ct[i, t])
        model.addConstr(z_charge[i, t] >= a[i, t] + ct[i, t] - 1)

        # Big-M constraints for z_order 
        model.addConstr(rt[i, t] + a[i, t] - 1 <= M * (1 - z_order[i, t]))
        model.addConstr(rt[i, t] + M * a[i, t] - 1 >= 1 - M * z_order[i, t])
        model.addConstr(z_order[i, t] <= 1 - a[i, t])

        objective += w[t] * assigned_rt[i, t] * z_order[i, t] -\
              (h + p[t] * c_r) * z_connect[i, t] -\
                p[t] * c_r * z_charge[i, t]

model.setObjective(objective, GRB.MAXIMIZE)

# Optimize
model.optimize()

# After model.optimize()
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    
    # print("Variable values:")
    # for var in model.getVars():
    #     print(f"{var.varName}: {var.X}")

    print("\nDetailed outputs for specific variables:")
    for i in [0]:
        for t in range(T):
          print(f"rt_samples[{i},{t}]: {rt_samples[i][t]}")
          print(f"assigned_rt[{i},{t}]: {assigned_rt[i, t].X}")
          print(f"z_order[{i},{t}]: {z_order[i, t].X}")
          print(f"z_ride[{i},{t}]: {z_ride[i, t].X}")
          print(f"z_connect[{i},{t}]: {z_connect[i, t].X}")
          print(f"z_charge[{i},{t}]: {z_charge[i, t].X}")
          print(f"rt[{i},{t+1}]: {rt[i, t+1].X}")
          print(f"z_okSoC[{i},{t}]: {z_okSoC[i, t].X}")
          print(f"SoC[{i},{t}]: {SoC[i, t].X}")
          print("\n")
            # Add other variables as needed
else:
    print("No optimal solution found. Status:", model.status)

# rt_values = [[] for _ in range(n_agents)]
# ct_values = [[] for _ in range(n_agents)]
# SoC_values = [[] for _ in range(n_agents)]
# assigned_rt_values = [[] for _ in range(n_agents)]
# a_values = [[] for _ in range(n_agents)]
# for i in range(n_agents):
#   for t in range(T+1):
#       rt_values[i].append(rt[i, t].X)
#       ct_values[i].append(ct[i, t].X)
#       SoC_values[i].append(SoC[i, t].X)

# for i in range(n_agents):
#   for t in range(T):
#       a_values[i].append(a[i, t].X)
#       assigned_rt_values[i].append(a[i, t].X)


