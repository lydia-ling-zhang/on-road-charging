from gurobipy import Model, GRB

# Initialize model
model = Model("MultiAgentMDP")

# Parameters
n_agents = 2  # Example: number of agents
T = 4         # Time steps (0, 1, 2, 3, 4)
tau = [2, 3, 4, 5]  # Example: tau values for each t
M = 1000        # A large constant for Big-M formulation
c_rate = 0.1    # Charging rate
d_rate = 0.2    # Discharging rate
w = [0.85]*4
h = 1.5
max_cap = 72
c_r = c_rate * max_cap
p = [5.6] * 4


# Variables
a = model.addVars(n_agents, T, vtype=GRB.BINARY, name="a")  # Action
z_order = model.addVars(n_agents, T, vtype=GRB.BINARY, name="take_order")  # Indicator
z_connect = model.addVars(n_agents, T, vtype=GRB.BINARY, name="start_charge")  # Indicator
z_charge = model.addVars(n_agents, T, vtype=GRB.BINARY, name="cont_charge")  # Indicator
z_ride = model.addVars(n_agents, T, vtype=GRB.BINARY, name="on_ride")  # On a ride
rt = model.addVars(n_agents, T+1, vtype=GRB.INTEGER, name="rt")  # Ride time
ct = model.addVars(n_agents, T+1, vtype=GRB.BINARY, name="ct")  # Charging time
SoC = model.addVars(n_agents, T+1, vtype=GRB.CONTINUOUS, name="SoC")  # State of Charge

# Constraints
objective = 0
for i in range(n_agents):
    for t in range(T): # 0, 1, ..., T-1

        # Transition dynamics for 'rt'
        model.addGenConstrIndicator(z_ride[i, t], True, rt[i, t+1] == rt[i, t] - 1)
        model.addGenConstrIndicator(z_order[i, t], True, rt[i, t+1] == tau[t])
        model.addConstr(rt[i, t+1] == 0, "Default case")

        # Transition dynamics for 'ct'
        model.addConstr(ct[i, t+1] == a[i, t+1])

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


        objective += w[t] * tau[t] * z_order[i, t] -\
              (h + p[t] * c_r[i]) * z_connect[i, t] -\
                p[t] * c_r[i] * z_charge[i, t]

model.setObjective(objective, GRB.MAXIMIZE)

# Optimize
model.optimize()
