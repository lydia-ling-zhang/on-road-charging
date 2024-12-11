import numpy as np


M = 10000
N = M # as long as N is >=2, then I think this is okay

# def take_orders_constr(rt,ct,a):
   

#     indicator = []
#     for z in [0,1]:
#         if rt+ct+a-1<=M*(1-z) and z<=1-a and rt+ct+N*a-1>=1-M*z:
#             indicator.append(z)

#     return indicator

def take_orders_constr(rt,ct,a):
   

    indicator = []
    for z in [0,1]:
        if rt+a-1<=M*(1-z) and z<=1-a and rt+M*a-1>=1-M*z:
            indicator.append(z)

    return indicator




def start_charge_constr(ct, a):
    
    lb = a-ct
    ub = np.minimum(a, 1-ct)

    indicator = []
    for z in [0,1]:
        if z <= ub and z >= lb:
            indicator.append(z)

    return indicator


def continue_charge_constr(ct, a):
    lb = a + ct -1
    ub = np.minimum(a, ct)

    indicator = []
    for z in [0,1]:
        if z <= ub and z >= lb:
            indicator.append(z)

    return indicator

def valid_SoC(SoC, low_SoC, epsilon=0.0001):
    indicator = []
    for z in [0,1]:
        lb = low_SoC - M * (1 - z) 
        ub = low_SoC + M * z - epsilon
        # ub = low_SoC + M * z
        # ub = low_SoC + M * z - 1
        
        if SoC>=lb and SoC<=ub:
            indicator.append(z)
            
    return indicator
            

values = [[4, 0, 0], # on a ride
          [1, 0, 0], # about to finish ride, start taking orders
          [1, 0, 1], # about to finish ride, start charging
          [0, 0, 0], # about to finish idle, start taking orders
          [0, 0, 1], # about to finish idle, start charging
          [0, 1, 0], # about to finish charging, unplug, start taking orders
          [0, 1, 1] # about to finish charging, continue charging
          ]
keys = ["on a ride", "finish ride, start taking orders", 
        "finish ride, start charging",
        "start taking orders",
        "start charging",
        "unplug, start taking orders",
        "continue charging"]


# print("First time charge indicator:")
# print("Continue charging indicator:")
# print("Start taking orders indicator:")
for i in range(len(values)):
    rt, ct, a = values[i]

    
    # print(f"Cases{i}: {keys[i]}  indicator: {start_charge_constr(ct, a)}\n")
    
    # print(f"Cases{i}: {keys[i]}  indicator: {continue_charge_constr(ct, a)}\n")
    
    # print(f"Cases{i}: {keys[i]}  indicator: {take_orders_constr(rt, ct, a)}\n")
    # print(f"Indicator variable: {start_charge_constr(ct, a)}")
    # print(f"\n")
    # print(f"Indicator variable: {continue_charge_constr(ct, a)}")
    # print(f"Indicator variable: {take_orders_constr1(rt,ct,a)}")
    # print(f"\n")
    # print(f"Indicator variable: {take_orders_constr2(rt,ct,a)}")
    # print(f"\n")
    # print(f"Indicator variable: {take_orders_constr(rt,ct,a)}")
    # print(f"\n")
    

SoC_values = [[0.1, 0.1],
              [0.11, 0.1],
              [0.111, 0.1],
          [0.5, 0.1],
          [0.09, 0.1],
          [0.099, 0.1],
          [0.05, 0.1]]
for i in range(len(SoC_values)):
    row = SoC_values[i]
    print(f"SoC is {row[0]}, low_SoC is {row[1]}: {valid_SoC(row[0], row[1])}")
