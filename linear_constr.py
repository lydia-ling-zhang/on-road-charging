import numpy as np


M = 10000
N = M # as long as N is >=2, then I think this is okay

# def take_orders_constr(rt,ct,a):
   

#     feasible_set = []
#     for z in [0,1]:
#         if rt+ct+a-1<=M*(1-z) and z<=1-a and rt+ct+N*a-1>=1-M*z:
#             feasible_set.append(z)

#     return feasible_set

def take_orders_constr(rt,ct,a):
   

    feasible_set = []
    for z in [0,1]:
        if rt+a-1<=M*(1-z) and z<=1-a and rt+M*a-1>=1-M*z:
            feasible_set.append(z)

    return feasible_set




def start_charge_constr(ct, a):
    
    lb = a-ct
    ub = np.minimum(a, 1-ct)

    feasible_set = []
    for z in [0,1]:
        if z <= ub and z >= lb:
            feasible_set.append(z)

    return feasible_set


def continue_charge_constr(ct, a):
    lb = a + ct -1
    ub = np.minimum(a, ct)

    feasible_set = []
    for z in [0,1]:
        if z <= ub and z >= lb:
            feasible_set.append(z)

    return feasible_set


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
print("Continue charging indicator:")
# print("Start taking orders indicator:")
for i in range(len(values)):
    rt, ct, a = values[i]

    
    # print(f"Cases{i}: {keys[i]}  indicator: {start_charge_constr(ct, a)}\n")
    
    print(f"Cases{i}: {keys[i]}  indicator: {continue_charge_constr(ct, a)}\n")
    
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