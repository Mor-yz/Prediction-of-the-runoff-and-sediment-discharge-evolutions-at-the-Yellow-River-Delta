from fs_data import FSData

# RL

alhpa = 0.1
gamma = 0.9
epsilon = 0.01

# BSO

flip = 5
max_chance = 3
bees_number = 10
maxIterations = 10
locIterations = 10

# Test type

typeOfAlgo = 1
nbr_exec = 1


method = "qbso_simple"
test_param = "rl"
param = "gamma"
val = str(locals()[param])
regressor = "rr"
location = r"D:\Desktop\qbso-fs-master\y1_linear.csv"

instance = FSData(typeOfAlgo, location, nbr_exec, method, test_param, param, val, regressor, alhpa, gamma, epsilon)
instance.run(flip, max_chance, bees_number, maxIterations, locIterations)
