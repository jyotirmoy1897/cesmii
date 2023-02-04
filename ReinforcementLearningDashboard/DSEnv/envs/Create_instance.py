import pandas as pd
import gym
import numpy as np

version = 1
nb_jobs = 2
nb_operations = 2
# define number of machines and types
# define a array size of number of machines and type is in the array.
nb_machines = 2
nb_machines_types = np.array([2, 2])

machine_info_file = "./instances/Machine_nb_machine_" + \
    str(nb_machines)+"_of_type_"+str(nb_machines_types) + \
    "_version_"+str(version)+".npy"
np.save(machine_info_file, nb_machines_types)

machine_out = np.load(machine_info_file)
print(machine_out)
nb_type_max = np.max(nb_machines_types)

jobs_length = np.zeros(nb_jobs, dtype=np.int64)

max_time_op = 0
sum_op = 0

instance_matrix = np.zeros(
    (nb_jobs, nb_operations), dtype=(np.int64, 2))
job_init_cnt = 0

while job_init_cnt < nb_jobs:

    temp_nb_proc = nb_operations  # create random number of process

    # randomely generate the machine requirements and times
    proc_cnt = 0
    while proc_cnt < temp_nb_proc:
        machine, time = np.random.randint(0, nb_machines), \
            np.random.randint(8, 10)
        print(proc_cnt, machine, time)
        # random machine type, machine processing
        # print(job_init_cnt, proc_cnt)
        instance_matrix[job_init_cnt][proc_cnt] = (
            machine, time)
        max_time_op = max(max_time_op, time)
        jobs_length[job_init_cnt] += time
        sum_op += time

        proc_cnt += 1
    job_init_cnt += 1


instance_file = "./instances/Instance_with_nb_jobs_"+str(nb_jobs)+"nb_operations_"+str(
    nb_operations)+"_nb_machine_"+str(nb_machines)+"_of_type_"+str(nb_machines_types)+"_version_"+str(version)+".npy"

np.save(instance_file, instance_matrix)

out = np.load(instance_file)
print(out)
