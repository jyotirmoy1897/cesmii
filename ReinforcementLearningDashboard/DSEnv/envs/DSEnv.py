import bisect
import datetime
import random

import pandas as pd

import gym
import numpy as np

import plotly.express as px
from pathlib import Path


class DsEnv(gym.Env):

    def __init__(self, nb_jobs, nb_operations, nb_machines, nb_machines_types, instance_file=None):
        """
        Environment to model the job shop scheduling with multiple parrallel machines
        We have also breakdown probabilty based which increases with dissimilar object

        Reward:
        Green or tradiness : multi objective needs to be weighted for now
         - one way of solving multi objective is using multi agent--different agent for different objective

        Action: number of legal jobs and not doing anything

        State: same as JSS + machine probability state

        """

        # First Episodic

        # Can use env config to intitialize

        # Problem size and intiatization
        # define number of jobs machines has each point of time

        self.nb_jobs = nb_jobs
        self.nb_operations = nb_operations
        # define number of machines and types
        # define a array size of number of machines and type is in the array.
        self.nb_machines = nb_machines
        self.nb_machines_types = nb_machines_types

        self.nb_type_max = np.max(self.nb_machines_types).astype("int")
        # instance matrix with the machines type

        if instance_file is not None:
            self.instance_matrix = instance_file
        else:
            self.instance_matrix = None

        # jobs value
        self.jobs_length = None

        self.max_time_op = 0.0
        self.max_time_jobs = 0
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0

        # each problem variables
        # final solution
        self.solution = None

        # Last time event occured
        self.last_time_step = float('inf')
        # current event
        self.current_time_step = float('inf')

        # next event list
        self.next_time_step = list()
        # next job
        self.next_jobs = list()

        # (a1) A Boolean to represent if the job can be allocated.
        self.legal_actions = None

        # (a5) The required time until the machine needed to perform the next job’s operation is free,
        # scaled by the longest duration of an operation to be in the range [0, 1].
        self.time_until_available_machine = None

        #  (a2) The left-over time for the currently performed operation on the job.
        # This value is scaled by the longest operation in the schedule to be in the range [0,1].
        self.time_until_finish_current_op_jobs = None

        # a3) The percentage of operations finished for a job.
        self.todo_time_step_job = None

        # (a4) The left-over time until total completion of the job, scaled by
        # the longest job total completion time to be in the range [0,1].
        self.total_perform_op_time_jobs = None

        # (a6) The IDLE time since last job’s performed operation,
        # scaled by the sum of durations of all operations to be in the range [0,1).
        self.idle_time_jobs_last_op = None

        # (a7) The cumulative job’s IDLE time in the schedule, scaled as a6 to be in the range [0,1).)
        self.total_idle_time_jobs = None

        # the required machines for each job
        self.needed_machine_jobs = None
        # the state a 7*job np array
        self.state = None

        # just to variables to handle no op
        # mark machine and job pair as illegal is no op
        self.illegal_actions = None
        # mark a job illegal
        self.action_illegal_no_op = None

        # mark usuable machines
        self.machine_legal = None

        self.start_timestamp = datetime.datetime.now().timestamp()

        self.sum_op = 0.00
        self.repair_time = 30
        # processing time for each machine
        self.jobs_length = np.zeros(self.nb_jobs, dtype=np.int)
        # print(self.instance_matrix)

        # not only job also the machine need to chosen for CB
        # not required rigth now, just chose on eif possible.
        self.action_space = gym.spaces.Discrete(self.nb_jobs + 1)

        if self.instance_matrix is None:
            self.instance_matrix = np.zeros(
                (self.nb_jobs, self.nb_operations), dtype=(np.int, 2))
            job_init_cnt = 0

            while job_init_cnt < self.nb_jobs:

                temp_nb_proc = self.nb_operations  # create random number of process

                # randomely generate the machine requirements and times
                proc_cnt = 0
                while proc_cnt < temp_nb_proc:
                    machine, time = np.random.randint(0, self.nb_machines), \
                        np.random.randint(8, 10)

                    # random machine type, machine processing
                    # print(job_init_cnt, proc_cnt)
                    self.instance_matrix[job_init_cnt][proc_cnt] = (
                        machine, time)
                    # get max time for a single operation
                    self.max_time_op = max(self.max_time_op, time)

                    # time length for all jobs
                    self.jobs_length[job_init_cnt] += time
                    self.sum_op += time

                    proc_cnt += 1
                job_init_cnt += 1
        else:
            job_init_cnt = 0

            while job_init_cnt < self.nb_jobs:

                temp_nb_proc = self.nb_operations

                proc_cnt = 0
                while proc_cnt < temp_nb_proc:

                    # random machine type, machine processing
                    machine, time = self.instance_matrix[job_init_cnt][proc_cnt]
                    self.max_time_op = max(self.max_time_op, time)
                    self.jobs_length[job_init_cnt] += time
                    self.sum_op += time

                    proc_cnt += 1
                job_init_cnt += 1
        # TO do color scheme based on groups
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.nb_machines)
        ]

        #print("Instance: ", self.instance_matrix)

        #Machine Heaalth based on time

        self.machine_health = np.ones(
            (self.nb_machines, self.nb_type_max), dtype=np.float)*5

        self.breakdown_martix = np.eye(5, 5)*0.9

        '''
        matrix with the following attributes for each job:
        -Legal job
        -Left over time on the current op
        -Current operation %
        -Total left over time
        -When next machine available
        -Time since IDLE: 0 if not available, time otherwise
        -Total IDLE time in the schedule
        '''
        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0, 1, shape=(self.nb_jobs + 1,)),
            "real_obs": gym.spaces.Box(low=0.0, high=1.0, shape=(self.nb_jobs, 7), dtype=np.float),
        })

    def _get_current_state_representation(self):
        self.state[:, 0] = self.legal_actions[:-1]
        return {
            "real_obs": self.state,
            "action_mask": self.legal_actions,
        }

    def get_instance_matrix(self):
        return self.instance_matrix

    def get_legal_actions(self):
        return self.nb_legal_actions

    def reset(self):

        # get the max time of all jobs
        self.max_time_jobs = max(self.jobs_length)

        # print("max", self.max_time_op)
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.nb_jobs > 0
        assert self.nb_machines > 1, 'We need at least 2 machines'
        assert self.instance_matrix is not None

        self.current_time_step = 0
        self.next_time_step = list()
        self.next_jobs = list()
        # check on this-> is it at max

        # number of allowed actions
        self.nb_legal_actions = self.nb_jobs
        self.nb_machine_legal = 0

        # iniialize leagal actions

        self.legal_actions = np.ones(self.nb_jobs + 1, dtype=np.bool)
        self.legal_actions[self.nb_jobs] = False

        self.solution = np.full(
            (self.nb_jobs, self.nb_operations), -1, dtype=np.float)
        #change here TO DO

        # change the unavailabe machines to inf or machine legal
        #print(self.nb_machines, self.nb_type_max)

        ### Changed this part
        self.time_until_available_machine = np.zeros(
            (self.nb_machines, self.nb_type_max), dtype=np.float)
        for time_mach in range(self.time_until_available_machine.shape[0]):
            self.time_until_available_machine[time_mach][int(
                self.nb_machines_types[time_mach]):] = self.sum_op

        self.time_until_finish_current_op_jobs = np.zeros(
            self.nb_jobs, dtype=np.float)

        # on which job or fraction of jobs completed
        self.todo_time_step_job = np.zeros(self.nb_jobs, dtype=np.int)
        self.total_perform_op_time_jobs = np.zeros(self.nb_jobs, dtype=np.int)

        # change this to reflect left over time
        self.total_perform_op_time_jobs = np.copy(self.jobs_length)
        # store the machine needed for all jobs
        self.needed_machine_jobs = np.zeros(self.nb_jobs, dtype=np.int)
        self.total_idle_time_jobs = np.zeros(self.nb_jobs, dtype=np.float)
        self.idle_time_jobs_last_op = np.zeros(self.nb_jobs, dtype=np.float)

        # for no op
        self.illegal_actions = np.zeros(
            (self.nb_machines, self.nb_jobs), dtype=np.bool)
        self.action_illegal_no_op = np.zeros(self.nb_jobs, dtype=np.bool)

        # store availabe machines

        #change this to set as per number of machines
        self.machine_legal = np.ones(
            self.nb_machines, dtype=np.int)  # *self.nb_type_max
        for legal_mach in range(self.machine_legal.shape[0]):
            self.machine_legal[legal_mach] = int(
                self.nb_machines_types[legal_mach])

        # Can set machine legal as false have to check the update step for this and time availabe

        for job in range(self.nb_jobs):
            needed_machine = int(self.instance_matrix[job][0][0])
            self.needed_machine_jobs[job] = needed_machine

            # if machine availabe, use it and block it
            # check row sum and negate the first available

            # this is to check which machines are required\

            if self.machine_legal[needed_machine] > 0:
                self.machine_legal[needed_machine] -= 1
                # required is slightly diff as it is to set to keep count of requirement
                self.nb_machine_legal += 1
        #why did I repeat
        # to mark all machines as available
        self.machine_legal = np.ones(
            self.nb_machines, dtype=np.int)  # *self.nb_type_max
        for legal_mach in range(self.machine_legal.shape[0]):
            self.machine_legal[legal_mach] = int(
                self.nb_machines_types[legal_mach])
        # print(self.machine_legal)
        self.state = np.zeros((self.nb_jobs, 7), dtype=np.float)

        return self._get_current_state_representation()

    def remove_machine(self, action: int, machine_needed: int):
        print("remove", self.needed_machine_jobs[action], machine_needed)

        if self.needed_machine_jobs[action] == machine_needed:
            print("check broke machine")
            for i in range(np.shape(self.time_until_available_machine[machine_needed])[0]):
                #This will allow stopping extra machines
                if self.time_until_available_machine[machine_needed][i] == 0:
                    print("okay")
                    self.time_until_available_machine[machine_needed][i] = self.sum_op
                    self.legal_actions[self.nb_jobs] = True
                    return 0
        return 1
        """
        self.machine_legal[machine_needed] -= 1
        self.nb_machine_legal -= 1
        for job in range(self.nb_jobs):
            if self.needed_machine_jobs[job] == machine_needed:
                #print(
                #    "time_machine", self.time_until_available_machine[machine_needed])
                if self.time_until_available_machine[machine_needed].all() and self.legal_actions[job]:
                    self.legal_actions[job] = False
                    self.nb_legal_actions -= 1
        """

    def add_machine(self, machine_needed: int):
        for i in range(np.shape(self.time_until_available_machine[machine_needed])[0]):
            #This will allow stopping extra machines
            if self.time_until_available_machine[machine_needed][i] > self.max_time_op:
                self.time_until_available_machine[machine_needed][i] = 0
                self.legal_actions[self.nb_jobs] = True
                return 0
        return 1
        """
        self.machine_legal[machine_needed] += 1
        self.nb_machine_legal += 1
        for job in range(self.nb_jobs):
            if self.needed_machine_jobs[job] == machine_needed:
                #print(
                #    "time_machine", self.time_until_available_machine[machine_needed])
                if self.legal_actions[job]:
                    self.legal_actions[job] = True
                    self.nb_legal_actions += 1
        """
    """
    def stop_reset(self):

        #self.current_time_step = 0
        #self.next_time_step = list()
        #self.next_jobs = list()
        # check on this-> is it at max

        # number of allowed actions
        self.nb_legal_actions = self.nb_jobs
        self.nb_machine_legal = 0

        # iniialize leagal actions

        self.legal_actions = np.ones(self.nb_jobs + 1, dtype=np.bool)
        self.legal_actions[self.nb_jobs] = False

        self.time_until_available_machine = np.zeros(
            (self.nb_machines, self.nb_type_max), dtype=np.float)
        for time_mach in range(self.time_until_available_machine.shape[0]):
            self.time_until_available_machine[time_mach][int(
                self.nb_machines_types[time_mach]):] = self.sum_op

        #self.time_until_finish_current_op_jobs = np.zeros(
        #    self.nb_jobs, dtype=np.float)

        self.needed_machine_jobs = np.zeros(self.nb_jobs, dtype=np.int)

        self.illegal_actions = np.zeros(
            (self.nb_machines, self.nb_jobs), dtype=np.bool)
        self.action_illegal_no_op = np.zeros(self.nb_jobs, dtype=np.bool)

        # store availabe machines

        #change this to set as per number of machines
        self.machine_legal = np.ones(
            self.nb_machines, dtype=np.int)  # *self.nb_type_max
        for legal_mach in range(self.machine_legal.shape[0]):
            self.machine_legal[legal_mach] = int(
                self.nb_machines_types[legal_mach])

        # Can set machine legal as false have to check the update step for this and time availabe

        for job in range(self.nb_jobs):
            job_time_step = np.where(self.solution[job] == -1)

            needed_machine = int(
                self.instance_matrix[job][job_time_step[0][0]][0])
            self.needed_machine_jobs[job] = needed_machine

            # if machine availabe, use it and block it
            # check row sum and negate the first available

            # this is to check which machines are required\

            if self.machine_legal[needed_machine] > 0:
                self.machine_legal[needed_machine] -= 1
                # required is slightly diff as it is to set to keep count of requirement
                self.nb_machine_legal += 1
        #why did I repeat
        # to mark all machines as available
        self.machine_legal = np.ones(
            self.nb_machines, dtype=np.int)  # *self.nb_type_max
        for legal_mach in range(self.machine_legal.shape[0]):
            self.machine_legal[legal_mach] = int(
                self.nb_machines_types[legal_mach])

        return self._get_current_state_representation()
    """

    def _prioritization_non_final(self):
        if self.nb_machine_legal >= 1:
            for machine in range(self.nb_machines):
                # make changes
                if self.machine_legal[machine]:
                    final_job = list()
                    non_final_job = list()
                    min_non_final = float('inf')
                    for job in range(self.nb_jobs):
                        if self.needed_machine_jobs[job] == machine and self.legal_actions[job]:
                            if self.todo_time_step_job[job] == (self.nb_operations - 1):
                                final_job.append(job)
                            else:
                                current_time_step_non_final = self.todo_time_step_job[job]
                                time_needed_legal = self.instance_matrix[job][current_time_step_non_final][1]
                                machine_needed_nextstep = int(self.instance_matrix[
                                    job][current_time_step_non_final + 1][0])
                                if not np.all(self.time_until_available_machine[machine_needed_nextstep]):
                                    min_non_final = min(
                                        min_non_final, time_needed_legal)
                                    non_final_job.append(job)

                    if len(non_final_job) > 0:
                        for job in final_job:
                            current_time_step_final = self.todo_time_step_job[job]
                            time_needed_legal = self.instance_matrix[job][current_time_step_final][1]
                            if time_needed_legal > min_non_final:
                                self.legal_actions[job] = False
                                self.nb_legal_actions -= 1

    def _check_no_op(self):
        self.legal_actions[self.nb_jobs] = False
        if len(self.next_time_step) > 0 and self.nb_machine_legal <= 3 and self.nb_legal_actions <= 4:
            # print("no op")
            machine_next = set()
            next_time_step = self.next_time_step[0]
            max_horizon = self.current_time_step
            # change this to max for all machine types
            max_horizon_machine = [self.current_time_step
                                   + self.max_time_op for _ in range(self.nb_machines)]
            for job in range(self.nb_jobs):
                if self.legal_actions[job]:
                    time_step = self.todo_time_step_job[job]
                    machine_needed = int(
                        self.instance_matrix[job][time_step][0])
                    time_needed = self.instance_matrix[job][time_step][1]
                    end_job = self.current_time_step + time_needed
                    if end_job < next_time_step:
                        return
                    max_horizon_machine[machine_needed] = min(
                        max_horizon_machine[machine_needed], end_job)
                    max_horizon = max(
                        max_horizon, max_horizon_machine[machine_needed])
            for job in range(self.nb_jobs):
                if not self.legal_actions[job]:
                    # print(
                    #    " debug no op", job, self.time_until_finish_current_op_jobs[job], self.todo_time_step_job[job] + 1)
                    if self.time_until_finish_current_op_jobs[job] > 0 and \
                            self.todo_time_step_job[job] + 1 < self.nb_operations:
                        # print("jog", job)
                        time_step = self.todo_time_step_job[job] + 1
                        time_needed = self.current_time_step + \
                            self.time_until_finish_current_op_jobs[job]
                        while time_step < self.nb_operations - 1 and max_horizon > time_needed:
                            machine_needed = int(
                                self.instance_matrix[job][time_step][0])
                            if max_horizon_machine[machine_needed] > time_needed and self.machine_legal[machine_needed]:
                                machine_next.add(machine_needed)
                                # print("jogd", job)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.nb_jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1
                    elif not self.action_illegal_no_op[job] and self.todo_time_step_job[job] < self.nb_operations:
                        time_step = self.todo_time_step_job[job]
                        machine_needed = int(
                            self.instance_matrix[job][time_step][0])
                        time_needed = self.current_time_step + \
                            np.min(
                                self.time_until_available_machine[machine_needed])
                        while time_step < self.nb_operations - 1 and max_horizon > time_needed:
                            # print("ssss", job)
                            machine_needed = int(
                                self.instance_matrix[job][time_step][0])
                            if max_horizon_machine[machine_needed] > time_needed and self.machine_legal[machine_needed]:
                                machine_next.add(machine_needed)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.nb_jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1

    def step(self, action: int):
        reward = 0.0
        #change this to include stop as work
        # print("Step_taking")
        # no op
        # print("Action in env", action, self.legal_actions)
        if self.legal_actions[action]:
            #print("okas")
            if action == self.nb_jobs:
                print(" no op taken")

                self.nb_machine_legal = 0
                self.nb_legal_actions = 0
                for job in range(self.nb_jobs):

                    self.legal_actions[job] = False
                    needed_machine = self.needed_machine_jobs[job]
                    # self.machine_legal[needed_machine] -= 1
                    self.illegal_actions[needed_machine][job] = False
                    self.action_illegal_no_op[job] = False
                while self.nb_machine_legal == 0:
                    reward -= self._increase_time_step()
                    print("sdsd")

                self._prioritization_non_final()
                self._check_no_op()
                scaled_reward = self._reward_scaler(reward)
                return self._get_current_state_representation(), scaled_reward, self._is_done(), {}
            else:

                current_time_step_job = self.todo_time_step_job[action]
                machine_needed = self.needed_machine_jobs[action]
                time_needed = self.instance_matrix[action][current_time_step_job][1]
                #print(time_needed)
                reward += time_needed
                flag_not_allocated = 1
                for i in range(np.shape(self.time_until_available_machine[machine_needed])[0]):
                    #This will allow stopping extra machines
                    if self.time_until_available_machine[machine_needed][i] == 0 and self.machine_health[machine_needed][i] > 1:
                        self.time_until_available_machine[machine_needed][i] = time_needed
                        if random.uniform(0, 1) > 0.9:
                            self.machine_health[machine_needed][i] -= 1
                        flag_not_allocated = 0
                        break

                if flag_not_allocated:
                    for i in range(np.shape(self.time_until_available_machine[machine_needed])[0]):
                        #This will allow stopping extra machines
                        if self.time_until_available_machine[machine_needed][i]\
                         == 0 and self.machine_health[machine_needed][i] == 1:
                            self.time_until_available_machine[machine_needed][i] = self.repair_time
                            self.machine_health[machine_needed][i] = 5
                            break

                    print(" no op taken as machine broken")

                    self.nb_machine_legal = 0
                    self.nb_legal_actions = 0
                    for job in range(self.nb_jobs):

                        self.legal_actions[job] = False
                        needed_machine = self.needed_machine_jobs[job]
                        # self.machine_legal[needed_machine] -= 1
                        self.illegal_actions[needed_machine][job] = False
                        self.action_illegal_no_op[job] = False
                    while self.nb_machine_legal == 0:
                        reward -= self._increase_time_step()

                    self._prioritization_non_final()
                    self._check_no_op()
                    scaled_reward = self._reward_scaler(reward)
                    return self._get_current_state_representation(), scaled_reward, self._is_done(), {}

                self.time_until_finish_current_op_jobs[action] = time_needed
                self.state[action][1] = time_needed / self.max_time_op
                to_add_time_step = self.current_time_step + time_needed
                if to_add_time_step not in self.next_time_step:
                    index = bisect.bisect_left(
                        self.next_time_step, to_add_time_step
                    )
                    self.next_time_step.insert(index, to_add_time_step)
                    self.next_jobs.insert(index, action)
                #print(self.next_time_step)
                self.solution[action][current_time_step_job] = self.current_time_step
                self.legal_actions[action] = False
                self.nb_legal_actions -= 1
                self.machine_legal[machine_needed] -= 1
                self.nb_machine_legal -= 1
                for job in range(self.nb_jobs):
                    if self.needed_machine_jobs[job] == machine_needed:
                        #print(
                        #    "time_machine", self.time_until_available_machine[machine_needed])
                        if self.time_until_available_machine[machine_needed].all() and self.legal_actions[job] and job != action:
                            self.legal_actions[job] = False
                            self.nb_legal_actions -= 1
                            # print("sds")

                for job in range(self.nb_jobs):
                    if self.illegal_actions[machine_needed][job]:
                        self.action_illegal_no_op[job] = False
                        self.illegal_actions[machine_needed][job] = False

                # print("oks", self.nb_machine_legal,  len(
                #    self.next_time_step), self.nb_legal_actions)
                while self.nb_machine_legal == 0 and len(self.next_time_step) > 0:
                    #print("ok increae time")
                    reward -= self._increase_time_step()
                # print(self.legal_actions, " skksks")
                self._prioritization_non_final()
                self._check_no_op()
                # print("reward", reward)
                scaled_reward = self._reward_scaler(reward)
                #print("number", self.nb_machine_legal, self.nb_legal_actions)
                return self._get_current_state_representation(), scaled_reward, self._is_done(), {}
        else:
            return self._get_current_state_representation(), 0, self._is_done(), {}

    def _reward_scaler(self, reward):
        return reward / (self.max_time_op)

    def _increase_time_step(self):
        """
        Simulation for time..check when a  event happens
        """
        # print("Timestep", self.next_time_step)
        # if not self.next_time_step:
        # print(self.state)
        hole_planning = 0
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.next_jobs.pop(0)

        difference = next_time_step_to_pick - self.current_time_step
        self.current_time_step = next_time_step_to_pick
        #print("Next time step", self.current_time_step)
        # print("self.current", self.current_time_step, self.machine_legal)
        for job in range(self.nb_jobs):
            # print("job", job, self.nb_jobs)
            was_left_time = self.time_until_finish_current_op_jobs[job]
            # print(was_left_time)
            if was_left_time > 0.0:
                performed_op_job = min(difference, was_left_time)
                # update time in job
                self.time_until_finish_current_op_jobs[job] = max(
                    0.0, self.time_until_finish_current_op_jobs[job]-difference)
                self.state[job][1] = self.time_until_finish_current_op_jobs[job] / \
                    self.max_time_op
                self.total_perform_op_time_jobs[job] -= performed_op_job
                self.state[job][3] = self.total_perform_op_time_jobs[job] / \
                    self.max_time_jobs
                if self.time_until_finish_current_op_jobs[job] == 0.0:
                    self.total_idle_time_jobs[job] += (
                        difference - was_left_time)
                    self.state[job][6] = self.total_idle_time_jobs[job]/self.sum_op
                    self.idle_time_jobs_last_op[job] = (
                        difference - was_left_time)
                    self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op

                    self.todo_time_step_job[job] += 1
                    self.state[job][2] = self.todo_time_step_job[job] / \
                        self.nb_operations
                    if self.todo_time_step_job[job] < self.nb_operations:
                        self.needed_machine_jobs[job] = self.instance_matrix[job][self.todo_time_step_job[job]][0]
                        self.state[job][4] = max(
                            0.0, np.min(self.time_until_available_machine[self.needed_machine_jobs[job]])-difference) / self.max_time_op
                    else:
                        self.needed_machine_jobs[job] = -1
                        self.state[job][4] = -1.0
                        if self.legal_actions[job]:
                            self.legal_actions[job] = False
                            self.nb_legal_actions -= 1
            elif self.todo_time_step_job[job] < self.nb_operations:
                self.total_idle_time_jobs[job] += difference
                self.idle_time_jobs_last_op[job] += difference
                self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
        for machine in range(self.nb_machines):
            if np.min(self.time_until_available_machine[machine]) < difference:
                """
                check_machine_requirement = 0
                for job in range(self.nb_jobs):
                    if self.time_until_finish_current_op_jobs[job] == 0 and \
                            self.needed_machine_jobs[job] == machine and self.idle_time_jobs_last_op[job] != 0:
                        check_machine_requirement = 1
                        #print(job, self.idle_time_jobs_last_op[job])
                if check_machine_requirement:
                """
                idle_machines = self.time_until_available_machine[machine] < difference
                idle_avg = idle_machines * (difference
                                            - self.time_until_available_machine[machine])/idle_machines.shape[0]
                empty = np.sum(idle_avg)
                #print("hole", machine,
                #      self.time_until_available_machine[machine], empty)
                hole_planning += empty
            self.time_until_available_machine[machine] = np.maximum(
                  np.zeros(self.nb_type_max), self.time_until_available_machine[machine] - difference)
            # print("machine", machine,
            #      self.time_until_available_machine[machine], np.shape(
            #          np.where(self.time_until_available_machine[machine] == 0))[1])
            if not self.time_until_available_machine[machine].all():
                self.machine_legal[machine] = np.shape(
                    np.where(self.time_until_available_machine[machine] == 0))[1]
            else:
                self.machine_legal[machine] = 0
            temp_machine_count = self.machine_legal[machine]
            if not np.all(self.time_until_available_machine[machine]):
                for job in range(self.nb_jobs):
                    # print(self.needed_machine_jobs[job], self.legal_actions[job],
                    #      self.illegal_actions[machine][job], self.machine_legal[machine], self.time_until_finish_current_op_jobs[job])
                    if self.needed_machine_jobs[job] == machine and not self.legal_actions[job] and not self.illegal_actions[machine][job] and self.time_until_finish_current_op_jobs[job] == 0:
                        self.legal_actions[job] = True
                        self.nb_legal_actions += 1
                        if temp_machine_count > 0:
                            temp_machine_count -= 1
                            self.nb_machine_legal += 1

        return hole_planning

    def _is_done(self):
        if self.nb_legal_actions == 0:
            self.last_time_step = self.current_time_step
            return True
        return False

    def render(self, mode='human'):
        df = []
        for job in range(self.nb_jobs):
            i = 0
            while i < self.nb_operations and self.solution[job][i] != -1:
                #print("pl")
                dict_op = dict()
                dict_op["Task"] = 'Job {}'.format(job)
                start_sec = self.solution[job][i]  # *3600
                # print(start_sec, self.start_timestamp)
                finish_sec = start_sec + \
                    self.instance_matrix[job][i][1]  # *3600
                dict_op["Start"] = start_sec
                dict_op["Finish"] = finish_sec
                dict_op["Resource"] = "Machine {}".format(
                    self.instance_matrix[job][i][0])
                df.append(dict_op)
                i += 1
        fig = None

        if len(df) > 0:
            #print(self.solution)

            df = pd.DataFrame(df)
            df['Delta'] = df['Finish'] - df['Start']
            #print(df.head())
            # print("fig", df)
            fig = px.timeline(df, x_start="Start", x_end="Finish",
                              y="Task", color="Resource", hover_data=["Delta"], title="Current schedule makespan is "+str(self.makespan()))

            fig.layout.xaxis.type = 'linear'
            for i in range(df.Resource.nunique()):
                fig.data[i].x = df[df["Resource"]
                                   == fig.data[i].name].Delta.tolist()
            #fig.show()
            import io
            from PIL import Image

            #fig_bytes = fig.to_image(format="png")
            #buf = io.BytesIO(fig_bytes)
            #img_val = Image.open(buf)

            return fig  # np.asarray(img_val)

    def render_breakdown(self, break_time, repair_time, machin_num, mode='human'):
        df = []
        for job in range(self.nb_jobs):
            i = 0
            while i < self.nb_operations and self.solution[job][i] != -1:
                #print("pl")
                dict_op = dict()
                dict_op["Task"] = 'Job {}'.format(job)
                start_sec = self.solution[job][i]  # *3600
                # print(start_sec, self.start_timestamp)
                finish_sec = start_sec + \
                    self.instance_matrix[job][i][1]  # *3600
                dict_op["Start"] = start_sec
                dict_op["Finish"] = finish_sec
                dict_op["Resource"] = "Machine {}".format(
                    self.instance_matrix[job][i][0])
                df.append(dict_op)
                i += 1
        fig = None

        if len(df) > 0:
            #print(self.solution)

            df = pd.DataFrame(df)
            df['Delta'] = df['Finish'] - df['Start']
            #print(df.head())
            # print("fig", df)
            fig = px.timeline(df, x_start="Start", x_end="Finish",
                              y="Task", color="Resource", hover_data=["Delta"], title="Current schedule makespan is "+str(self.makespan()))

            fig.layout.xaxis.type = 'linear'
            for i in range(df.Resource.nunique()):
                fig.data[i].x = df[df["Resource"]
                                   == fig.data[i].name].Delta.tolist()

            fig.add_vrect(x0=break_time, x1=break_time+repair_time,
                          line_width=0, fillcolor="red", opacity=0.2)
            #fig.update_yaxes(autorange="reversed")
            #fig.show()
            import io
            from PIL import Image

            #fig_bytes = fig.to_image(format="png")
            #buf = io.BytesIO(fig_bytes)
            #img_val = Image.open(buf)

            return fig  # np.asarray(img_val)
    """
    def render(self, mode='human'):
        df = []
        for job in range(self.nb_jobs):
            i = 0
            while i < self.nb_operations and self.solution[job][i] != -1:
                dict_op = dict()
                dict_op["Task"] = 'Fabrication {}'.format(job)
                start_sec = 0+self.solution[job][i]
                # print(start_sec, self.start_timestamp)
                finish_sec = start_sec + self.instance_matrix[job][i][1]
                dict_op["Start"] = start_sec
                dict_op["Finish"] = finish_sec
                dict_op["Resource"] = "Machine {}".format(
                    self.instance_matrix[job][i][0])
                df.append(dict_op)
                i += 1
        fig = None
        if len(df) > 0:
            #print(self.solution)
            df = pd.DataFrame(df)
            print(df.head())
            df["Start"] = pd.to_datetime(df["Start"]*3600, unit="s")
            df["Start"] = df["Start"].apply(
                lambda x: x.replace(year=2022, month=6, day=1))
            df["Finish"] = pd.to_datetime(df["Finish"]*3600, unit="s")
            df["Finish"] = df["Finish"].apply(
                lambda x: x.replace(year=2022, month=6, day=1))
            print(df.head())
            # print("fig", df)
            fig = px.timeline(df, x_start="Start", x_end="Finish",
                              y="Task", color="Resource")
            fig.update_xaxes(
                tickformat="%H",
                tickformatstops=[
                    dict(dtickrange=[3600000, 86400000], value="%H")]  # range is 1 hour to 24 hours
            )
            # otherwise tasks are listed from the bottom up
            fig.update_yaxes(autorange="reversed")
        return fig
    """

    def makespan(self):
        max_make = 0
        finish_sec = datetime.datetime.fromtimestamp(self.start_timestamp)
        start_time = datetime.datetime.fromtimestamp(self.start_timestamp)
        for job in range(self.nb_jobs):
            i = 0
            while i < self.nb_operations and self.solution[job][i] != -1:
                start_sec = self.start_timestamp + self.solution[job][i]*3600.0
                finish_sec = datetime.datetime.fromtimestamp(
                    start_sec + self.instance_matrix[job][i][1]*3600.0)
                i += 1
            if job == 0:
                max_finish = finish_sec
            elif finish_sec > max_finish:
                max_finish = finish_sec
                #print(max_finish-start_time)

        td = max_finish-start_time
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        Total_hrs = days*24 + hours
        return Total_hrs
