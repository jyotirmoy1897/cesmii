"""
Created by Dyutimoy

Combined Wandb with streamlit and apply rl using stable baselines for PPO

Be careful -
Have modified ppo for handling the logit
It gives very less probaility to illegal action
File changed: stable_baselines3/commom/policies.py
the function get _get_action_dist_from_latent has been modified for categorical distribution
The changes are PPO Mod folder
"""


# from stable_baselines3 import PPO


# Create a heading
import streamlit as st
import pandas as pd
import imageio
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from datetime import datetime
import math
import io
from PIL import Image
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import gym
import wandb
from wandb.integration.sb3 import WandbCallback
from ppo_mod import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import random
import time
import requests

import streamlit.components.v1 as components


st.title("🔎 Reinforcement Learning based Job scheduling")

st.sidebar.image("Images/psulogo.png")

page_select = st.sidebar.selectbox("Select Page", options=(
    'Introduction', 'Train Algorithm', "Breakdown Simulation", "Simantha integration"))

if page_select == "Introduction":
    st.header("Scheduling Problems")
    st.write("High Mix Low Volume (HMLV) fabrication involves customized \
    product manufacturing in small quantities.  For each fabrication a \
    sequence of machinary is required with a corresponding processing time. \
     This forms a combinatorial optimization problem\
      to minimize the makespan.")

    st.image('Images/blue.png')

    st.header("Reinforcement Learning")
    st.write("Reinforcement learning involves setting up an environment and \
    training an agent to take the optimal action through a reward system.\
     In scheduling problem, the reward is given to minimize shut down periods\
      of machine. For example, if any assignment of job to machine results in\
       idle time in another machine, the allotment is penalized.")

if page_select == "Train Algorithm":

    st.header("Information regarding scheduling task list and factory setup")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Factory setup")
        st.write("Please provide information regarding available machinary.\
        Our system accepts an excel sheet with table containing detail of number\
        of available machines of each type")

        st.markdown(
            """
            <style>
            .streamlit-expanderHeader {
                font-size: large;
            }
            </style>
            """,
            unsafe_allow_html=True,
            )

        with st.expander(("See example 🦾 ")):
            st.markdown("Example: if said factory has 3 machine types")
            st.markdown("- Type 1(indexed by 0) consists of 2 machines")
            st.markdown("- Type 2(indexed by 1) consists of 3 machines")
            st.markdown("- Type 3(indexed by 2) consists of 4 machines")

            machine_default = np.array([2, 3, 4])
            machine_info_file = "./instances/Machine_nb_machine_" + \
                str(3)+"_of_type_"+str(machine_default) + \
                "_version_"+str(1)+".csv"
            machine_info_default = np.loadtxt(machine_info_file, delimiter=",")
            st.dataframe(machine_info_default)
            st.warning("The machine type indexing starts from 0")
    with c2:
        st.subheader("Task information")
        st.write("Information regarding each fabrication should be given serially, \
        where each row corresponds to a fabrication and all the machine required \
        in process sequence followed by processing time on each machine.")

        with st.expander(("See example 🎛")):
            st.write(
                "Example: For 5 fabrications where each job takes 5 tasks to get\
                 fabricated ")
            machine_default = np.array([2, 3, 4])
            instance_file_default = "./instances/Instance_with_nb_jobs_"+str(5)+"nb_operations_"+str(
                5)+"_nb_machine_"+str(3)+"_of_type_"+str(machine_default)+"_version_"+str(1)+".csv"
            instance_matrix = np.loadtxt(instance_file_default, delimiter=",")

            nb_jobs, nb_operations = np.shape(instance_matrix)
            nb_operations = int(nb_operations)/2

            #instance_matrix = np.reshape(
            #    instance_matrix, (int(nb_jobs), int(nb_operations), 2), order='F')

            st.write("The csv or excel file should be in following format:")
            st.dataframe(instance_matrix)

            st.write("In above excel file, the first five columns(the light green)\
            corresponding to the routing information where each row is for a single\
             fabrication job.The light blue part correponds to the time taken for each\
             fabrication at corresponding machine type in its route.")
            st.image("Images/example3.png")
            st.warning("All indexing starts from 0")

    with st.form("Problem_Instance"):
        tab_data, tab_para = st.tabs(["Input Data", "Input Parameters"])
        with tab_data:
            st.write("Please input the factory information")
            nb_machines_types_file = st.file_uploader(
                "🏭 ", help=("The file should contain an excel of type of"))
            st.write("Input the fabrication informations")
            instance_file = st.file_uploader("📑")

        with tab_para:
            st.write("Input the range of parameters to tune using WandB")
            st.write("enter your wandb username")
            username = st.text_input("Username", value='xenopsu')
            st.write("Project name")
            projectname = st.text_input("Project", value="rl")

            timestep_vals = st.multiselect(
                "Number of timesteps", [10000, 300000, 500000, 600000, 700000, 800000], [600000, 800000])

            entropy_vals = st.slider(
                "Select entropy value 1e-2", 0.01, 0.50, (0.05, 0.10))

            value_func_vals = st.slider(
                "Select vf coeffecient range", 0.5, 1.0, (0.7, 0.8))

            vf_vals = st.multiselect("Select layer size for value function", [
                                     '319', '256', '512', '1024'], ['319', '256'])
            pi_vals = st.multiselect("Select layer size for policy function", [
                                     '319', '256', '512', '1024'], ['319', '256'])

        st.warning("💡 Make sure to input both folders and tuning parameters")
        prob_submitted = st.form_submit_button(
            "Submit after Inputing files")

    if prob_submitted:
        st.info("💡 Proceed to results page")
        instance_matrix = np.loadtxt(instance_file, delimiter=",")
        print(instance_matrix)
        nb_machines_types = np.loadtxt(
            nb_machines_types_file, delimiter=",")

        nb_jobs, nb_operations = np.shape(instance_matrix)
        nb_jobs = int(nb_jobs)
        nb_operations = nb_operations/2
        nb_operations = int(nb_operations)
        instance_matrix = np.reshape(
            instance_matrix, (int(nb_jobs), int(nb_operations), 2), order='F')
        max_ep_len = (nb_jobs)*(nb_operations)*2
        nb_machines = np.shape(nb_machines_types)
        nb_machines = nb_machines[0]

        op_columns = []
        for op_col in range(nb_operations):
            op_columns.append("step_"+str(op_col))

        op_rows = []
        for op_row in range(nb_jobs):
            op_rows.append("Fabrication_job_"+str(op_row))
        route_df = instance_matrix[:, :, 0].astype("int")
        df1 = pd.DataFrame(
            route_df, columns=op_columns, index=op_rows)
        st.write("Route Information")
        st.dataframe(df1)

        st.write("Processing Time in each operation")
        op_columns = []
        for op_col in range(nb_operations):
            op_columns.append("Time_Taken_"+str(op_col))

        op_rows = []
        for op_row in range(nb_jobs):
            op_rows.append("Fabrication_job_"+str(op_row))

        df2 = pd.DataFrame(
            instance_matrix[:, :, 1], columns=op_columns, index=op_rows)
        st.dataframe(df2)

        entropy_vals_min = float(entropy_vals[0])*0.01
        entropy_vals_max = float(entropy_vals[1])*0.01
        st.write("Entropy value range from ",
                 entropy_vals_min, " to ", entropy_vals_max)
        with st.expander("View Results"):
            device = torch.device('cpu')

            if(torch.cuda.is_available()):
                device = torch.device('cuda:0')
                torch.cuda.empty_cache()
                print("Device set to :"
                      + str(torch.cuda.get_device_name(device)))
            else:
                print("Device set to: cpu")
            instance_matrix = instance_matrix.astype("int")
            nb_machines_types = nb_machines_types.astype("int")

            env = gym.make('DSEnv:ds-v1', nb_jobs=nb_jobs, nb_operations=nb_operations,
                           nb_machines=int(nb_machines), nb_machines_types=nb_machines_types, instance_file=instance_matrix)

            pi_vals = [float(x) for x in pi_vals]
            vf_vals = [float(x) for x in vf_vals]
            sweep_config = {
                'method': 'random'
                }

            parameters_dict = {
                'policy_type': {
                    'value': 'MultiInputPolicy'
                    },
                'total_timesteps': {
                    'values': timestep_vals
                    },
                'entropy': {
                    'distribution': 'uniform',
                    'min': entropy_vals_min,
                    'max': entropy_vals_max,
                    },
                'value_func': {
                    'distribution': 'uniform',
                    'min': float(value_func_vals[0]),
                    'max': float(value_func_vals[1]),
                    },
                #'hidden_layer': {
                #    'values': [256, 512, 1024]
                #    },
                'vf_value': {
                    'values': vf_vals
                    },
                'pi_value': {
                    'values': pi_vals
                    },
                'learning_rate': {
                    'distribution': 'uniform',
                    'min': 0.0007,
                    'max': 0.005
                    },
                'n_epochs': {
                    'values': [12, 14]
                    }
                }

            sweep_config['parameters'] = parameters_dict

            # ***********************************************

            word_site = "https://www.mit.edu/~ecprice/wordlist.10000"

            response = requests.get(word_site)
            WORDS = [w.decode("UTF-8")
                     for w in response.content.splitlines()]

            # ************************************************
            project = projectname
            entity = username

            HEIGHT = 720

            def get_project(api, name, entity=None):
                return api.project(name, entity=entity).to_html(height=HEIGHT)

            wandb.login(anonymous="must")
            api = wandb.Api()

            config_pl = st.empty()
            config_pl.info("sweep_started")

            reward_pl = st.empty()
            reward_pl.info(" Processing")
            components.html(get_project(
                api, project, entity), height=HEIGHT)
            sweep_id = wandb.sweep(
                sweep_config, project=project)
            mean_val = 0

            def train(config=None, env=env):
                experiment_name = "-".join(random.choices(WORDS, k=2)
                                           ) + f"-{random.randint(0,100)}"
                run = wandb.init(name=experiment_name, config=config,
                                 sync_tensorboard=True, monitor_gym=True, save_code=True)
                config = wandb.config
                print("Config", config)

                from typing import Union, Callable

                def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
                    """
                    Linear learning rate schedule.
                    :param initial_value: (float or str)
                    :return: (function)
                    """
                    if isinstance(initial_value, str):
                        initial_value = float(initial_value)

                    def func(progress_remaining: float) -> float:
                        """
                        Progress will decrease from 1 (beginning) to 0
                        :param progress_remaining: (float)
                        :return: (float)
                        """
                        if (math.sqrt(progress_remaining) * initial_value) > 5e-4:
                            return (math.sqrt(progress_remaining) * initial_value)
                        else:
                            return 5e-4

                    return func
                #model = PPO(config["policy_type"], env,
                #            verbose=1, tensorboard_log=f"runs/{run.id}")
                #env = make_vec_env(env, n_envs=4)
                #config_pl.info("Current config")
                model = PPO(config["policy_type"], env, verbose=1, batch_size=5000, n_steps=5000, n_epochs=config['n_epochs'], gamma=1,
                            learning_rate=linear_schedule(config['learning_rate']), ent_coef=config['entropy'], vf_coef=config["value_func"], policy_kwargs={
                            "net_arch": [dict(vf=[config['vf_value'], config['vf_value']], pi=[config['pi_value'], config['pi_value']])]}, tensorboard_log=f"runs/{run.id}")

                model.learn(
                    total_timesteps=config["total_timesteps"],
                    callback=WandbCallback(
                        gradient_save_freq=100,
                        model_save_path=f"models/{experiment_name}+{run.id}",
                        verbose=2,
                    ),
                )

                #mean_reward, std_reward = evaluate_policy(
                #    model, env, n_eval_episodes=5, deterministic=True)

                #print("mean_val", mean_reward)
                #reward_pl.info(f"reward {mean_reward}")
                run.finish()

            wandb.agent(sweep_id, train, count=10)


if page_select == "Breakdown Simulation":
    st.header("Simulate breakdown")
    st.write("We generate breakdown and explore various scenarios")
    with st.form("Problem and model input"):
        tab_input, tab_breakdown = st.tabs(
            ["Input Data and model", "Input breakdown details"])
        with tab_input:
            st.write("Please input the factory information")
            nb_machines_types_file = st.file_uploader(
                "🏭 ", help=("The file should contain an excel of type of"))
            st.write("Input the fabrication informations")
            instance_file = st.file_uploader("📑")

            st.write("Input the trained model file")
            model_file = st.file_uploader("📑 model")
        with tab_breakdown:
            st.write("Performing reschedule")
            st.header("Modify schedule")
            st.write("Input machine number and delay time")

            mach_num = st.number_input("Input machine number", 0, 100)
            start_time = st.slider(
                    " break time", min_value=0, max_value=1000, value=0)
            break_time = st.slider(" time", min_value=0,
                                   max_value=500, value=0)

        model_submitted = st.form_submit_button(
                            "Submit after Inputing model files")

        if model_submitted:

            st.info("💡 Genereting Schedule")
            instance_matrix = np.loadtxt(instance_file, delimiter=",")
            print(instance_matrix)
            nb_machines_types = np.loadtxt(
                        nb_machines_types_file, delimiter=",")

            nb_jobs, nb_operations = np.shape(instance_matrix)
            nb_jobs = int(nb_jobs)
            nb_operations = nb_operations/2
            nb_operations = int(nb_operations)
            instance_matrix = np.reshape(
                        instance_matrix, (int(nb_jobs), int(nb_operations), 2), order='F')
            max_ep_len = (nb_jobs)*(nb_operations)*2
            nb_machines = np.shape(nb_machines_types)
            nb_machines = nb_machines[0]
            instance_matrix = instance_matrix.astype("int")
            nb_machines_types = nb_machines_types.astype("int")
            env = gym.make('DSEnv:ds-v1', nb_jobs=nb_jobs, nb_operations=nb_operations,
                           nb_machines=nb_machines, nb_machines_types=nb_machines_types, instance_file=instance_matrix)
            actions_list = []

            tab_opt, tab_fifo, tab_mtwr = st.tabs(
                ["RL solution", "FIFO", "MTWR"])

            with tab_opt:
                image_pl = st.empty()
                obs = env.reset()
                dones = False
                model = PPO.load(model_file)
                count = 0
                while (not dones and count < 1000):
                    action, _states = model.predict(obs)
                    actions_list.append(action)
                    obs, rewards, dones, info = env.step(action)
                    count += 1

                    if dones:
                        print("Makespan", env.makespan())
                        st.write("Makespan")
                        st.write(env.makespan())
                        print(env.solution)
                        temp_image = env.render()
                        #buf = io.BytesIO(temp_image)
                        #img = Image.fromarray(temp_image)
                        image_pl.plotly_chart(temp_image)

                        break

                count = 0
                obs = env.reset()
                dones = False
                machine_break = mach_num
                flag_remove = 1
                flag_add = 1
                mode_pl = st.empty()
                info_pl = st.empty()
                while (not dones and count < 1000):

                    if count % 100 == 0:
                        print("Makespan", env.makespan(),
                              env.time_until_available_machine[machine_break])

                    if env.current_time_step <= start_time:
                        obs, rewards, dones, info = env.step(
                            actions_list[count])
                    else:
                        action, _states = model.predict(obs)
                        if env.current_time_step > start_time and env.current_time_step <= start_time + break_time and flag_remove and action != nb_jobs:
                            flag_remove = env.remove_machine(
                                action, machine_break)
                            #print(flag_remove)
                            info_pl.warning(
                                "The machine "+str(machine_break)+" has stopped working")
                            if flag_remove == 0:
                                action = nb_jobs

                        if env.current_time_step > start_time + break_time and flag_add:
                            flag_add = env.add_machine(machine_break)
                            info_pl.success(
                                "The machine "+str(machine_break)+" has been reparied")
                            #print(flag_add)
                            if flag_add == 0:
                                action = nb_jobs

                        obs, rewards, dones, info = env.step(action)

                    count += 1

                    temp_image = env.render_breakdown(
                        start_time, break_time, machine_break)
                    #buf = io.BytesIO(temp_image)
                    #img = Image.fromarray(temp_image)
                    mode_pl.plotly_chart(temp_image)

                    if dones:
                        print("Makespan", env.makespan())
                        st.write("Makespan")
                        st.write(env.makespan())
                        print(env.solution)

                        break
            with tab_fifo:
                st.write("For First In First Out (FIFO)")
                fifo_pl = st.empty()
                obs = env.reset()
                dones = False
                model = PPO.load(model_file)
                count = 0
                while (not dones and count < 1000):
                    count += 1
                    real_state = np.copy(obs['real_obs'])
                    legal_actions = obs['action_mask'][:-1]
                    reshaped = np.reshape(real_state, (env.nb_jobs, 7))
                    remaining_time = reshaped[:, 5]
                    illegal_actions = np.invert(legal_actions)
                    mask = illegal_actions * -1e8
                    remaining_time += mask
                    FIFO_action = np.argmax(remaining_time)
                    print(FIFO_action, legal_actions)
                    assert legal_actions[FIFO_action]
                    state, reward, done, _ = env.step(FIFO_action)

                    if done:
                        print("Makespan", env.makespan())
                        st.write("Makespan")
                        st.write(env.makespan())
                        print(env.solution)
                        temp_image = env.render()
                        #buf = io.BytesIO(temp_image)
                        #img = Image.fromarray(temp_image)
                        fifo_pl.plotly_chart(temp_image)

                        break

                count = 0
                obs = env.reset()
                dones = False
                machine_break = mach_num
                flag_remove = 1
                flag_add = 1
                mode_pl = st.empty()
                info_pl = st.empty()
                while (not dones and count < 1000):

                    if count % 100 == 0:
                        print("Makespan", env.makespan(),
                              env.time_until_available_machine[machine_break])

                    if env.current_time_step <= start_time:
                        count += 1
                        real_state = np.copy(obs['real_obs'])
                        legal_actions = obs['action_mask'][:-1]
                        reshaped = np.reshape(real_state, (env.nb_jobs, 7))
                        remaining_time = reshaped[:, 5]
                        illegal_actions = np.invert(legal_actions)
                        mask = illegal_actions * -1e8
                        remaining_time += mask
                        FIFO_action = np.argmax(remaining_time)
                        print(FIFO_action, legal_actions)
                        assert legal_actions[FIFO_action]
                        state, reward, done, _ = env.step(FIFO_action)

                    else:
                        count += 1
                        real_state = np.copy(obs['real_obs'])
                        legal_actions = obs['action_mask'][:-1]
                        reshaped = np.reshape(real_state, (env.nb_jobs, 7))
                        remaining_time = reshaped[:, 5]
                        illegal_actions = np.invert(legal_actions)
                        mask = illegal_actions * -1e8
                        remaining_time += mask
                        FIFO_action = np.argmax(remaining_time)
                        print(FIFO_action, legal_actions)
                        assert legal_actions[FIFO_action]
                        action = FIFO_action
                        if env.current_time_step > start_time and env.current_time_step <= start_time + break_time and flag_remove and action != nb_jobs:
                            flag_remove = env.remove_machine(
                                action, machine_break)
                            #print(flag_remove)
                            info_pl.warning(
                                "The machine "+str(machine_break)+" has stopped working")
                            if flag_remove == 0:
                                action = nb_jobs

                        if env.current_time_step > start_time + break_time and flag_add:
                            flag_add = env.add_machine(machine_break)
                            info_pl.success(
                                "The machine "+str(machine_break)+" has been reparied")
                            #print(flag_add)
                            if flag_add == 0:
                                action = nb_jobs

                        obs, rewards, dones, info = env.step(action)

                    temp_image = env.render_breakdown(
                        start_time, break_time, machine_break)
                    #buf = io.BytesIO(temp_image)
                    #img = Image.fromarray(temp_image)
                    mode_pl.plotly_chart(temp_image)

                    if dones:
                        print("Makespan", env.makespan())
                        st.write("Makespan")
                        st.write(env.makespan())
                        print(env.solution)

                        break
            with tab_mtwr:
                st.write("For MTWR")
                mtwr_pl = st.empty()
                obs = env.reset()
                dones = False
                model = PPO.load(model_file)
                count = 0
                while (not dones and count < 1000):
                    count += 1
                    real_state = np.copy(obs['real_obs'])
                    legal_actions = obs['action_mask'][:-1]
                    reshaped = np.reshape(real_state, (env.nb_jobs, 7))
                    remaining_time = (
                        reshaped[:, 3] * env.max_time_jobs) / env.jobs_length
                    illegal_actions = np.invert(legal_actions)
                    mask = illegal_actions * 1e8
                    remaining_time += mask
                    MTWR_action = np.argmin(remaining_time)
                    assert legal_actions[MTWR_action]
                    state, reward, done, _ = env.step(MTWR_action)

                    if done:
                        print("Makespan", env.makespan())
                        st.write("Makespan")
                        st.write(env.makespan())
                        print(env.solution)
                        temp_image = env.render()
                        #buf = io.BytesIO(temp_image)
                        #img = Image.fromarray(temp_image)
                        mtwr_pl.plotly_chart(temp_image)

                        break

                count = 0
                obs = env.reset()
                dones = False
                machine_break = mach_num
                flag_remove = 1
                flag_add = 1
                mode_pl = st.empty()
                info_pl = st.empty()
                while (not dones and count < 1000):

                    if count % 100 == 0:
                        print("Makespan", env.makespan(),
                              env.time_until_available_machine[machine_break])

                    if env.current_time_step <= start_time:
                        count += 1
                        real_state = np.copy(obs['real_obs'])
                        legal_actions = obs['action_mask'][:-1]
                        reshaped = np.reshape(real_state, (env.nb_jobs, 7))
                        remaining_time = (
                            reshaped[:, 3] * env.max_time_jobs) / env.jobs_length
                        illegal_actions = np.invert(legal_actions)
                        mask = illegal_actions * 1e8
                        remaining_time += mask
                        MTWR_action = np.argmin(remaining_time)
                        assert legal_actions[MTWR_action]
                        state, reward, done, _ = env.step(MTWR_action)

                    else:
                        count += 1
                        real_state = np.copy(obs['real_obs'])
                        legal_actions = obs['action_mask'][:-1]
                        reshaped = np.reshape(real_state, (env.nb_jobs, 7))
                        remaining_time = (
                            reshaped[:, 3] * env.max_time_jobs) / env.jobs_length
                        illegal_actions = np.invert(legal_actions)
                        mask = illegal_actions * 1e8
                        remaining_time += mask
                        MTWR_action = np.argmin(remaining_time)
                        assert legal_actions[MTWR_action]

                        action = MTWR_action
                        if env.current_time_step > start_time and env.current_time_step <= start_time + break_time and flag_remove and action != nb_jobs:
                            flag_remove = env.remove_machine(
                                action, machine_break)
                            #print(flag_remove)
                            info_pl.warning(
                                "The machine "+str(machine_break)+" has stopped working")
                            if flag_remove == 0:
                                action = nb_jobs

                        if env.current_time_step > start_time + break_time and flag_add:
                            flag_add = env.add_machine(machine_break)
                            info_pl.success(
                                "The machine "+str(machine_break)+" has been reparied")
                            #print(flag_add)
                            if flag_add == 0:
                                action = nb_jobs

                        obs, rewards, dones, info = env.step(action)

                    temp_image = env.render_breakdown(
                        start_time, break_time, machine_break)
                    #buf = io.BytesIO(temp_image)
                    #img = Image.fromarray(temp_image)
                    mode_pl.plotly_chart(temp_image)

                    if dones:
                        print("Makespan", env.makespan())
                        st.write("Makespan")
                        st.write(env.makespan())
                        print(env.solution)

                        break
            #rerun = st.checkbox('Rerun', on_change=rerun_schedule(
            #    env, mach_num, start_time, break_time))


if page_select == "Simantha integration":

    st.header("Visual machine health progress")

    st.write("We take a sequence of machine schedule and simulate optimal path \
         and visual corresponding machhine health trajectory")
    st.write(" Each sequence consist of multiple job list in order of occurance")

    with st.form("Sequence Input"):
        st.write("Please input all the relevant files")
        file_list = st.file_uploader(
            "Upload file", accept_multiple_files=True)
        st.write("Please input the factory information")
        nb_machines_types_file = st.file_uploader(
            "🏭 ", help=("The file should contain an excel of type of"))

        st.write("Input the trained model file")
        model_file = st.file_uploader("📑 model")
        folder_submitted = st.form_submit_button(
            "Submit after uploading folder")

        if folder_submitted:

            st.info("Simulating optimal trajectory for sequence")
            sequence_count = 0
            info_pl = st.empty()
            image_pl = st.empty()
            machine_pl = st.empty()
            machine_health_end = None
            nb_machines_types = np.loadtxt(
                        nb_machines_types_file, delimiter=",")
            for filename in file_list:
                instance_file = filename
                sequence_count += 1

                info_pl.info("Processing file number "
                             + str(sequence_count))

                instance_matrix = np.loadtxt(instance_file, delimiter=",")
                print(instance_matrix)

                nb_jobs, nb_operations = np.shape(instance_matrix)
                nb_jobs = int(nb_jobs)
                nb_operations = nb_operations/2
                nb_operations = int(nb_operations)
                instance_matrix = np.reshape(
                            instance_matrix, (int(nb_jobs), int(nb_operations), 2), order='F')
                max_ep_len = (nb_jobs)*(nb_operations)*2
                nb_machines = np.shape(nb_machines_types)
                nb_machines = nb_machines[0]
                instance_matrix = instance_matrix.astype("int")
                nb_machines_types = nb_machines_types.astype("int")

                env = gym.make('DSEnv:ds-v1', nb_jobs=nb_jobs, nb_operations=nb_operations,
                               nb_machines=nb_machines, nb_machines_types=nb_machines_types, instance_file=instance_matrix, machine_health=machine_health_end)
                actions_list = []
                obs = env.reset()
                dones = False
                model = PPO.load(model_file)
                count = 0
                while (not dones and count < 1000):
                    action, _states = model.predict(obs)
                    actions_list.append(action)
                    obs, rewards, dones, info = env.step(action)
                    count += 1

                    #if info["breakdown"] != -1:
                    #    machine_pl.info(
                    #        f"The following machine {info['breakdown']} of index {info['index']} has broken down")
                    temp_image = env.render()
                    #buf = io.BytesIO(temp_image)
                    #img = Image.fromarray(temp_image)
                    image_pl.plotly_chart(temp_image)
                    info_df = pd.DataFrame(info['machine_health'])
                    fig_machine = px.bar(
                        info_df)
                    machine_pl.plotly_chart(fig_machine)

                    if dones:
                        print("Makespan", env.makespan())
                        st.write("Makespan")
                        st.write(env.makespan())
                        print(env.solution)
                        temp_image = env.render()
                        #buf = io.BytesIO(temp_image)
                        #img = Image.fromarray(temp_image)
                        image_pl.plotly_chart(temp_image)
                        machine_health_end = info['machine_health']

                        break
                        break
