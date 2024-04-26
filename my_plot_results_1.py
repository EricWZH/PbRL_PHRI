#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot results
"""

import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from stable_baselines3.common.monitor import get_monitor_files

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def read_csv_data(par_path, tensorboard_log_list, total_timesteps, n_envs,
                  n_steps, smooth):

    nums = len(tensorboard_log_list)
    x = np.arange(0, total_timesteps, n_envs * n_steps)

    total_return_array = np.zeros(
        (nums, total_timesteps // (n_envs * n_steps)))
    task_success_array = np.zeros(
        (nums, total_timesteps // (n_envs * n_steps)))
    pref_return_array = np.zeros((nums, total_timesteps // (n_envs * n_steps)))

    for i in range(nums):
        cur_path = os.path.join(par_path, tensorboard_log_list[i])
        monitor_logs_path_list = get_monitor_files(cur_path)
        for logs_path in monitor_logs_path_list:
            with open(logs_path, "rt") as file_handler:
                first_line = file_handler.readline()
                assert first_line[0] == "#"
                header = json.loads(first_line[1:])
                data_frame = pd.read_csv(file_handler, index_col=None)
                total_return_array[i, :] += data_frame[
                    'total_reward'].values[:total_timesteps //
                                           (n_envs * n_steps)]
                task_success_array[i, :] += data_frame[
                    'task_success'].values[:total_timesteps //
                                           (n_envs * n_steps)]
                pref_return_array[i, :] += data_frame[
                    'pref_reward'].values[:total_timesteps //
                                          (n_envs * n_steps)]
    total_return_array = total_return_array / n_envs
    task_success_array = task_success_array / n_envs
    pref_return_array = pref_return_array / n_envs

    # 对数据作平滑处理
    # sac:100, ppo:5
    smooth_total_return_array = np.zeros(total_return_array.shape)
    smooth_pref_return_array = np.zeros(pref_return_array.shape)
    smooth_task_success_array = np.zeros(task_success_array.shape)
    for j in range(total_return_array.shape[0]):
        for k in range(total_return_array.shape[1]):
            smooth_total_return_array[j, k] = np.mean(
                total_return_array[j, max(0, k - smooth + 1):k + 1])
            smooth_pref_return_array[j, k] = np.mean(
                pref_return_array[j, max(0, k - smooth + 1):k + 1])
            smooth_task_success_array[j, k] = np.mean(
                task_success_array[j, max(0, k - smooth + 1):k + 1])

    mean_total_return_array = np.mean(smooth_total_return_array, axis=0)
    std_total_return_array = np.std(smooth_total_return_array, axis=0) / 2
    mean_pref_return_array = np.mean(smooth_pref_return_array, axis=0)
    std_pref_return_array = np.std(smooth_pref_return_array, axis=0) / 2
    mean_task_success_array = np.mean(smooth_task_success_array, axis=0)
    std_task_success_array = np.std(smooth_task_success_array, axis=0) / 2
    return x, mean_total_return_array, std_total_return_array, mean_pref_return_array, std_pref_return_array, mean_task_success_array, std_task_success_array


def plot_curve(axs, x, mean_total_return_array, std_total_return_array,
               mean_pref_return_array, std_pref_return_array,
               mean_task_success_array, std_task_success_array, color, label):
    index = np.arange(0, len(x), len(x) // 100)
    axs[0].plot(x[index], mean_task_success_array[index], lw=2, color=color)
    axs[0].fill_between(
        x[index],
        mean_task_success_array[index] - std_task_success_array[index],
        mean_task_success_array[index] + std_task_success_array[index],
        alpha=.3,
        lw=0,
        color=color)
    axs[0].set_ylabel('Success Rate', fontsize='x-large')
    axs[0].set_xlabel('Environment Steps', fontsize='x-large')
    axs[1].plot(x[index],
                mean_total_return_array[index],
                lw=2,
                color=color,
                label=label)
    axs[1].fill_between(
        x[index],
        mean_total_return_array[index] - std_total_return_array[index],
        mean_total_return_array[index] + std_total_return_array[index],
        alpha=.3,
        lw=0,
        color=color)
    axs[1].set_ylabel('Episode True Return', fontsize='x-large')
    axs[1].set_xlabel('Environment Steps', fontsize='x-large')
    axs[2].plot(x[index], mean_pref_return_array[index], lw=2, color=color)
    axs[2].fill_between(
        x[index],
        mean_pref_return_array[index] - std_pref_return_array[index],
        mean_pref_return_array[index] + std_pref_return_array[index],
        alpha=.3,
        lw=0,
        color=color)
    axs[2].set_ylabel('Episode Preference Return', fontsize='x-large')
    axs[2].set_xlabel('Environment Steps', fontsize='x-large')
    for i in range(3):
        axs[i].set_facecolor('white')
        axs[i].spines['bottom'].set_color('black')
        axs[i].spines['bottom'].set_linewidth(3.0)
        axs[i].spines['left'].set_color('black')
        axs[i].spines['left'].set_linewidth(3.0)
        axs[i].spines['top'].set_color('black')
        axs[i].spines['top'].set_linewidth(3.0)
        axs[i].spines['right'].set_color('black')
        axs[i].spines['right'].set_linewidth(3.0)
        axs[i].grid(color='black', axis='y', ls='--')


if __name__ == '__main__':
    on_policy = True
    #on_policy = False
    if on_policy:
        path_list = [
            'test', 'PrefPPO', 'DecoupledPrefPPO_8_mistake', 'DecoupledPrefPPO_8'
        ]
        labels = ['PPO', 'PrefPPO', 'Decoupled PrefPPO', 'CG-PrefPPO']
        # PPO_total_timesteps 8000000; SAC_total_timesteps 4000000
        total_timesteps = 10000000
        # PPO_n_envs 16; SAC_n_envs 1
        n_envs = 16
        n_steps = 200
        color = ['blue', 'orange', 'green', 'red', 'darkorchid', 'salmon']

        fig, axs = plt.subplots(1, 3, figsize=(30, 4))
        # plt.rcParams["font.weight"] = "bold"
        # plt.rcParams["axes.labelweight"] = "bold"
        for i in range(len(path_list)):
            path_list[i] = 'D:\\+MyWork\\logs\\' + path_list[i]
            file_path_list = os.listdir(path_list[i])
            x, mean_total_return_array, std_total_return_array, mean_pref_return_array, std_pref_return_array, mean_task_success_array, std_task_success_array = read_csv_data(
                path_list[i],
                file_path_list,
                total_timesteps,
                n_envs,
                n_steps,
                smooth=5)
            plot_curve(axs, x, mean_total_return_array, std_total_return_array,
                       mean_pref_return_array, std_pref_return_array,
                       mean_task_success_array, std_task_success_array,
                       color[i], labels[i])

        fig.legend(fontsize='x-large',
                   loc='upper left',
                   bbox_to_anchor=(0.81, 0.51))
        plt.tight_layout()
        plt.savefig('on_policy.png')
        plt.show()
    else:
        path_list = ['test', 'PEBBLE', 'DecoupledPEBBLE', 'DecoupledPEBBLE_8']
        labels = ['SAC', 'PEBBLE', 'Decoupled PEBBLE', 'CG-PEBBLE']

        total_timesteps = 4000000
        n_envs = 1
        n_steps = 200
        color = [
            'royalblue', 'darkorange', 'seagreen', 'red', 'darkorchid',
            'salmon'
        ]

        fig, axs = plt.subplots(1, 3, figsize=(30, 4))

        for i in range(len(path_list)):
            path_list[i] = 'D:\\+MyWork\\logs\\' + path_list[i]
            file_path_list = os.listdir(path_list[i])
            x, mean_total_return_array, std_total_return_array, mean_pref_return_array, std_pref_return_array, mean_task_success_array, std_task_success_array = read_csv_data(
                path_list[i],
                file_path_list,
                total_timesteps,
                n_envs,
                n_steps,
                smooth=100)
            plot_curve(axs, x, mean_total_return_array, std_total_return_array,
                       mean_pref_return_array, std_pref_return_array,
                       mean_task_success_array, std_task_success_array,
                       color[i], labels[i])

        fig.legend(fontsize='x-large',
                   loc='upper left',
                   bbox_to_anchor=(0.81, 0.51))
        plt.tight_layout()
        plt.savefig('off_policy.png')
        plt.show()
