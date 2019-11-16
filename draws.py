import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
sns.set(color_codes=True)
import os
import pandas as pd
import numpy as np
import ast


def get_data(log_dirs, name_func=None, smooth_keys=[], min_length=False):
    split_char = '_'
    df = pd.DataFrame()
    row_nums = {}
    dfs = []

    for log_dir in log_dirs:
        for f_name in os.listdir(log_dir):
            if name_func is not None and not name_func(f_name):
                continue
            name_list = f_name.split(split_char)
            if 'seed' in name_list[-1]:
                base_name = split_char.join(name_list[:-1])
                file_path = os.path.join(log_dir, f_name, 'progress.csv')
                #             print('start:{}'.format(file_path))
                try:
                    df_tmp = pd.read_csv(file_path)
                except:
                    continue
                row_num = df_tmp.shape[0]
                row_nums[f_name] = row_num
                df_tmp['base_name'] = [base_name] * row_num
                df_tmp['full_name'] = [f_name] * row_num

                for k in smooth_keys:
                    if k in df_tmp.keys():
                        df_tmp[k] = df_tmp[k].rolling(25).mean()

                #         df = pd.concat([df, df_tmp], sort=False)
                dfs.append(df_tmp)
                print('Done:{},rows:{}'.format(file_path, row_num))

    min_row_num = min(row_nums.values())
    for d in dfs:
        if min_length:
            df = pd.concat([df, d.head(min_row_num)], ignore_index=True, sort=False)
        else:
            df = pd.concat([df, d], ignore_index=True, sort=False)

    print('rows:', min_row_num)

    return df


def draw(df, keys, name):
    for k in keys:
        fig = plt.figure()
        fig.set_tight_layout(True)
        fig.set_size_inches(14, 5)

        # ax = sns.lineplot(x='n_updates', y='eprew', hue='base_name',ci=80, n_boot=24, data=df)
        #         ax = sns.lineplot(x='n_updates', y=k, hue='base_name', data=df)
        ax = sns.lineplot(x='n_updates', y=k, hue='base_name', ci=40, data=df, n_boot=1)
        ax.set_title(name)
        handles, labels = ax.get_legend_handles_labels()
        print(labels)


log_dirs = [
    # 'logs/mtzm',
    # 'logs/mtzm/l2nor'
    'logs/gravitar',
    'logs/gravitar/l2nor',
    'logs/gravitar/sdnor',
]


def name_func(name):
    # return 'env32' in name
    # return 'env32' in name and ('abc0.5_' in name or 'abc1_' in name)
    # return 'l2nor_allcol0.01_abc1_arrayAction1_env32' in name or 'l2nor_abc1_arrayAction0_env32' in name or 'rnd_baselineCnn_env32' in name
    # return 'cnn_abc1_l2nor_arrayAction1_allcol0.01_env32' in name or 'cnn_abc1_l2nor_arrayAction0_env32' in name or 'cnn_abc0.1_arrayAction0_env32' in name
    # return 'rnd_baselineCnn_env32' in name or 'l2nor_allcol0.01_abc1_arrayAction1_env32' in name or 'l2nor_abc1_arrayAction0_env32' in name or 'abc0.5_' in name
    return 'l2nor_allcol0.01_abc1_arrayAction1_env32' in name or 'sdnor_allcol0.01_abc0.01_' in name

keys = []
keys.extend(['eprew'])
# keys.extend(['loss/entropy_ab', 'loss/auxloss_ab'])
# keys.extend(['loss/entropy_ab'])
# for ld in log_dirs:
#     if 'mtzm' in ld:
#         keys.extend(['n_rooms', 'eprooms'])

df_data = get_data(log_dirs, name_func, smooth_keys=['loss/entropy_ab', 'loss/auxloss_ab', 'eprew'], min_length=False)
draw(df_data, keys=list(set(keys)), name=log_dirs[0].split('/')[1])
plt.show()
