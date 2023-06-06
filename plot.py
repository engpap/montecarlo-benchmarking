import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Read the CSV file
data = pd.read_csv('./times.csv')

# Filter the data for rows where Number_of_GPUs is 1
data = data[(data['Number_of_GPUs'] == 1)]# & (data['Total_number_of_options'] == 16777216)]

data['Version'] = pd.Categorical(data.Version, categories=['baseline', 'UMv0', 'UMv1', 'UMv2', 'UMv3'], ordered=True)
data.sort_values('Version')

# Define the baseline version
baseline_version = 'baseline'

# Filter the data for the baseline version
baseline_data = data[data['Version'] == baseline_version]

# Group the data by Parallelization_method and Problem_scaling
grouped_data = data.groupby(['Parallelization_method', 'Problem_scaling', 'Total_number_of_options'])

# Determine the number of subplots
#num_subplots = len(grouped_data)

# Create subplots
#fig, axes = plt.subplots(num_subplots, 2, figsize=(12, 8), sharey='row')
#fig.subplots_adjust(hspace=0.4)

colors = ['#3490dc', '#f6993f', '#38c172', '#ffed4a', '#f66d9b']
edge_size = 1

# Iterate over each group and create subplots
for i, (group, group_data) in enumerate(grouped_data):

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    # Calculate the average initialization and execution times for each version in the group
    avg_init_times = group_data.groupby(['Version'])['Init_time_ms'].mean()
    avg_exec_times = group_data.groupby(['Version'])['Execution_time_ms'].mean()

    baseline_group_data = baseline_data.loc[(baseline_data['Parallelization_method'] == group[0]) & (baseline_data['Problem_scaling'] == group[1]) & (baseline_data['Total_number_of_options'] == group[2])]
    # Calculate the ratios of each time value to the baseline time value
    init_time_ratios = avg_init_times / baseline_group_data['Init_time_ms'].mean()
    exec_time_ratios = avg_exec_times / baseline_group_data['Execution_time_ms'].mean()

    # Reset the index of the ratios
    init_time_ratios = init_time_ratios.reset_index()
    exec_time_ratios = exec_time_ratios.reset_index()

    #max_ratio = max(init_time_ratios['Init_time_ms'].max(), exec_time_ratios['Execution_time_ms'].max())

    # Plot initialization time ratios
    axes[0].bar(init_time_ratios['Version'], init_time_ratios['Init_time_ms'], color=colors, edgecolor='black', linewidth=edge_size)
    axes[0].axhline(y=1, color='red', linestyle='--')
    axes[0].set_xlabel('Version')
    axes[0].set_ylabel('Initialization Time Ratio')
    axes[0].set_title(f'Initialization Time Comparison - {group[0]}, {group[1]}, {group[2]}')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylim(0,1.05)
    axes[0].set_yticks([x for x in np.arange(0, 1.05, 0.1)])

    # Plotecution time ratios
    axes[1].bar(exec_time_ratios['Version'], exec_time_ratios['Execution_time_ms'], color=colors, edgecolor='black', linewidth=edge_size)
    axes[1].axhline(y=1, color='red', linestyle='--')
    axes[1].set_xlabel('Version')
    axes[1].set_ylabel('Execution Time Ratio')
    axes[1].set_title(f'Execution Time Comparison - {group[0]}, {group[1]}, {group[2]}x')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0.95,1.05)
    axes[1].set_yticks([x for x in np.arange(0.95, 1.05, 0.01)])

    # Show the plot in a separate window
    plt.show()

#plt.show()