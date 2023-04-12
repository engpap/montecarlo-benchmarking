import matplotlib.pyplot as plt
import numpy as np
import os 
from collections import defaultdict

# set width of bar
barWidth = 0.3

data = defaultdict(list)

'''
# need to check on vm actual output file name
GPUs = ['1', '2', '4']
scaling = ['strong', 'weak']

# export data from results folder
for g in GPUs:
    for s in scaling:
        # open file and save data in relative list
        cur_path = os.path.dirname(__file__)

        f = open(os.path.dirname(__file__) + '/../results/res' + g + 'g_' + s + '.txt')

        line = f.readline()

        line = line.split(',')

        data[s].append(line[-1])

'''
# initialize data (hardcoded for now)
data['strong'] = [1061.562012, 545.804016, 0]
data['weak'] = [1051.198975, 1061.215942, 0]

strong = data['strong']
weak = data['weak']

normalized_strong = []
normalized_weak = []

# normalize strong data
base_strong = strong[0]
for time in strong:
    normalized_strong.append(time/base_strong)

# normalize weak data
base_weak = weak[0]
for time in weak:
    normalized_weak.append(time/base_weak)

# generate subplot
fig, ax = plt.subplots(figsize = (8, 5))
 
# set strong and weak x position 
br1 = np.arange(len(strong))
br2 = [x + barWidth for x in br1]

# create the two bar plots
ax.bar(br1, normalized_strong, color = 'tab:red', edgecolor ='black', 
        width = barWidth, label='Strong')

ax.bar(br2, normalized_weak, color = 'tab:blue', edgecolor ='black', 
        width = barWidth, label='Weak')

# set secondary ylabel for GPU type
ax2 = ax.twinx()
ax2.set_ylabel("GPU(s): Tesla V100-SXM2-16GB", fontweight ='bold', fontsize = 10)
ax2.set_yticks([])

# set xlabel and ylabel
ax.set_xlabel("Number of GPUs", fontweight ='bold', fontsize = 10)
ax.set_ylabel("Normalized Latency", fontweight ='bold', fontsize = 10)

# set title
plt.title("Normalized Latency Reduction", fontweight ='bold', fontsize = 15)

# set xticks and yticks
ax.set_xticks([r + barWidth/2 for r in range(len(strong))], ['1', '2', '4'])
ax.set_yticks(np.arange(0, 2.1, 0.5))

# create horizontal dotted lines for better comparison
for tick in np.arange(0, 2.1, 0.5):
        ax.axhline(y=tick, color='black', linestyle=':')

# add margins for y axis
ax.margins(y=0.05)      

# plot legend
ax.legend(loc='upper right')

# show plot
plt.show()

# save plot?
