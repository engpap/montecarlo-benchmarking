import matplotlib.pyplot as plt
import numpy as np
import os 
from collections import defaultdict

# set width of bar
barWidth = 0.3

data = defaultdict(list)
GPUs = set()


# open file and save data in dictionary
cur_path = os.path.dirname(__file__)

with open(os.path.dirname(__file__) + '/../../result/res_scale-up.txt') as f:
    for line in f:
        line = line.split(',')
        data[line[2]].append(float(line[-1][:-2]))    
        GPUs.add(line[3])

GPUs = list(GPUs)

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
ax.set_xticks([r + barWidth/2 for r in range(len(strong))], GPUs)
ax.set_yticks(np.arange(0, 2.1, 0.5))

# create horizontal dotted lines for better comparison
for tick in np.arange(0, 2.1, 0.5):
        ax.axhline(y=tick, color='black', linestyle=':')

# add margins for y axis
ax.margins(y=0.05)      

# plot legend
ax.legend(loc='upper right')

# show plot
#plt.show()

# save plot?
