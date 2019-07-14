# Import the library
import SimpSOM as sps
import random
import numpy as np

data = []
for x in range(1600):
    data.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    # print(data)
raw_data =np.asarray(data)
print('The 40X40 raw data is generated.')

# Build a network 40x40 with a weights format taken from the raw_data and activate Periodic Boundary Conditions.
net = sps.somNet(40, 40, raw_data, PBC=True)

# Train the network for 10000 epochs and with initial learning rate of 0.01.
net.train(0.01, 10000)

# Save the weights to file
net.save('filename_weights')

# Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
# and repeat that for the other 2 nodes
# and then according to the distance between each node and its neighbours.

net.nodes_graph(colnum=0)
net.nodes_graph(colnum=1)
net.nodes_graph(colnum=2)
net.diff_graph()

# Project the datapoints on the new 2D network map.
net.project(raw_data)

# Cluster the datapoints according to the Quality Threshold algorithm.
net.cluster(raw_data, type='qthresh')
