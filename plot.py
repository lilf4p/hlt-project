# plot a scatter plot of a list of data 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# set blue palette
X = [1,2,4,8]
Y = [86,85,78,85]

plt.scatter(X,Y, s=[i * 2 for i in Y])

# map size of points to y values

# add dotted line between points
plt.plot(X,Y, linestyle='dotted')

# plot all y values from 0 
plt.yticks(np.arange(0, 100, 5))

# set y axis to log scale
plt.yscale('linear')

# set ticks to bold
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

# set axis labels to bold
plt.xlabel('Number of Chunks', fontweight='bold', fontsize=12)
plt.ylabel('F1 (%)', fontweight='bold', fontsize=12)

plt.show()
