# plot a scatter plot of a list of data 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# set blue palette
X = [1,2,4,8]
Y = [86,85,78,85]

plt.plot(X,Y, marker='o')

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
