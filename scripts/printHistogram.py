import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

file = open("../data/round.txt", "r")

myList = []

for line in file:
    myList.append(line)
#print(myList)

numbers = myList[0].split(' ')

i=0
data=[]
for num in numbers:
    if(i%2):
        data.append(num)
    i=i+1
data.sort(reverse=1)
num_bins = 10

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(data, num_bins,normed=1)
#ax.bar(np.arange(len(data)),data)

# add a 'best fit' line
ax.set_xlabel('Times of appearance')
ax.set_ylabel('Percentage')
ax.set_title(r'Histogram of round1:')

# Tweak spacing to prevent clipping of ylabel
#fig.tight_layout()
fig.savefig('round.png')   # save the figure to file
plt.close(fig)    # close the figure
