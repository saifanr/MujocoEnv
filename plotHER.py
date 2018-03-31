import numpy as np
import matplotlib.pyplot as plt
#HER1
with open("logFetchReachHer.txt") as f:
    lines = f.readlines()

word = "test/success_rate"
y = [0]
i = 0
for line in lines:
    if word in line:
        if(i%1 == 0):
            y1 = float(line.split()[3])
            y.append(y1)
            print(y)
        i = i+1

x = []
for i in range(0,len(y)):
    x.append(i)

#HER2
with open("logFetchPickandPlaceHer.txt") as g:
    lines = g.readlines()

word = "test/success_rate"
y1 = [0]
j = 0
for line in lines:
    if word in line:
        if(j%1 == 0):
            z = float(line.split()[3])
            y1.append(z)
            print(z)
        j= j+1

x1 = []
for i in range(0,len(y1)):
    x1.append(i)






plt.plot(x,y, label = "Fetch and Reach")
plt.plot(x1,y1, label = "Pick and Place")
plt.ylabel("Success Rate")
plt.xlabel("Epoch")

plt.xlim(xmin=0)

plt.legend(loc=4)
plt.show()
