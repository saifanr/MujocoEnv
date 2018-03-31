import numpy as np
import matplotlib.pyplot as plt
#DDPG
with open("ddpgSoccer.txt") as f:
    lines = f.readlines()

word = "rollout/return"
y = []
i = 0
for line in lines:
    if word in line:
        if(i%10 == 0):
            y1 = float(line.split()[3])
            y.append(y1)
            print(y)
        i = i+1

x = []
for i in range(1,len(y)+1):
    x.append(i)

#PPO
with open("ppo1Soccer.txt") as g:
    lines = g.readlines()

word = "EpRewMean"
y1 = []
j = 0
for line in lines:
    if word in line:
        if(j%5 == 0):
            z = float(line.split()[3])
            y1.append(z)
            print(z)
        j= j+1

x1 = []
for i in range(1,len(y1)+1):
    x1.append(i)

#PG
with open("pglogSoccer.txt") as h:
    lines = h.readlines()

word = "Average reward:"
y2 = []
j = 0
for line in lines:
    if word in line:
        if(j%2 == 0):
            z = float(line.split()[4])
            y2.append(z)
            print(z)
        j= j+1
y2 = y2[:-1]
x2 = []
for i in range(1,len(y2)+1):
    x2.append(i)
plt.plot(x,y, label = "DDPG")
plt.plot(x1,y1, label = "PPO")
plt.plot(x2,y2, label = "Vanilla Policy Gradient")
plt.ylabel("Reward")
plt.xlabel("Epoch")

plt.legend(loc=3)
plt.show()
