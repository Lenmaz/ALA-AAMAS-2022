from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'arial',
        'size'   : 28}

matplotlib.rc('font', **font)

episodes = 10000

points = np.load('example.npy')
x_0 = points[:,0]
x_E = points[:,1]

for i in range(10):
    x_0[i] = -20


mean_scores0 = list()
scores0 = deque(maxlen=250)
for point in x_0:
    scores0.append(point)
    mean_score = [np.mean(scores0)]
    mean_scores0.append(mean_score)

mean_scoresE = list()
scoresE = deque(maxlen=250)
for point in x_E:
    scoresE.append(point)
    mean_score = [np.mean(scoresE)]
    mean_scoresE.append(mean_score)


fig, ax = plt.subplots()
plt.ylim([-21, 20])
plt.axhline(y=(15+18)*0.5, color='red')
ax.axhspan((15+18)*0.5, (17+18)*0.5, facecolor='red', alpha=0.2)
plt.axhline(y=(17+18)*0.5, color='red', ls='--')
plt.axhline(y=5, color='green')
plt.axhline(y=-5, color='green', ls='--')
ax.axhspan(5, -5, facecolor='green', alpha=0.2)
plt.plot(range(episodes), mean_scores0[:episodes], markersize=10, marker='s', markevery=500, label="Individual rewards $R_0$", color='red') # plotting t, a separately
plt.plot(range(episodes), mean_scoresE[:episodes], markersize=10, marker='^', markevery=500, label="Ethical rewards $R_N + R_E$", color='green')

plt.yticks([-20, -16, -5, 5, 16, 20])
ax.set(xlabel='Episode', ylabel='Sum of rewards per episode')
plt.legend(loc='best')
plt.show()