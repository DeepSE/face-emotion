import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from recog import screen_shot

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

all_res = []

MAX_RESULTS = 20 

def animate(i):
    # {'angry': 0, 'disgust': 0, 'fear': 2, 'happy': 1, 'sad': 2, 'surprise': 0, 'neutral': 5}
    res = screen_shot()
    labels  = res.keys()

    if len(all_res) >= MAX_RESULTS:
        all_res.pop(0)
    
    all_res.append(res)

    x = list(range(1, len(all_res)+1))
    y=np.empty((len(labels), len(all_res)), int)

    for idx, item in enumerate(all_res):
        for idx2, key in enumerate(labels):
            y[idx2][idx] = item[key]

    print("update")
    ax1.clear()

    ax1.stackplot(x, y, labels=labels)
    ax1.legend(loc='upper left')

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()