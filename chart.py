import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from recog import grap

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

all_res = []
def animate(i):
    res =  {'angry': 0, 'disgust': 0, 'fear': 2, 'happy': 1, 'sad': 2, 'surprise': 0, 'neutral': 5}
    res = grap()
    labels  = res.keys()
    print(labels)

    if len(all_res) >= 10:
        all_res.pop(0)
    
    all_res.append(res)

    vstack=np.empty((len(labels), len(all_res)), int)

    for idx, item in enumerate(all_res):
        for idx2, key in enumerate(labels):
            vstack[idx2][idx] = item[key]


    x = list(range(1, len(all_res)+1))
    y = vstack
    print(x)
    print(y)

   
    print("update")
    ax1.clear()

    ax1.stackplot(x, y, labels=labels)
    ax1.legend(loc='upper left')

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()