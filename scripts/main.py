import matplotlib.pyplot as plt
import numpy as np

v = np.array([3.0, 2.0, 4.0])

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.quiver(
    0,
    0,
    0,  # Starting coordinates
    v[0],
    v[1],
    v[2],  # Vector components
    length=1,
    normalize=False,  # Preserve the vector's magnitude
    color="blue",
)

ax.set(
    xlim=(-1, 5),
    ylim=(-1, 5),
    zlim=(-1, 5),
    xlabel="x",
    ylabel="y",
    zlabel="z",
    title="3D vector",
)
ax.set_box_aspect((1, 1, 1))  # Equal scale with these equal limits

plt.show()
