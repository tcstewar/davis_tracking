# Introduction

This dataset includes DAVIS240C events and concurrently captured images of a 36mm ball being rolled across the field of view.  The data covers a range of velocities from a fixed overhead view.  It also includes changes in direction (hitting the wall). There are 125 sample trajectories ranging from less than 1 second to 10 seconds.

To load and view the dataset, you can use the following code.  The `load_data` function is included in this repo.  This data is provided courtesy of Intel and ABR with no guarantees.

```
times, images, targets = davis_tracking.load_data('../dataset/retinaTest95.events',
                                                  dt=0.1,                  # time step between images
                                                  decay_time=0.01,         # low pass filter time constant
                                                  separate_channels=False, # do positive and negative spikes separately
                                                  saturation=10,           # clip values to this range
                                                  merge=1)                 # merge pixels together to reduce size


N = 5
plt.figure(figsize=(14,8))
indices = np.linspace(0, len(times)-1, N).astype(int)
for i, index in enumerate(indices):
    plt.subplot(1, N, i+1)
    plt.imshow(images[index], vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.axhline(targets[index,1], c='w', ls='--')
    plt.axvline(targets[index,0], c='w', ls='--')
    ```
