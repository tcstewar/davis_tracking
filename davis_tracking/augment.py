import numpy as np
       

def augment(inputs, targets, separate_channels):
    width = inputs.shape[2]
    height = inputs.shape[1]
    
    if separate_channels:
        height = height // 2
        inputs = inputs.reshape(-1, height, width)

    inputs_flip_lr = inputs[:,::-1,:]
    targets_flip_lr = np.array(targets, copy=True)
    targets_flip_lr[:,1] = height - targets_flip_lr[:,1]

    inputs_flip_ud = inputs[:,:,::-1]
    targets_flip_ud = np.array(targets, copy=True)
    targets_flip_ud[:,0] = width - targets_flip_ud[:,0]

    inputs_flip_both = inputs[:,::-1,:]
    inputs_flip_both = inputs_flip_both[:,:,::-1]
    targets_flip_both = np.array(targets, copy=True)
    targets_flip_both[:,1] = height - targets_flip_both[:,1]
    targets_flip_both[:,0] = width - targets_flip_both[:,0]

    inputs_aug = np.vstack([inputs,
                            inputs_flip_lr,
                            inputs_flip_ud,
                            inputs_flip_both])
    targets_aug = np.vstack([targets,
                             targets_flip_lr,
                             targets_flip_ud,
                             targets_flip_both])
    
    if separate_channels:
        inputs_aug = inputs_aug.reshape(-1, height*2, width)

    return inputs_aug, targets_aug
