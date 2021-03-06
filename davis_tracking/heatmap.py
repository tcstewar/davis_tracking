import numpy as np

def make_heatmap(targets, merge=1, size=1, strip_edges=0):
    shape = (240//merge, 180//merge)
    grid_x = np.linspace(0, 240//merge, shape[0]*2+1)[1::2]
    grid_y = np.linspace(0, 180//merge, shape[1]*2+1)[1::2]
    if strip_edges > 0:
        grid_x = grid_x[strip_edges:-strip_edges]
        grid_y = grid_y[strip_edges:-strip_edges]
    xx, yy = np.meshgrid(grid_x, grid_y)
    
    heat = np.zeros((targets.shape[0], shape[1]-strip_edges*2, shape[0]-strip_edges*2))
    for i, (x, y, r, v) in enumerate(targets):
        heat[i] = np.exp(-((y-yy)**2+(x-xx)**2)/(2*(size*r/2)**2))
    return heat
    
def find_peak(image):
    peak = np.argmax(image)
    x = peak % image.shape[-1]
    y = peak // image.shape[-1]
    return x+0.5, y+0.5