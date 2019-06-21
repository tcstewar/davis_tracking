from collections import namedtuple
import os
# os.environ['SLURM'] = '1'
# # os.environ['BOARD'] = 'loihimh2'
# os.environ['LMTOPTIONS'] = '--skip-power=1'

import matplotlib.pyplot as plt
import numpy as np

import nengo
import nengo_loihi
# from nengo_loihi.inputs import DVSFileChipNode  # branch: dvs-file-cores
from nengo_loihi.inputs import DVSChipNode  # branch: dvs-live
from nengo_loihi.hardware.allocators import GreedyChip

from pytry.read import text

import davis_tracking
print(davis_tracking.__file__)

import sdl2.ext


# model file generated via:
# repo: https://github.com/tcstewar/davis_tracking/
# branch: nengo_spiking
# pytry track_deep_heat_spiking.py --n_epochs=200 --n_data=-1 --test_set=one --save_params=True --seed=0 --saturation=1000000000 --n_features_1=3 --n_parallel=1 --max_rate=8000

trained = os.path.abspath("data/TrackingTrial#20190621-114844-871ad8af")
pdict = text("%s.txt" % trained)
p = namedtuple("Params", pdict.keys())(*pdict.values())
params = np.load("%s.params.npy" % trained, allow_pickle=True)

dt = 0.001

print(pdict)



# In[4]:
# testfile = '../dataset/retinaTest95.events'

# times, images, targets = davis_tracking.load_data(
#     testfile,
#     dt=dt,
#     decay_time=p.decay_time,
#     separate_channels=p.separate_channels,
#     saturation=p.saturation,
#     merge=p.merge)

# times.shape, images.shape, targets.shape


# In[5]:
# N = 10
# plt.figure(figsize=(14,8))
# indices = np.linspace(0, len(times)-1, N).astype(int)
# for i, index in enumerate(indices):
#     plt.subplot(1, N, i+1)
#     plt.imshow(images[index]) #, vmin=-1, vmax=1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.axhline(targets[index,1], c='w', ls='--')
#     plt.axhline(images.shape[1]//2, c='k')
#     plt.axhline(targets[index,1]+images.shape[1]//2, c='w', ls='--')
#     plt.axvline(targets[index,0], c='w', ls='--')

# plt.figure()
# plt.plot(times, targets)
# plt.legend(['x', 'y', 'radius', 'present'])

# plt.show()


# In[6]:

# copied over from track_deep_heat.py:
strip_edges = 3  #  the number of edge pixels to remove due to convolution
assert p.separate_channels
shape = (2, 180//p.merge, 240//p.merge)
output_shape = shape[1]-strip_edges*2, shape[2]-strip_edges*2


# In[10]:


# this is a hack to handle the fact that the CNN
# was trained with filtered events while
# spikes are implicitly scaled by 1/dt
scale_input_by_dt = True
tau_probe = 0.02

with nengo.Network() as model:
    # inp = DVSFileChipNode(filename=testfile, t_start=times[0], pool=(p.merge, p.merge),
    #                       channels_last=False)
    # assert inp.dt == dt

    inp = DVSChipNode(pool=(p.merge, p.merge), channels_last=False)

    #inp = nengo.Node(output=nengo.processes.PresentInput(
    #    images.reshape(images.shape[0], -1), dt))

    #p_inp = nengo.Probe(inp, synapse=p.decay_time)

    # force `out` to run on the host so that the merged
    # output is stapled together on the host, and then
    # use_neurons for the final layer because nengo_loihi doesn't
    # currently support probing a convolutional tranform
    out = nengo.Node(lambda t, x: x, size_in=np.prod(output_shape))

    convnet = davis_tracking.ConvNet(nengo.Network(), max_rate=p.max_rate)

    convnet.input = inp

    convnet.make_input_layer(
            shape,
            spatial_stride=(p.spatial_stride, p.spatial_stride),
            spatial_size=(p.spatial_size, p.spatial_size))

    #nengo.Connection(inp, convnet.input)

    init = params[2]['transform'].init #if params is not None else nengo.dists.Uniform(-1, 1)
    if scale_input_by_dt:
        init = init * dt
    convnet.make_middle_layer(n_features=p.n_features_1, n_parallel=p.n_parallel, n_local=1,
                              kernel_stride=(1,1), kernel_size=(3,3), init=init)
    init = params[3]['transform'].init #if params is not None else nengo.dists.Uniform(-1, 1)
    convnet.make_middle_layer(n_features=p.n_features_2, n_parallel=p.n_parallel, n_local=1,
                              kernel_stride=(1,1), kernel_size=(3,3), init=init)
    init = params[4]['transform'].init #if params is not None else nengo.dists.Uniform(-1, 1)
    convnet.make_middle_layer(n_features=1, n_parallel=p.n_parallel, n_local=1,
                              kernel_stride=(1,1), kernel_size=(3,3), init=init, use_neurons=True)
    convnet.make_merged_output(output_shape)
    nengo.Connection(convnet.output, out)

    #probes = []
    #for ensemble in model.all_ensembles:
    #    probes.append(
    #        nengo.Probe(ensemble.neurons, synapse=None))

    p_out = nengo.Probe(out, synapse=tau_probe)

    if params is not None:
        assert np.allclose(params[0]['gain'], p.max_rate, atol=1e-5)
        assert np.allclose(params[1]['gain'], p.max_rate, atol=1e-5)

        # Copied over from commit 909c38f5f35b1c31794d7cb8282eccb247746774
        def assign_bias(edge, full_bias, layer, n_features, stride):
            used = np.zeros_like(full_bias)
            start_x = 0
            start_y = 0
            w = p.spatial_size - edge
            h = p.spatial_size - edge
            full_w = shape[2] - edge
            full_h = shape[1] - edge
            assert len(full_bias) == full_w*full_h*n_features
            for row in layer:
                for patches in row:
                    for patch in patches:
                        bias = np.zeros(patch.size_out)
                        assert patch.size_out == w*h*n_features
                        for i in range(w):
                            for j in range(h):
                                for k in range(n_features):
                                    local_index = i + j*w + k*h*w
                                    full_index = (start_x + i) + (start_y+j)*full_w + k*(full_w*full_h)
                                    bias[local_index] = full_bias[full_index]
                                    used[full_index] += 1
                        patch.ensemble.bias = bias
                        patch.ensemble.gain = np.ones(patch.size_out)*100
                    start_x += stride
                start_x = 0
                start_y += stride

        # TODO: figure out why this isn't making things better:
        # assign_bias(edge=2, full_bias=params[0]['bias'], layer=convnet.layers[1], n_features=p.n_features_1, stride=p.spatial_stride)
        # assign_bias(edge=4, full_bias=params[1]['bias'], layer=convnet.layers[2], n_features=p.n_features_2, stride=p.spatial_stride)


# In[11]:


convert_to_spiking = True
add_synapses = 0.001  # p.decay_time

print("=====================")

if convert_to_spiking:
    n_ensembles = 0
    for ens in model.all_ensembles:
        if isinstance(ens.neuron_type, nengo.RectifiedLinear):
            assert ens.neuron_type.amplitude == 1 / p.max_rate, ens.neuron_type.amplitude
            n_ensembles += 1
            ens.neuron_type = nengo.SpikingRectifiedLinear(
                amplitude=ens.neuron_type.amplitude)
    print("Changed %d ensembles" % n_ensembles)

if add_synapses is not None:
    n_synapses = 0
    for conn in model.all_connections:
        if conn.synapse is None:
            n_synapses += 1
            conn.synapse = add_synapses
    print("Changed %d synapses" % n_synapses)

print("=====================")


height, width = output_shape
print("Width: %d, height %d" % (width, height))


def plot(neural_data, renderer):
    data = neural_data[-1].reshape(output_shape)
    data_peak = davis_tracking.find_peak(data)
    print("Data max: %0.1e, 95th: %0.1e, peak: %0.1f, %0.1f" % (
        data.max(), np.percentile(data, 95), data_peak[0], data_peak[1]))

    n_levels = 5
    levels = np.linspace(0, 0.08, n_levels)
    brightness = np.linspace(0, 1, n_levels)
    grey_color = lambda v: sdl2.ext.Color(r=v, g=v, b=v)
    colors = [grey_color(int(255 * b)) for b in brightness]

    renderer.clear(color=colors[0])
    for level, color in zip(levels[1:], colors[1:]):
        i, j = (data >= level).nonzero()

        x = j
        y = height - 1 - i
        renderer.draw_point([c for (xx, yy) in zip(x, y) for c in (xx, yy)], color=color)

    i, j = np.round(data_peak).astype(int)
    x = j
    y = height - 1 - i
    renderer.draw_point([x, y], color=sdl2.ext.Color(r=255, g=100, b=0))

    renderer.present()


t_step = 0.03

with nengo_loihi.Simulator(model, dt=dt, precompute=True, target="loihi",
                           hardware_options={'allocator': GreedyChip(2)}) as sim:
    sdl2.ext.init()
    scale = 5 * 4
    window = sdl2.ext.Window("DVS", size = (width * scale, height * scale))
    window.show()
    backgroundColor = sdl2.ext.RGBA(0x808080FF)
    offColor = sdl2.ext.RGBA(0x000000FF)
    onColor = sdl2.ext.RGBA(0xFFFFFFFF)
    renderer = sdl2.ext.Renderer(window)
    renderer.logical_size = (width, height)
    renderer.clear(color=backgroundColor)
    renderer.present()
    running = True
    while running:
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
        if running:
            sim.run(t_step)
            plot(sim.data[p_out], renderer)

sdl2.ext.quit()
