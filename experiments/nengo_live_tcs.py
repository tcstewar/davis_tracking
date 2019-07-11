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
from nengo_loihi.inputs import DVSFileChipNode  # branch: dvs-live
from nengo_loihi.hardware.allocators import GreedyChip

from pytry.read import text

import davis_tracking
print(davis_tracking.__file__)


# model file generated via:
# repo: https://github.com/tcstewar/davis_tracking/
# branch: nengo_spiking
# pytry track_deep_heat_spiking.py --n_epochs=200 --n_data=-1 --test_set=one --save_params=True --seed=0 --saturation=1000000000 --n_features_1=3 --n_parallel=1 --max_rate=8000

trained = os.path.abspath("data/TrackingTrial#20190621-135559-32d7c7d3")
pdict = text("%s.txt" % trained)
p = namedtuple("Params", pdict.keys())(*pdict.values())
params = np.load("%s.params.npy" % trained, allow_pickle=True)

dt = 0.001

strip_edges = 3  #  the number of edge pixels to remove due to convolution
assert p.separate_channels
shape = (2, 180//p.merge, 240//p.merge)
output_shape = shape[1]-strip_edges*2, shape[2]-strip_edges*2


# this is a hack to handle the fact that the CNN
# was trained with filtered events while
# spikes are implicitly scaled by 1/dt
scale_input_by_dt = True
tau_probe = 0.02

with nengo.Network() as model:
    #inp = DVSChipNode(pool=(p.merge, p.merge), channels_last=False)
    testfile = '../dataset/retinaTest95.events'
    inp = DVSFileChipNode(filename=testfile, t_start=1.7, pool=(p.merge, p.merge), channels_last=False)

    '''
    h = inp.height
    w = inp.width

    e = nengo.Ensemble(
        h * w //2, 1,
        neuron_type=nengo.SpikingRectifiedLinear(),
        gain=nengo.dists.Choice([0.1]),
        bias=nengo.dists.Choice([0]),
    )
    '''

    #nengo.Connection(inp[::4], e.neurons, synapse=0.01)
    #probe = nengo.Probe(e.neurons)
    #inp = nengo.Node(output=nengo.processes.PresentInput(
    #    images.reshape(images.shape[0], -1), dt))

    #p_inp = nengo.Probe(inp, synapse=p.decay_time)

    # force `out` to run on the host so that the merged
    # output is stapled together on the host, and then
    # use_neurons for the final layer because nengo_loihi doesn't
    # currently support probing a convolutional tranform
    #out = nengo.Node(lambda t, x: x, size_in=np.prod(output_shape))

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
    scale = 0.1
    convnet.make_middle_layer(n_features=1, n_parallel=p.n_parallel, n_local=1,
                              kernel_stride=(1,1), kernel_size=(3,3), init=init*scale, use_neurons=True)
    

    #out = nengo.Node(lambda t, x: x, size_in=1)
    #nengo.Connection(convnet.layers[-1][0][0][0], out, transform=np.ones((1, 36)), synapse=None)
    #nengo.Probe(out)
    #nengo.Probe(convnet.layers[-1][0][0][0])
    #convnet.make_merged_output(output_shape)
    

    #probes = []
    #for ensemble in model.all_ensembles:
    #    probes.append(
    #        nengo.Probe(ensemble.neurons, synapse=None))

    #p_out = nengo.Probe(convnet.output, synapse=tau_probe)

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
            #assert ens.neuron_type.amplitude == 1 / p.max_rate, ens.neuron_type.amplitude
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

t_step = 3.0
import timeit
now = None


with nengo_loihi.Simulator(model, dt=dt, 
                           #precompute=True, 
                           #target="loihi",
                           target="sim",
                           hardware_options={'allocator': GreedyChip(2)}
                           ) as sim:

    running = True
    while running:
            sim.run(t_step)
            if now is None:
                now = timeit.default_timer()
            else:
                t2 = timeit.default_timer()
                print('Rate:', t_step/(t2-now))
                now = t2
            1/0

