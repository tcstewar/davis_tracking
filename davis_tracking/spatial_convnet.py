import nengo
import nengo_dl
import numpy as np


def conv_to_dense(conv):
    dense = np.empty((conv.size_out, conv.size_in))
    
    dt = 1
    with nengo.Network(add_to_container=False) as model:
        stim = nengo.Node(output=nengo.processes.PresentInput(
            np.eye(conv.size_in), dt))
        x = nengo.Node(size_in=conv.size_out)
        nengo.Connection(stim, x, transform=conv, synapse=None)
        p = nengo.Probe(x, synapse=None)
    
    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        sim.run_steps(conv.size_in, progress_bar=False)

    return sim.data[p].T


class ConvNet(object):
    def __init__(self, net, max_rate=100):
        amp = 1 / max_rate
        net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear(amplitude=amp)
        net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
        net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        net.config[nengo.Connection].synapse = None    
        self.net = net
        self.layers = []
        self.output_shapes = []
        self.input = None
        
    def make_input_layer(self, source_shape,
                         spatial_stride=(1, 1),
                         spatial_size=None,
                         use_separate_nodes=False):        
        if spatial_size is None:
            spatial_size = (source_shape[2], source_shape[1])

        with self.net:
            if self.input is None:
                self.input = nengo.Node(
                    None,
                    size_in=source_shape[0]*source_shape[1]*source_shape[2],
                    label='input')

            j = 0
            w = spatial_size[0]
            h = spatial_size[1]
            items = np.arange(source_shape[1]*source_shape[2])
            items.shape = source_shape[1:]
            layer = []
            while j + h <= source_shape[1]:
                row = []
                i = 0
                while i + w <= source_shape[2]:
                    if use_separate_nodes:
                        sp = nengo.Node(None, size_in=w*h*source_shape[0],
                                    label='[%d:%d,%d:%d]' % (j,j+h,i,i+w))
                        row.append([sp])            

                    indices = np.array((items[j:j+h][:,i:i+w]).flat)
                    all_indices = []
                    for q in range(source_shape[0]):
                        all_indices.extend(indices+q*source_shape[1]*source_shape[2])
                    
                    if use_separate_nodes:
                        nengo.Connection(self.input[all_indices], sp)
                    else:
                        row.append([self.input[all_indices]])

                    i += spatial_stride[0]
                j += spatial_stride[1]
                layer.append(row)
            self.layers.append(layer)
            self.output_shapes.append((source_shape[0],
                                       spatial_size[0],
                                       spatial_size[1]))
            
    def make_middle_layer(self, n_features, n_parallel,
                          n_local, kernel_stride, kernel_size, padding='valid',
                          use_neurons=True, init=nengo.dists.Uniform(-1,1)):
        with self.net:
            prev_layer = self.layers[-1]
            prev_output_shape = self.output_shapes[-1]
            layer = []
            for prev_row in prev_layer:
                row = []
                for prev_col in prev_row:
                    col = []
                    
                    index = 0
                    for k in range(n_parallel):
                        conv = nengo.Convolution(n_features, prev_output_shape,
                                                 channels_last=False,
                                                 kernel_size=kernel_size,
                                                 padding=padding,
                                                 strides=kernel_stride,
                                                 init=init)

                        if use_neurons:
                            ens = nengo.Ensemble(conv.output_shape.size, dimensions=1,
                                                 label='%s' % conv.output_shape)
                            ens_neurons = ens.neurons
                        else:
                            ens = nengo.Node(None, size_in=conv.output_shape.size,
                                             label='%s' % conv.output_shape)
                            ens_neurons = ens
                            
                        if len(self.layers) == 1:
                            # work-around for NxSDK limitation with convolutional
                            # weights in the first layer that have overlap in the
                            # input slices. this requires the number of features
                            # in the first layer to be fairly small so that
                            # T.size is approximately <= 300^2
                            T = conv_to_dense(conv)
                        else:
                            T = conv
                            
                        for kk in range(n_local):
                            prev_k = prev_col[index%len(prev_col)]
                            conv = nengo.Convolution(n_features, prev_output_shape,
                                                     channels_last=False,
                                                     kernel_size=kernel_size,
                                                     padding=padding,
                                                     strides=kernel_stride,
                                                     init=init)
                            nengo.Connection(prev_k, ens_neurons, transform=T)
                            index += 1
                        col.append(ens_neurons)
                    row.append(col)
                layer.append(row)
            self.layers.append(layer)
            self.output_shapes.append(conv.output_shape)
            
    def make_output_layer(self, dimensions):
        with self.net:
            self.output = nengo.Node(None, dimensions, label='output')
            for row in self.layers[-1]:
                for col in row:
                    for k in col:
                        nengo.Connection(k, self.output,
                                         transform=nengo_dl.dists.Glorot())
                        
    def make_merged_output(self, shape):
        with self.net:
            self.output = nengo.Node(None, size_in=shape[0]*shape[1], label='output')
            indices = np.arange(shape[0]*shape[1]).reshape(shape)

            count = np.zeros(self.output.size_out)

            patch_shape = self.output_shapes[-1].shape
            assert patch_shape[0] == 1
            i = 0
            j = 0
            for row in self.layers[-1]:
                for n in row:
                    assert len(n) == 1
                    n = n[0]
                    items = indices[j:j+patch_shape[2],i:i+patch_shape[1]]
                    nengo.Connection(n, self.output[items.flatten()])
                    count[items.flatten()] += 1
                    i += patch_shape[1]
                j += patch_shape[2]
                i = 0
            assert count.min() == count.max() == 1

