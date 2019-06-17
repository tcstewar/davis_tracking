import nengo
import nengo_dl
import numpy as np


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
        
    def make_input_layer(self, source_shape,
                         spatial_stride=(1, 1),
                         spatial_size=None):        
        if spatial_size is None:
            spatial_size = (source_shape[2], source_shape[1])

        with self.net:
            self.input = nengo.Node(None,
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
                    sp = nengo.Node(None, size_in=w*h*source_shape[0],
                                    label='[%d:%d,%d:%d]' % (j,j+h,i,i+w))
                    row.append([sp])            

                    indices = np.array((items[j:j+h][:,i:i+w]).flat)
                    all_indices = []
                    for q in range(source_shape[0]):
                        all_indices.extend(indices+q*source_shape[1]*source_shape[2])
                    
                    nengo.Connection(self.input[all_indices], sp)
                    i += spatial_stride[0]
                j += spatial_stride[1]
                layer.append(row)
            self.layers.append(layer)
            self.output_shapes.append((source_shape[0],
                                       spatial_size[0],
                                       spatial_size[1]))
            
    def make_middle_layer(self, n_features, n_parallel,
                          n_local, kernel_stride, kernel_size, padding='valid',
                          use_neurons=True, init=None):
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
                        for kk in range(n_local):
                            prev_k = prev_col[index%len(prev_col)]
                            conv = nengo.Convolution(n_features, prev_output_shape,
                                                     channels_last=False,
                                                     kernel_size=kernel_size,
                                                     padding=padding,
                                                     strides=kernel_stride,
                                                     init=init)
                            nengo.Connection(prev_k, ens_neurons, transform=conv)
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

