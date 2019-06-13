import pytry
import os
import random
import nengo
import nengo_extras
import numpy as np
import nengo_dl
import tensorflow as tf

import davis_tracking


class TrackingTrial(pytry.PlotTrial):
    def params(self):
        self.param('number of data sets to use', n_data=-1)
        self.param('data directory', dataset_dir=r'../dataset')
        self.param('dt', dt=0.1)
        self.param('dt_test', dt_test=0.001)
        self.param('decay time (input synapse)', decay_time=0.01)
        self.param('test set (odd|one)', test_set='one')
        self.param('augment training set with flips', augment=False)
        self.param('miniback size', minibatch_size=200)
        self.param('learning rate', learning_rate=1e-3)
        self.param('number of epochs', n_epochs=5)
        self.param('saturation', saturation=5)
        self.param('separate positive and negative channels', separate_channels=True)
        self.param('number of features in layer 1', n_features_1=28)
        self.param('number of features in layer 2', n_features_2=64)
        self.param('kernel size layer 1', kernel_size_1=5)
        self.param('stride layer 1', stride_1=3)
        self.param('kernel size layer 2', kernel_size_2=3)
        self.param('stride layer 2', stride_2=1)
        self.param('kernel size layer 3', kernel_size_3=3)
        self.param('split spatial configuration', split_spatial=True)
        self.param('spatial stride', spatial_stride=10)
        self.param('spatial kernel size', spatial_size=20)
        self.param('number of parallel ensembles', n_parallel=2)
        self.param('merge pixels (to make a smaller image)', merge=3)
        self.param('normalize inputs', normalize=False)
        
        
    def evaluate(self, p, plt):
        files = []
        sets = []
        for f in os.listdir(p.dataset_dir):
            if f.endswith('events'):
                files.append(os.path.join(p.dataset_dir, f))

        if p.test_set == 'one':
            test_file = random.sample(files, 1)[0]
            files.remove(test_file)
        
        if p.n_data != -1:
            files = random.sample(files, p.n_data)
            
        inputs = []
        targets = []
        for f in files:
            times, imgs, targs = davis_tracking.load_data(f, dt=p.dt, decay_time=p.decay_time,
                                                  separate_channels=p.separate_channels, 
                                                  saturation=p.saturation, merge=p.merge)
            inputs.append(imgs)
            targets.append(davis_tracking.make_heatmap(targs, merge=p.merge).reshape(len(targs),-1))
                                
        inputs_all = np.vstack(inputs)
        targets_all = np.vstack(targets)
        
        if p.test_set == 'odd':
            inputs_train = inputs_all[::2]
            inputs_test = inputs_all[1::2]
            targets_train = targets_all[::2]
            targets_test = targets_all[1::2]
            dt_test = p.dt*2
        elif p.test_set == 'one':
            times, imgs, targs = davis_tracking.load_data(test_file, dt=p.dt_test, decay_time=p.decay_time,
                                                  separate_channels=p.separate_channels, 
                                                  saturation=p.saturation, merge=p.merge)
            inputs_test = imgs

            targets_test = davis_tracking.make_heatmap(targs, merge=p.merge).reshape(len(targs), -1)
            inputs_train = inputs_all
            targets_train = targets_all
            dt_test = p.dt_test
            
        if p.separate_channels:
            shape = (2, 180//p.merge, 240//p.merge)
        else:
            shape = (1, 180//p.merge, 240//p.merge)
        
        dimensions = shape[0]*shape[1]*shape[2]

        
        if p.normalize:
            magnitude = np.linalg.norm(inputs_train.reshape(-1, dimensions), axis=1)
            inputs_train = inputs_train*(1.0/magnitude[:,None,None])
            
            magnitude = np.linalg.norm(inputs_test.reshape(-1, dimensions), axis=1)
            inputs_test = inputs_test*(1.0/magnitude[:,None,None])


                    
        
        
        max_rate = 100
        amp = 1 / max_rate

        model = nengo.Network()
        with model:
            model.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear(amplitude=amp)
            model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
            model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
            model.config[nengo.Connection].synapse = None

            inp = nengo.Node(
                nengo.processes.PresentInput(inputs_test.reshape(-1, dimensions), dt_test),
                size_out=dimensions,
                )

            out = nengo.Node(None, size_in=targets_train.shape[-1])
            
            if not p.split_spatial:
                # do a standard convnet
                conv1 = nengo.Convolution(p.n_features_1, shape, channels_last=False, strides=(p.stride_1,p.stride_1),
                                          padding='same',
                                          kernel_size=(p.kernel_size_1, p.kernel_size_1))
                layer1 = nengo.Ensemble(conv1.output_shape.size, dimensions=1)
                nengo.Connection(inp, layer1.neurons, transform=conv1)

                conv2 = nengo.Convolution(p.n_features_2, conv1.output_shape, channels_last=False, strides=(p.stride_2,p.stride_2),
                                          padding='same',
                                          kernel_size=(p.kernel_size_2, p.kernel_size_2))
                layer2 = nengo.Ensemble(conv2.output_shape.size, dimensions=1)
                nengo.Connection(layer1.neurons, layer2.neurons, transform=conv2)

                conv3 = nengo.Convolution(1, conv2.output_shape, channels_last=False, strides=(1,1),
                                          padding='same',
                                          kernel_size=(p.kernel_size_3, p.kernel_size_3))

                nengo.Connection(layer2.neurons, out, transform=conv3)
            else:
                assert False
                # do the weird spatially split convnet
                convnet = davis_tracking.ConvNet(nengo.Network())
                convnet.make_input_layer(
                        shape,
                        spatial_stride=(p.spatial_stride, p.spatial_stride), 
                        spatial_size=(p.spatial_size,p.spatial_size))
                nengo.Connection(inp, convnet.input)
                convnet.make_middle_layer(n_features=p.n_features_1, n_parallel=p.n_parallel, n_local=1,
                                          kernel_stride=(p.stride_1,p.stride_1), kernel_size=(p.kernel_size_1,p.kernel_size_1))
                convnet.make_middle_layer(n_features=p.n_features_2, n_parallel=p.n_parallel, n_local=1,
                                          kernel_stride=(p.stride_2,p.stride_2), kernel_size=(p.kernel_size_2,p.kernel_size_2))
                convnet.make_output_layer(2)
                nengo.Connection(convnet.output, out)
                         

            p_out = nengo.Probe(out)


        N = len(inputs_train)
        n_steps = int(np.ceil(N/p.minibatch_size))
        dl_train_data = {inp: np.resize(inputs_train, (p.minibatch_size, n_steps, dimensions)),
                         p_out: np.resize(targets_train, (p.minibatch_size, n_steps, targets_train.shape[-1]))}
        N = len(inputs_test)
        n_steps = int(np.ceil(N/p.minibatch_size))
        dl_test_data = {inp: np.resize(inputs_test, (p.minibatch_size, n_steps, dimensions)),
                        p_out: np.resize(targets_test, (p.minibatch_size, n_steps, targets_train.shape[-1]))}
        with nengo_dl.Simulator(model, minibatch_size=p.minibatch_size) as sim:
            #loss_pre = sim.loss(dl_test_data)

            if p.n_epochs > 0:
                sim.train(dl_train_data, tf.train.RMSPropOptimizer(learning_rate=p.learning_rate),
                          n_epochs=p.n_epochs)

            loss_post = sim.loss(dl_test_data)

            sim.run_steps(n_steps, data=dl_test_data)

        data = sim.data[p_out].reshape(-1,targets_train.shape[-1])[:len(targets_test)]

        data_peak = np.array([davis_tracking.find_peak(d.reshape(shape[1:])) for d in data])
        target_peak = np.array([davis_tracking.find_peak(d.reshape(shape[1:])) for d in targets_test])


        
        rmse_test = np.sqrt(np.mean((target_peak-data_peak)**2, axis=0))*p.merge          
        if plt:
            plt.plot(data_peak*p.merge)
            plt.plot(target_peak*p.merge, ls='--')
            
        return dict(
            rmse_test = rmse_test,
            max_n_neurons = max([ens.n_neurons for ens in model.all_ensembles]),
            test_targets = targets_test,
            test_output = data,
            target_peak = target_peak,
            data_peak = data_peak,
            test_loss = loss_post,
            )
