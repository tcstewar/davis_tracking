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
        self.param('number of features in layer 1', n_features_1=10)
        self.param('number of features in layer 2', n_features_2=10)
        self.param('split spatial configuration', split_spatial=False)
        self.param('spatial stride', spatial_stride=6)
        self.param('spatial kernel size', spatial_size=12)
        self.param('number of parallel ensembles', n_parallel=1)
        self.param('merge pixels (to make a smaller image)', merge=5)
        self.param('normalize inputs', normalize=False)
        self.param('save parameters', save_params=True)
        self.param('load parameters from a file', load_params_from='')
        self.param('use nengo (instead of nengo_dl)', use_nengo=False)
        self.param('use frame input (instead of events)', use_frames=False)
        
        
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

        if len(p.load_params_from) > 0:
            params = np.load(p.load_params_from, allow_pickle=True)
        else:
            params = None
            
        
            
        strip_edges = 3  #  the number of edge pixels to remove due to convolution
        
        inputs = []
        targets = []
        targets_raw = []
        for f in files:
            times, imgs, targs = davis_tracking.load_data(f, dt=p.dt, decay_time=p.decay_time,
                                                  separate_channels=p.separate_channels, 
                                                  saturation=p.saturation, merge=p.merge)
            if p.use_frames:
                times_frames, frames_raw = davis_tracking.load_frames(f.replace('.events', '.frame'), merge=p.merge)
                frames = []
                for t in times:
                    index = np.searchsorted(times_frames, t)
                    if index == len(times_frames):
                        index = -1
                    frames.append(frames_raw[index])
                imgs = np.array(frames)

            inputs.append(imgs)
            targets_raw.append(targs[:, :2])
            targets.append(davis_tracking.make_heatmap(targs, merge=p.merge, strip_edges=strip_edges).reshape(len(targs),-1))
                                
        inputs_all = np.vstack(inputs)
        targets_all = np.vstack(targets)
        targets_all_raw = np.vstack(targets_raw)
        
        if p.test_set == 'odd':
            inputs_train = inputs_all[::2]
            inputs_test = inputs_all[1::2]
            targets_train = targets_all[::2]
            targets_test = targets_all[1::2]
            targets_test_raw = targets_all_raw[1::2]
            dt_test = p.dt*2
        elif p.test_set == 'one':
            times, imgs, targs = davis_tracking.load_data(test_file, dt=p.dt_test, decay_time=p.decay_time,
                                                  separate_channels=p.separate_channels, 
                                                  saturation=p.saturation, merge=p.merge)
            if p.use_frames:
                times_frames, frames_raw = davis_tracking.load_frames(test_file.replace('.events', '.frame'), merge=p.merge)
                frames = []
                for t in times:
                    index = np.searchsorted(times_frames, t)
                    frames.append(frames_raw[index])
                imgs = np.array(frames)
            inputs_test = imgs

            targets_test_raw = targs
            targets_test = davis_tracking.make_heatmap(targs, merge=p.merge, strip_edges=strip_edges).reshape(len(targs), -1)
            inputs_train = inputs_all
            targets_train = targets_all
            dt_test = p.dt_test
            
        if p.separate_channels and not p.use_frames:
            shape = (2, 180//p.merge, 240//p.merge)
        else:
            shape = (1, 180//p.merge, 240//p.merge)
        output_shape = shape[1]-strip_edges*2, shape[2]-strip_edges*2
        
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
                init = params[2]['transform'].init if params is not None else nengo.dists.Uniform(-1, 1)
                conv1 = nengo.Convolution(p.n_features_1, shape, channels_last=False, strides=(1,1),
                                          padding='valid',
                                          kernel_size=(3,3),
                                          init=init)
                layer1 = nengo.Ensemble(conv1.output_shape.size, dimensions=1)
                p_layer1 = nengo.Probe(layer1.neurons)
                if params is not None:
                    layer1.gain = params[0]['gain']
                    layer1.bias = params[0]['bias']
                nengo.Connection(inp, layer1.neurons, transform=conv1)

                init = params[3]['transform'].init if params is not None else nengo.dists.Uniform(-1, 1)
                conv2 = nengo.Convolution(p.n_features_2, conv1.output_shape, channels_last=False, strides=(1,1),
                                          padding='valid',
                                          kernel_size=(3,3),
                                          init=init)
                layer2 = nengo.Ensemble(conv2.output_shape.size, dimensions=1)
                p_layer2 = nengo.Probe(layer2.neurons)
                if params is not None:
                    layer2.gain = params[1]['gain']
                    layer2.bias = params[1]['bias']
                nengo.Connection(layer1.neurons, layer2.neurons, transform=conv2)

                init = params[4]['transform'].init if params is not None else nengo.dists.Uniform(-1, 1)
                conv3 = nengo.Convolution(1, conv2.output_shape, channels_last=False, strides=(1,1),
                                          padding='valid',
                                          kernel_size=(3,3),
                                          init=init)

                nengo.Connection(layer2.neurons, out, transform=conv3)
            else:
                # do the weird spatially split convnet
                convnet = davis_tracking.ConvNet(nengo.Network())
                convnet.make_input_layer(
                        shape,
                        spatial_stride=(p.spatial_stride, p.spatial_stride), 
                        spatial_size=(p.spatial_size,p.spatial_size))
                nengo.Connection(inp, convnet.input)
                init = params[2]['transform'].init if params is not None else nengo.dists.Uniform(-1, 1)
                convnet.make_middle_layer(n_features=p.n_features_1, n_parallel=p.n_parallel, n_local=1,
                                          kernel_stride=(1,1), kernel_size=(3,3), init=init)
                init = params[3]['transform'].init if params is not None else nengo.dists.Uniform(-1, 1)
                convnet.make_middle_layer(n_features=p.n_features_2, n_parallel=p.n_parallel, n_local=1,
                                          kernel_stride=(1,1), kernel_size=(3,3), init=init)
                init = params[4]['transform'].init if params is not None else nengo.dists.Uniform(-1, 1)
                convnet.make_middle_layer(n_features=1, n_parallel=p.n_parallel, n_local=1,
                                          kernel_stride=(1,1), kernel_size=(3,3), init=init, use_neurons=False)
                convnet.make_merged_output(output_shape)
                nengo.Connection(convnet.output, out)

                if params is not None:
                    assert np.allclose(params[0]['gain'], 100, atol=1e-5)
                    assert np.allclose(params[1]['gain'], 100, atol=1e-5)


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

                    assign_bias(edge=2, full_bias=params[0]['bias'], layer=convnet.layers[1], n_features=p.n_features_1, stride=p.spatial_stride)
                    assign_bias(edge=4, full_bias=params[1]['bias'], layer=convnet.layers[2], n_features=p.n_features_2, stride=p.spatial_stride)




            p_out = nengo.Probe(out)

        if p.use_nengo:
            #for ens in model.all_ensembles:
            #    ens.neuron_type = nengo.SpikingRectifiedLinear(amplitude=amp)
            with nengo.Simulator(model) as sim:
                sim.run(len(inputs_test)*dt_test)
            data = sim.data[p_out]
            

            data_peak = np.array([davis_tracking.find_peak(d.reshape(output_shape)) for d in data])
            target_peak = np.array([davis_tracking.find_peak(d.reshape(output_shape)) for d in targets_test])

            data_peak_f = nengo.synapses.Lowpass(0.01).filt(data_peak)
            data_peak_f_slices = data_peak_f[::int(dt_test/0.001)]

            rmse_test = np.sqrt(np.mean((target_peak-data_peak_f_slices)**2, axis=0))*p.merge          
            if plt:
                plt.subplot(1,1,1)
                plt.plot(data_peak_f_slices*p.merge)
                plt.plot(target_peak*p.merge, ls='--')
                plt.plot((targets_test_raw-strip_edges)*p.merge, ls=':')

                #plt.subplot(2,2,2)
                #plt.plot(np.mean(data, axis=0))

                '''
                plt.subplot(2,2,2)
                act = np.mean(sim.data[p_layer1], axis=0)
                plt.hist(act[act>0], bins=np.linspace(0,0.3,30))
                plt.subplot(2,2,3)
                act = np.mean(sim.data[p_layer2], axis=0)
                plt.hist(act[act>0], bins=np.linspace(0,0.3,30))
                plt.subplot(2,2,4)
                act = np.mean(sim.data[p_out], axis=0)
                plt.hist(act[act>0], bins=np.linspace(0,0.5,30))
                '''
                
            return dict(
                rmse_test = rmse_test,
                max_n_neurons = max([ens.n_neurons for ens in model.all_ensembles]),
                test_targets_raw = targets_test_raw,
                target_peak = target_peak,
                data_peak = data_peak,
                )


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

            if p.save_params:
                assert not p.split_spatial

                objects = list(model.all_ensembles) + list(model.all_connections)
                params = sim.get_nengo_params(objects, as_dict=False)

                np.save(os.path.join(p.data_dir, p.data_filename + '.params.npy'), params)


        data = sim.data[p_out].reshape(-1,targets_train.shape[-1])[:len(targets_test)]

        data_peak = np.array([davis_tracking.find_peak(d.reshape(output_shape)) for d in data])
        target_peak = np.array([davis_tracking.find_peak(d.reshape(output_shape)) for d in targets_test])

        rmse_test = np.sqrt(np.mean((target_peak-data_peak)**2, axis=0))*p.merge          
        if plt:
            plt.plot(data_peak*p.merge)
            plt.plot(target_peak*p.merge, ls='--')
            plt.plot((targets_test_raw-strip_edges)*p.merge, ls=':')
            
        return dict(
            rmse_test = rmse_test,
            max_n_neurons = max([ens.n_neurons for ens in model.all_ensembles]),
            #test_targets = targets_test,
            test_targets_raw = targets_test_raw,
            #test_output = data,
            target_peak = target_peak,
            data_peak = data_peak,
            test_loss = loss_post,
            )
