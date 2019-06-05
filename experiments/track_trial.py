import pytry
import os
import random
import nengo
import nengo_extras
import numpy as np

import davis_tracking

class TrackingTrial(pytry.PlotTrial):
    def params(self):
        self.param('number of data sets to use', n_data=-1)
        self.param('data directory', dataset_dir=r'../dataset')
        self.param('dt', dt=0.01)
        self.param('dt_test', dt_test=0.001)
        self.param('decay time (input synapse)', decay_time=0.01)
        self.param('number of neurons', n_neurons=100)
        self.param('gabor size', gabor_size=11)
        self.param('solver regularization', reg=0.03)
        self.param('test set (odd|one)', test_set='one')
        self.param('augment training set with flips', augment=True)
        self.param('output filter', output_filter=0.01)
        self.param('saturation', saturation=5)
        self.param('separate positive and negative channels', separate_channels=True)
        self.param('merge pixels (to make a smaller image)', merge=1)
        
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
            print(f)
            times, imgs, targs = davis_tracking.load_data(f, dt=p.dt, decay_time=p.decay_time,
                                                separate_channels=p.separate_channels, 
                                                saturation=p.saturation, merge=p.merge)
            inputs.append(imgs)
            targets.append(targs[:,:2])
                                
        inputs_all = np.vstack(inputs)
        targets_all = np.vstack(targets)
        
        if p.test_set == 'odd':
            inputs_train = inputs_all[::2]
            inputs_test = inputs_all[1::2]
            targets_train = targets_all[::2]
            targets_test = targets_all[1::2]
        elif p.test_set == 'one':
            times, imgs, targs = davis_tracking.load_data(test_file, dt=p.dt_test, decay_time=p.decay_time,
                                                separate_channels=p.separate_channels, 
                                                saturation=p.saturation)
            inputs_test = imgs
            targets_test = targs[:, :2]
            inputs_train = inputs_all
            targets_train = targets_all
            
        if p.augment:
            inputs_train, targets_train = davis_tracking.augment(inputs_train, targets_train,
                                                                 separate_channels=p.separate_channels)
            
        if p.separate_channels:
            shape = (360//p.merge, 240//p.merge)
        else:
            shape = (180//p.merge, 240//p.merge)
        
        dimensions = shape[0]*shape[1]
        eval_points_train = inputs_train.reshape(-1, dimensions)
        eval_points_test = inputs_test.reshape(-1, dimensions)

        model = nengo.Network()
        with model:
            from nengo_extras.vision import Gabor, Mask
            encoders = Gabor().generate(p.n_neurons, (p.gabor_size, p.gabor_size))
            encoders = Mask(shape).populate(encoders, flatten=True)

            ens = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=dimensions,
                                 encoders=encoders,
                                 neuron_type=nengo.RectifiedLinear(),
                                 intercepts=nengo.dists.CosineSimilarity(p.gabor_size**2+2)
                                 )

            result = nengo.Node(None, size_in=targets_all.shape[1])

            c = nengo.Connection(ens, result, 
                                 eval_points=eval_points_train,
                                 function=targets_train,
                                 solver=nengo.solvers.LstsqL2(reg=p.reg),
                                 )
        sim = nengo.Simulator(model)
        
        error_train = sim.data[c].solver_info['rmses']

        _, a_train = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=eval_points_train)    
        outputs_train = np.dot(a_train, sim.data[c].weights.T)       
        rmse_train = np.sqrt(np.mean((targets_train-outputs_train)**2, axis=0))
        _, a_test = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=eval_points_test)    
        outputs_test = np.dot(a_test, sim.data[c].weights.T)       
        filt = nengo.synapses.Lowpass(p.output_filter)
        outputs_test = filt.filt(outputs_test, dt=p.dt_test)
        targets_test = filt.filt(targets_test, dt=p.dt_test)
        rmse_test = np.sqrt(np.mean((targets_test-outputs_test)**2, axis=0))*p.merge
        
        
        if plt:
            plt.subplot(2, 1, 1)
            plt.plot(targets_train, ls='--')
            plt.plot(outputs_train)
            plt.title('train\nrmse=%1.4f,%1.4f' % tuple(rmse_train))
            
            plt.subplot(2, 1, 2)
            plt.plot(targets_test, ls='--')
            plt.plot(outputs_test)
            plt.title('test\nrmse=%1.4f,%1.4f' % tuple(rmse_test))
            
        return dict(
            rmse_train=rmse_train,
            rmse_test=rmse_test,
        )
