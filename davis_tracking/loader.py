import numpy as np
import os
import traceback

class Trace(object):
    def __init__(self, params):
        self.times = []
        self.params = {}
        for p in params:
            self.params[p] = []
    def frame(self, t, **params):
        self.times.append(t)
        for p in self.params.keys():
            if p not in params:
                self.params[p].append(self.params[p][-1])
            else:
                self.params[p].append(params[p])

    def get(self, t, p):
        if t < self.times[0]:
            return None
        elif t > self.times[-1]:
            return None
        else:
            return np.interp([t], self.times, self.params[p])[0]


def load_trace(fn):
    if not os.path.isfile(fn):
        return None
    else:
        with open(fn) as f:
            code = f.read()
        locals = dict()
        globals = dict(Trace=Trace)
        try:
            exec(code, globals, locals)
        except:
            traceback.print_exc()
            return None
        for k, v in locals.items():
            if isinstance(v, Trace):
                return v
        else:
            return None

def extract_targets(filename, dt, t_start=None, t_end=None):
    trace = load_trace(filename+'.label')
    if t_end is None:        
        t_end = trace.times[-1] if trace is not None and len(trace.times)>0 else -1
    if t_start is None:
        t_start = 0
        
    times = []
    targets = []
    now = t_start
    while now < t_end:
        xx = trace.get(now, 'x')
        yy = trace.get(now, 'y')
        rr = trace.get(now, 'r')
        valid = 1 if xx is not None else 0
        if xx is None:
            xx = -1
        if yy is None:
            yy = -1
        if rr is None:
            rr = -1

        now += dt

        targets.append([xx, yy, rr, valid])
        times.append(now)

    targets = np.array(targets).reshape(-1, 4)
    times = np.array(times)
    
    return times, targets
        
    
    
def extract_images(filename,        # filename to load data from 
                   dt,              # time between images to create (seconds)
                   decay_time=0.1,  # spike decay time (seconds)
                   t_start=None,    # time to start generating images (seconds)
                   t_end=None,      # time to end generating images (seconds)
                   separate_channels=False, # separate the pos and neg channels
                   keep_pos=True,   # keep negative events
                   keep_neg=True,   # keep positive events
                   saturation=10,   # clip data to this limit
                   merge=1,         # merge pixels
                  ):

    if separate_channels:
        t_pos, images_pos = extract_images(filename, dt, decay_time,
                                           t_start, t_end, keep_neg=False,
                                           saturation=saturation, merge=merge)
        t_neg, images_neg = extract_images(filename, dt, decay_time,
                                           t_start, t_end, keep_pos=False,
                                           saturation=saturation, merge=merge)
        assert np.array_equal(t_pos, t_neg)
        return t_pos, np.hstack([images_pos, images_neg])


    fn = '%s_%g_%g_%g_%g_%g_%d%s%s.cache.npz' % (filename, dt, decay_time,
                                              t_start, t_end,
                                              saturation, merge,
                                              '_pos' if keep_pos else '', 
                                              '_neg' if keep_neg else '')
    if os.path.exists(fn):
        data = np.load(fn)
        return data['times'], data['images']
    
    packet_size = 8

    with open(filename, 'rb') as f:
        data = f.read()
    data = np.fromstring(data, np.uint8)

    # find x and y values for events
    y = ((data[1::packet_size].astype('uint16')<<8) + data[::packet_size]) >> 2
    x = ((data[3::packet_size].astype('uint16')<<8) + data[2::packet_size]) >> 1
    # get the polarity (+1 for on events, -1 for off events)
    p = np.where((data[::packet_size] & 0x02) == 0x02,
                 1 if keep_pos else 0,
                 -1 if keep_neg else 0)
    v = np.where((data[::packet_size] & 0x01) == 0x01, 1, -1)
    # find the time stamp for each event, in seconds from the start of the file
    t = data[7::packet_size].astype(np.uint32)
    t = (t << 8) + data[6::packet_size]
    t = (t << 8) + data[5::packet_size]
    t = (t << 8) + data[4::packet_size]
    #t = t - t[0]
    t = t.astype(float) / 1000000   # convert microseconds to seconds

    if t_start is None:
        t_start = 0
    if t_end is None:
        t_end = t[-1]

    image = np.zeros((180, 240), dtype=float)

    images = []
    targets = []
    times = []

    event_index = 0   # for keeping track of where we are in the file
    if t_start > 0:
        event_index = np.searchsorted(t, t_start)

    now = t_start

    while now < t_end:
        decay_scale = np.exp(-dt/decay_time)#1-dt/(dt+decay_time)
        image *= decay_scale

        count = np.searchsorted(t[event_index:], now + dt)
        s = slice(event_index, event_index+count)

        dts = dt-(t[s]-now)
        np.add.at(image, [y[s], x[s]], p[s] * np.exp(-dts/decay_time))
        event_index += count

        image = np.clip(image, -saturation, saturation)

        now += dt

        merged_images = []
        for i in range(merge):
            for j in range(merge):
                merged_images.append(image[i::merge,j::merge])
                
        images.append(np.mean(merged_images, axis=0))
        
        times.append(now)

    images = np.array(images)
    times = np.array(times)

    np.savez(fn, times=times, images=images)
    
    return times, images


def load_data(filename, dt, decay_time, separate_channels=False,
              saturation=10, merge=1):
    times, targets = extract_targets(filename, dt=dt)
    index = 0
    while targets[index][3] == 0:
        index += 1
                
    times2, images = extract_images(filename,
                                 dt=dt, decay_time=decay_time,
                                 t_start=times[index]-2*dt, t_end=times[-1]-dt,
                                 separate_channels=separate_channels,
                                 saturation=saturation, merge=merge
                                 )
    times = times[index:]
    targets = targets[index:]
    if len(images) > len(targets):
        assert len(images) == len(targets) + 1
        images = images[:len(targets)]

    assert len(times)==len(targets)
    assert len(targets)==len(images)

    return times, images, targets/merge

def load_frames(filename, merge=1):
    packet_size = 4+180*240*2

    with open(filename, 'rb') as f:
        data = f.read()
    data = np.fromstring(data, np.uint8)

    t = data[3::packet_size].astype(np.uint32)
    t = (t << 8) + data[2::packet_size]
    t = (t << 8) + data[1::packet_size]
    t = (t << 8) + data[0::packet_size]
    t = t.astype(float) / 1000000 
    images = []

    for index, tt in enumerate(t):
        d = data[index*packet_size+4:(index+1)*packet_size]
        high = d[1::2]
        low = d[0::2]
        v = high.astype(int)<<8 + low
        v = v.astype(float).reshape(180,240)/32768
        
        
        merged_images = []
        for i in range(merge):
            for j in range(merge):
                merged_images.append(v[i::merge,j::merge])
                
        images.append(np.mean(merged_images, axis=0))        
    return t, np.array(images)