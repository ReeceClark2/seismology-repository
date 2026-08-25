import os

# 1. Force math backends (like NumPy/OpenBLAS) to single-thread 
#    to prevent them from deadlocking inside worker processes.
#    MUST be set before importing numpy!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing
import numpy as np
from dracula import Dracula
import matplotlib.pyplot as plt
from bats import BATS
import pandas as pd
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client


class data():
    def __init__(self,
                 min_f,
                 max_f
                 ):

        self.min_f = min_f
        self.max_f = max_f


    def get_observed_data(self, 
                          network, 
                          station, 
                          channel, 
                          location, 
                          stream_index,
                          start_time, 
                          end_time
                          ):
        
        client = Client('IRIS')

        inventory = client.get_stations(network=network, 
                                        station=station,
                                        location=location, 
                                        channel=channel,
                                        starttime=start_time, 
                                        endtime=end_time,
                                        level='response'
                                        )
        
        stream = client.get_waveforms(network=network, 
                                      station=station,
                                      location=location, 
                                      channel=channel,
                                      starttime=start_time, 
                                      endtime=end_time
                                      )
        
        trace = stream[stream_index]
        trace.detrend('constant')
        trace.remove_response(inventory=inventory, output="ACC")
        
        if self.min_f and self.max_f:
            trace.filter('bandpass', freqmin=self.min_f, freqmax=self.max_f)

        trace.decimate(5, no_filter=False)
        trace.decimate(5, no_filter=False)

        delta = trace.stats.delta
        N = len(trace)
        t = np.arange(N) * delta
        d = np.array(trace.data)

        return t, d


    def get_normal_modes(self, path):
        df = pd.read_csv(f"{path}")

        print(df.head)



    
