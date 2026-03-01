#!/usr/bin/env python
from obspy.core import read
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client 
import sys
import tqdm
import pandas as pd
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


def animate(xs, ys, labels=None, colors=None, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, interval=50, filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set static plot attributes
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title)
    if xlim:   ax.set_xlim(xlim)
    if ylim:   ax.set_ylim(ylim)

    frames = []

    num_lines = len(xs[0]) if isinstance(xs[0], (list, tuple)) else 1
    
    if colors is None:
        colors = [None] * num_lines
    
    pbar = tqdm.tqdm(total=len(xs))
    # xs and ys are expected to be: [ [x1, x2], [x1, x2], ... ] for each frame
    for f_idx, (x_frame, y_frame) in enumerate(zip(xs, ys)):
        frame_artists = []
        
        if not isinstance(x_frame, (list, tuple)):
            x_frame, y_frame = [x_frame], [y_frame]
            
        for l_idx, (x_line, y_line) in enumerate(zip(x_frame, y_frame)):
            # Apply label only if it exists and it's the first frame (to avoid duplicate legend entries)
            line_label = labels[l_idx] if (labels and f_idx == 0 and l_idx < len(labels)) else None
            line_color = colors[l_idx] if colors and l_idx < len(colors) else None
            
            line, = ax.plot(x_line, y_line, color=line_color, label=line_label, alpha=0.7)
            frame_artists.append(line)
        
        frames.append(frame_artists)

        pbar.update(1)

    if labels:
        ax.legend(loc='upper right')

    ani = ArtistAnimation(fig, frames, interval=interval, blit=True)
    ani.save(f'{filename}.mp4', writer='ffmpeg', fps=2, dpi=200)
