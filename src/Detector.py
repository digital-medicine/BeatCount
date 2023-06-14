import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class Detector:
    def __init__(self, path, rois=None):
        if rois is None:
            rois = [{'x': 100, 'y': 100, 'w': 100, 'h': 100}, {'x': 200, 'y': 200, 'w': 100, 'h': 100}]
        self.rois = rois
        self.cap = cv2.VideoCapture(path)
        self.num_bb = 1
        self.timeseries = [[] for _ in range(self.num_bb)]

    def calculate_signal_change(self, roi):
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # TODO: Is this even necessary, i.e. afterwards the mean is taken.
        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        mean_pixel_value = np.mean(blurred_roi)
        return mean_pixel_value

    def process(self):
        _, image = self.cap.read()
        rois_image = [image[k['y']:k['y'] + k['height'], k['x']:k['x'] + k['width']] for k in [self.rois]]
        previous_signals = [self.calculate_signal_change(roi) for roi in rois_image]
        while True:
            ret, image = self.cap.read()
            if not ret:
                break

            rois_image = [image[k['y']:k['y'] + k['height'], k['x']:k['x'] + k['width']] for k in [self.rois]]
            current_signals = [self.calculate_signal_change(roi) for roi in rois_image]
            for i in range(0, self.num_bb):
                signal_change = current_signals[i] - previous_signals[i]
                self.timeseries[i].append(signal_change)
                previous_signals[i] = current_signals[i]

    def detect_peaks(self):
        peaks, _ = find_peaks(self.timeseries, distance=150)
        return peaks


    def create_figure(self, timeseries):
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        xs = range(len(timeseries))
        ys = timeseries
        axis.plot(xs, ys)
        return fig


    def get_results(self):
        df = pd.DataFrame(self.timeseries).T
        return df
