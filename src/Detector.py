import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


class Detector:
    def __init__(self, path, destination, rois=None):
        if rois is None:
            rois = [{'x': 100, 'y': 100, 'w': 100, 'h': 100}, {'x': 200, 'y': 200, 'w': 100, 'h': 100}]
        self.rois = rois
        self.cap = cv2.VideoCapture(path)
        self.num_bb = 1
        self.timeseries = [[] for _ in range(self.num_bb)]
        self.destination = destination

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

    def save_plots(self):
        for i, k in enumerate(self.timeseries):
            plt.plot(k)
            plt.xlabel('Frame')
            plt.xlabel('Intensity of Change')
            plt.savefig(self.destination + str(i) + '.png')

    def save_results(self):
        df = pd.DataFrame(self.timeseries)
        df.to_csv(self.destination + 'result.csv')
