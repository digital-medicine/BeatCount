import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse

roi_coordinates = []
current_roi = None
drawing = False
selection_finished = False


def select_roi(event, x, y, flags, param):
    global current_roi, roi_coordinates, drawing, selection_finished

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_roi = (x, y, 0, 0)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_coordinates.append(current_roi)
        current_roi = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_roi = (current_roi[0], current_roi[1], x - current_roi[0], y - current_roi[1])

    elif event == cv2.EVENT_RBUTTONDOWN:
        selection_finished = True


def finish_selection(event, x, y):
    global selection_finished

    if event == ord("f"):
        selection_finished = True
    elif event == ord("c"):
        selection_finished = False


def calculate_signal_change(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_pixel_value = np.mean(gray_roi)
    return mean_pixel_value


def track_roi_in_video(path_video, path_result, peak_threshold, roi_coords):
    cap = cv2.VideoCapture(path_video)
    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        return

    cv2.namedWindow("Select ROIs")
    cv2.setMouseCallback("Select ROIs", select_roi)

    while not selection_finished:
        display_frame = frame.copy()
        for roi in roi_coords:
            x, y, w, h = roi
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if current_roi is not None:
            x, y, w, h = current_roi
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Select ROIs", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if not selection_finished:
        print("ROI selection canceled")
        return

    rois = [frame[y:y+h, x:x+w] for x, y, w, h in roi_coords]

    signal_changes = [[] for _ in range(len(rois))]
    previous_signals = [calculate_signal_change(roi) for roi in rois]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_rois = [frame[y:y+h, x:x+w] for x, y, w, h in roi_coords]
        current_signals = [calculate_signal_change(roi) for roi in current_rois]
        for i in range(len(rois)):
            signal_change = current_signals[i] - previous_signals[i]
            signal_changes[i].append(signal_change)
            previous_signals[i] = current_signals[i]

    overall_frequency = 1 / (len(signal_changes[0]) / cap.get(cv2.CAP_PROP_FPS))

    for i in range(len(rois)):
        signal_changes[i] = (signal_changes[i] - np.min(signal_changes[i])) / (np.max(signal_changes[i]) - np.min(signal_changes[i]))
        peaks, _ = find_peaks(signal_changes[i] > peak_threshold)

        plt.subplot(len(rois), 1, i+1)
        plt.plot(signal_changes[i], label=f"ROI {i+1}")
        plt.plot(peaks, np.array(signal_changes[i])[peaks], "ro", label="Peaks")
        plt.xlabel("Frame")
        plt.ylabel("Signal Change")
        plt.legend()
        plt.title(f"Signal Change in ROI {i+1}\nNumber of Peaks: {len(peaks)}")

    plt.tight_layout()
    plt.show()

    with open(path_result, "w") as file:
        file.write("ROI,Frame,SignalChange,Peak\n")
        for i in range(len(rois)):
            for j, signal_change in enumerate(signal_changes[i]):
                if j in peaks:
                    file.write(f"{i+1},{j+1},{signal_change},{True}\n")
                else:
                    file.write(f"{i + 1},{j + 1},{signal_change},{False}\n")

    print(f"Signal changes saved to {path_result}")
    cap.release()

def main(args):
    track_roi_in_video(args.path_video, args.path_result, args.peak_threshold, roi_coordinates)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='BeatCount', description='Quantify the signal change in regions of interest in video sequences')
    parser.add_argument('-v', '--path_video', type=str, help='location to load the video from')
    parser.add_argument('-r', '--path_result', type=str, default='./results.txt', help='location to save the results (default: ./results.txt)')
    parser.add_argument('-t', '--peak_threshold', type=float, default=0.8, help='which values to consider for peak detection in the range [0,1] (default: 0.8)')

    args = parser.parse_args()
    main(args)
