import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple
import cv2


def convert_bbox_to_z(bbox):
    w, h = bbox[2:4] - bbox[0:2]
    x, y = (bbox[0:2] + bbox[2:4]) / 2
    s = w * h
    r = w / h
    return np.array([x, y, s, r])[:, None].astype(np.float64)


def convert_x_to_bbox(x):
    x = x[:, 0]
    center = x[0:2]
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    half_size = np.array([w, h]) / 2
    bbox = np.concatenate((center - half_size, center + half_size))
    return bbox.astype(np.float64)


class KalmanBoxTracker:

    def __init__(self, bbox, label):
        self.id = label
        self.time_since_update = 0
        self.hit_streak = 0

        self.kf = cv2.KalmanFilter(dynamParams=7, measureParams=4, type=cv2.CV_64F)
        self.kf.transitionMatrix = np.array([[1, 0, 0, 0, 1, 0, 0],
                                             [0, 1, 0, 0, 0, 1, 0],
                                             [0, 0, 1, 0, 0, 0, 1],
                                             [0, 0, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)
        self.kf.processNoiseCov = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4]).astype(np.float64)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0, 0]], dtype=np.float64)
        self.kf.measurementNoiseCov = np.diag([1, 1, 10, 10]).astype(np.float64)
        self.kf.statePost = np.vstack((convert_bbox_to_z(bbox), [[0], [0], [0]]))
        self.kf.errorCovPost = np.diag([1, 1, 1, 1, 1e-2, 1e-2, 1e-4]).astype(np.float64)

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.correct(convert_bbox_to_z(bbox))

    def predict(self):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        retval = self.kf.predict()
        return convert_x_to_bbox(retval)

    @property
    def current_state(self):
        return convert_x_to_bbox(self.kf.statePost)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_tl, a_br = a[:4].reshape((2, 2))
    b_tl, b_br = b[:4].reshape((2, 2))
    int_tl = np.maximum(a_tl, b_tl)
    int_br = np.minimum(a_br, b_br)
    int_area = np.product(np.maximum(0., int_br - int_tl))
    a_area = np.product(a_br - a_tl)
    b_area = np.product(b_br - b_tl)
    return int_area / (a_area + b_area - int_area)


def associate_detections_to_trackers(detections: np.ndarray, trackers: np.ndarray,
                                     iou_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float64)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.transpose(np.array([row_ind, col_ind]))

    iou_values = np.array([iou_matrix[detection, tracker]
                           for detection, tracker in matched_indices])
    good_matches = matched_indices[iou_values > 0.3]
    unmatched_detections = np.array(
        [i for i in range(len(detections)) if i not in good_matches[:, 0]])
    unmatched_trackers = np.array(
        [i for i in range(len(trackers)) if i not in good_matches[:, 1]])
    return good_matches, unmatched_detections, unmatched_trackers


class Sort:
    def __init__(self, max_age=10, min_hits=6):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.count = 0

    def next_id(self):
        self.count += 1
        return self.count

    def update(self, dets):
        self.trackers = [
            tracker for tracker in self.trackers if not np.any(
                np.isnan(tracker.predict()))]
        trks = np.array([tracker.current_state for tracker in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks)

        for detection_num, tracker_num in matched:
            self.trackers[tracker_num].update(dets[detection_num])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :], self.next_id()))

        ret = np.array([np.concatenate((trk.current_state, [trk.id + 1]))
                        for trk in self.trackers
                        if trk.time_since_update < 1 and trk.hit_streak >= self.min_hits])
        self.trackers = [
            tracker for tracker in self.trackers if tracker.time_since_update <= self.max_age]
        return ret
