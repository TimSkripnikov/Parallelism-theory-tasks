import argparse
import threading
import queue
import time
from timeit import default_timer as timer

import cv2
from ultralytics import YOLO


class PoseModel:
    def __init__(self, path="yolov8s-pose.pt"):
        self.model = YOLO(path)

    def __del__(self):
        del self.model

    def predict(self, frame):
        return self.model.predict(frame, verbose=False)[0]


def video_writer(cap, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h))


def process_single(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    out = video_writer(cap, output_path)
    model = PoseModel()
    start = timer()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = model.predict(frame)
        out.write(result.plot())

    cap.release()
    out.release()
    end = timer()
    print(f"Single-thread time: {end - start:.2f} seconds\n")


def process_multi(video_path, output_path, num_threads=4):
    cap = cv2.VideoCapture(video_path)
    out = video_writer(cap, output_path)
    model = PoseModel()

    frame_queue = queue.Queue(maxsize=3 * num_threads)
    result_list = []
    stop_event = threading.Event()
    start = timer()

    def worker():
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                frame_id, frame = frame_queue.get(timeout=0.1)
                result = model.predict(frame)
                result_list.append((frame_id, result.plot()))
                frame_queue.task_done()
            except queue.Empty:
                continue

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put((frame_id, frame))
        frame_id += 1

    frame_queue.join()
    stop_event.set()
    for t in threads:
        t.join()

    result_list.sort(key=lambda x: x[0])
    for _, frame in result_list:
        out.write(frame)

    cap.release()
    out.release()
    end = timer()
    print(f"Multi-thread ({num_threads} threads) time: {end - start:.2f} seconds\n")


def realtime_system():
    cap = cv2.VideoCapture(0)
    model = PoseModel()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = model.predict(frame)
        cv2.imshow("output", result.plot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='Path to input video')
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], help='Processing mode')
    parser.add_argument('--t', type=int, default=4, help='Number of threads for multi-mode')
    parser.add_argument('--output_path', type=str, help='Path to save output video')
    parser.add_argument('--camera', action='store_true', help='Run real-time camera inference')
    args = parser.parse_args()

    if args.camera:
        realtime_system()
    elif args.mode == 'single':
        process_single(args.video_path, args.output_path)
    elif args.mode == 'multi':
        process_multi(args.video_path, args.output_path, args.t)


if __name__ == "__main__":
    main()
