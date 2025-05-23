import time
import cv2
import logging
import threading
import queue
import cv2
import argparse

class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")
    

class SensorX(Sensor):
    '''Sensor X'''
    def __init__(self, delay:float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data
    

class SensorCam(Sensor):
    def __init__(self, cam_name: str, size: tuple[int, int]):
        self._cam = cv2.VideoCapture(cam_name)
        if not self._cam.isOpened():
            logging.error("Camera not found.")
            raise RuntimeError("Camera init failed")
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    def get(self):
        ret, frame = self._cam.read()
        if not ret:
            logging.error("Camera read error.")
            raise RuntimeError("Failed to read from camera.")
        return frame

    def __del__(self):
        self._cam.release()


class WindowImage():
    def __init__(self, fps: int):
        self.delay = int(1000/fps)

    def show(self, image):
        cv2.imshow("Camera", image)
        key = cv2.waitKey(self.delay) & 0xFF
        return key

    def __del__(self):
        cv2.destroyWindow("Camera")


def sensor_worker(sensor: Sensor, out_queue: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            data = sensor.get()
            if out_queue.full():
                out_queue.get()
            out_queue.put(data)
        except Exception as e:
            logging.error(f"Sensor error: {e}")
            stop_event.set()
            break

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="/dev/video0")
    parser.add_argument("--size", type=str, default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    res = tuple(map(int, args.size.split("x")))
    logging.basicConfig(filename="log/app.log", level=logging.INFO)

    try:
        cam_sensor = SensorCam(args.camera, res)
    except Exception as e:
        logging.exception(f"Failed to initialize camera: {e}")
        return


    sensor0 = SensorX(0.01)
    sensor1 = SensorX(0.1)
    sensor2 = SensorX(1)

    sensors = [cam_sensor, sensor0, sensor1, sensor2]
    queues = [queue.Queue(maxsize=1) for _ in sensors]
    stop_event = threading.Event()

    threads = []

    for s, q in zip(sensors, queues):
        t = threading.Thread(target=sensor_worker, args=(s, q, stop_event))
        t.start()
        threads.append(t)

    display = WindowImage(args.fps)
    last_values = [None for _ in sensors]


    try:
        while not stop_event.is_set():
            for i, q in enumerate(queues):
                try:
                    last_values[i] = q.get_nowait()
                except queue.Empty:
                    pass

            image = last_values[0]
            if image is None:
                continue

            for idx, val in enumerate(last_values[1:], 1):
                if val is not None:
                    text = f"Sensor {idx}: {val}"
                    cv2.putText(image, text, (10, 30 * idx), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            key = display.show(image)
            if key == ord("q"):
                stop_event.set()
    
    finally:
        for t in threads:
            t.join()
        del cam_sensor
        del display


if __name__ == "__main__":
    main()


#python3 main.py --camera /dev/video0 --size 640x480 --fps 30



