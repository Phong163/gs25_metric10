import argparse
from datetime import datetime
import cv2
import time
import threading
import os
import numpy as np
import onnxruntime as ort
import pytz   # ✅ thêm onnxruntime
from utils import setup_logger, send_time_to_kafka
from config.params import camera_configs
from rtsp_stream import RTSPStream

logger = setup_logger()

# ==== Load ONNX model ====
onnx_session = ort.InferenceSession("./weights/mobilenetv2_occupied.onnx", providers=["CPUExecutionProvider"])
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

def classify_crop(crop):
    img = cv2.resize(crop, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # normalize
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    # ✅ ép kiểu float32 (đảm bảo input đúng format cho model)
    img = img.astype(np.float32)

    preds = onnx_session.run([output_name], {input_name: img})[0]
    label = np.argmax(preds, axis=1)[0]
    return label



class CustomerTracker:
    def __init__(self, camera_id, output_path, size=640, show_video=False, send_api=False, save_crop=True):
        self.show_video = show_video
        self.send_api = send_api
        self.save_crop = save_crop

        if camera_id not in camera_configs:
            raise ValueError(f"No configuration found for camera_id {camera_id}")

        config = camera_configs[camera_id]
        logger.debug(f"Camera {camera_id} config: {config}")
        if not isinstance(config, dict):
            raise ValueError(f"Configuration for camera_id {camera_id} is not a dictionary: {config}")
        
        

        self.rtsp_url = config["rtsp_url"]
        self.camera_id = camera_id
        self.zone_id = config["zone_id"]
        self.box_id = config["box_id"]
        self.cam_id = config["cam_id"]
        self.zones = config["zone"]

        self.current_frame = config["current_frame"]
        self.output_path = output_path
        self.size = size

        self.rtsp_stream = RTSPStream(self.rtsp_url, self.camera_id)
        self.rtsp_stream.start()
        self.fps = int(self.rtsp_stream.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.rtsp_stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.rtsp_stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.zone_states = {idx: {"status": "Empty", "start_time": None, "end_time": 0, "total_time": 0} for idx in range(len(self.zones))}
        if self.show_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_file = output_path.replace('.mp4', f'_cam{camera_id}.mp4')
            self.out = cv2.VideoWriter(output_file, fourcc, self.fps, (self.width, self.height))

    def process_frame(self, frame):
        self.current_frame += 1
        annotated_frame = frame.copy()
        date_time = datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).strftime("%d%m%Y%H%M")

        for idx, zone in enumerate(self.zones):
            pts = [(int(x * self.width), int(y * self.height)) for x, y in zone]
            x_min = min([p[0] for p in pts])
            y_min = min([p[1] for p in pts])
            x_max = max([p[0] for p in pts])
            y_max = max([p[1] for p in pts])

            crop = frame[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue
            
            # ==== chạy phân loại ====
            label = classify_crop(crop)   # 0: empty, 1: occupied
            color = (0, 0, 255) if label == 0 else (0, 255, 0)

            # ==== xử lý state machine ====
            zone_state = self.zone_states[idx]

            if label == 1 and zone_state["status"] == "Empty":
                # chuyển từ Empty -> Occupied
                zone_state["status"] = "Occupied"
                zone_state["start_time"] = time.time()
            elif label == 1 and zone_state["status"] == "Occupied":
                zone_state["total_time"] = time.time() - zone_state["start_time"]
                zone_state["end_time"] = time.time()
                if zone_state["total_time"] > 7200 and self.send_api:
                        send_time_to_kafka(self.box_id, self.zone_id, self.cam_id, date_time, zone_state["total_time"])
                        # reset state
                        zone_state["status"] = "Empty"
                        zone_state["start_time"] = 0
                        zone_state["total_time"] = 0
                    
                
            elif label == 0 and zone_state["status"] == "Occupied":
                # chuyển từ Occupied -> Empty
                if zone_state["total_time"] < 300:
                    zone_state["status"] = "Empty"
                    zone_state["start_time"] = 0
                    zone_state["total_time"] = 0
                else:
                    out_time = time.time() - zone_state["end_time"]
                    if out_time > 10:  # chỉ log nếu > 30s
                        date_time = datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).strftime("%d%m%Y%H%M")
                        if self.send_api:
                            send_time_to_kafka(self.box_id, self.zone_id, self.cam_id, date_time, zone_state["total_time"])
                        zone_state["status"] = "Empty"
                        zone_state["start_time"] = 0
                        zone_state["total_time"] = 0
                    if self.show_video:
                        cv2.putText(annotated_frame,
                                f"out_time:{out_time:.1f}s",
                                (x_min, y_min - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    
            # ==== vẽ bounding zone ====
            if self.show_video:
                cv2.polylines(annotated_frame, [np.array(pts, dtype=int)], True, color, 2)
                cv2.putText(annotated_frame,
                            f"{zone_state['total_time']:.1f}s",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return annotated_frame


    def run(self):
        while self.rtsp_stream.running:
            ret, frame = self.rtsp_stream.get_frame()
            if not ret or frame is None:
                time.sleep(0.2)
                continue
            annotated_frame = self.process_frame(frame)
            if self.show_video:
                annotated_frame = cv2.resize(annotated_frame, (720, 480))
                cv2.imshow(f"Camera {self.camera_id}", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self.rtsp_stream.stop()
        if self.show_video and hasattr(self, 'out'):
            self.out.release()
            cv2.destroyAllWindows()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crop zone images from RTSP stream')
    parser.add_argument('--camera_id', type=int, nargs='*', default=None,
                        help='Camera ID(s) to process (default: all cameras in camera_configs)')
    parser.add_argument('--output', type=str, default='./output/output_video.mp4',
                        help='Base path to output video (camera_id will be appended)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Size for resizing frames')
    parser.add_argument('--show_video', action='store_true',
                        help='Whether to save the output video')
    parser.add_argument('--send_api', action='store_true',
                        help='Whether to save the output video')
    return parser.parse_args()

def run_tracker_for_camera(camera_id, output_path, size, show_video, send_api):
    """Run a CustomerTracker for a specific camera in a separate thread."""
    try:
        tracker = CustomerTracker(
            camera_id=camera_id,
            output_path=output_path,
            size=size,
            show_video=show_video,
            send_api=send_api,
        )
        tracker.run()
    except Exception as e:
        logger.error(f"Error running tracker for camera {camera_id}: {e}")

if __name__ == "__main__":
    args = parse_args()
    camera_ids = args.camera_id if args.camera_id is not None else list(camera_configs.keys())
    threads = []
    for camera_id in camera_ids:
        thread = threading.Thread(
            target=run_tracker_for_camera,
            args=(camera_id, args.output, args.imgsz, args.show_video, args.send_api)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
