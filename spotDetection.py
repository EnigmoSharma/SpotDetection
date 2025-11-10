import cv2
import numpy as np
import json
import os
import requests
from collections import defaultdict
import argparse
from datetime import datetime, timezone

# ==============================
# 🔗 Supabase Configuration
# ==============================
SUPABASE_URL = "Your_SUPABASE_URL"
SUPABASE_KEY = "Your_SUPABASE_KEY"
TABLE_NAME = "Put_Table_Name_Of_SupaBase"


def send_status_to_supabase(sensor_id, occupied):
    """
    Send parking spot occupancy data to Supabase table using PATCH.
    - Updates based on sensor_id
    - Matches schema: sensor_id, is_occupied, updated_at
    """
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}?sensor_id=eq.{sensor_id}"
    headers = {
        "Content-Type": "application/json",
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Prefer": "return=representation"
    }

    payload = {
        "is_occupied": occupied,
        "updated_at": "now()"  # handled by Supabase server
    }

    try:
        response = requests.patch(url, headers=headers, json=payload, timeout=10)
        if response.status_code in (200, 204):
            print(f"[Supabase] Sensor {sensor_id}: {'OCCUPIED ✓' if occupied else 'FREE ✓'}")
        else:
            print(f"[Supabase] Failed for {sensor_id}: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[Supabase] Connection error for {sensor_id}: {e}")


# ==============================
# 🎥 Parking Detection Class
# ==============================
class ParkingSpaceDetector:
    def __init__(self, video_source=0, stability_frames=15, occupancy_threshold=1000, resize_to=(960, 540)):
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)

        self.video_source = video_source
        print(f"Opening video source: {self.video_source}")
        self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")

        self.parking_spots = {}
        self.load_parking_spots()

        self.presence_counter = defaultdict(int)
        self.stability_frames = stability_frames
        self.occupancy_threshold = occupancy_threshold
        self.resize_to = resize_to
        self.last_status = {}

        if not self.parking_spots:
            print("No parking spots found. Press 'd' to draw parking areas.")

    def load_parking_spots(self):
        """Load parking spots from file."""
        if os.path.exists('parking_spots.json'):
            with open('parking_spots.json', 'r') as f:
                self.parking_spots = json.load(f)

    def save_parking_spots(self):
        """Save parking spots to file."""
        with open('parking_spots.json', 'w') as f:
            json.dump(self.parking_spots, f, indent=4)

    def add_parking_spot(self, spot_id, points):
        """Add and save a new parking spot."""
        self.parking_spots[spot_id] = {'points': points}
        self.save_parking_spots()

    def detect_occupancy(self, frame):
        """Detect if each parking spot is occupied using edge detection."""
        status = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny_edges = cv2.Canny(blur, 50, 150)

        for spot_id, spot in self.parking_spots.items():
            pts = np.array(spot['points'], np.int32)
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [pts], 255)

            masked_edges = cv2.bitwise_and(canny_edges, canny_edges, mask=mask)
            edge_pixel_count = cv2.countNonZero(masked_edges)
            occupied = edge_pixel_count > self.occupancy_threshold

            # Stability logic
            if occupied:
                self.presence_counter[spot_id] = min(self.stability_frames, self.presence_counter[spot_id] + 1)
            else:
                self.presence_counter[spot_id] = max(0, self.presence_counter[spot_id] - 1)

            stable_occupied = self.presence_counter[spot_id] >= self.stability_frames
            status[spot_id] = stable_occupied

            # Draw parking boundaries
            color = (0, 0, 255) if stable_occupied else (0, 255, 0)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            x, y = spot['points'][0]
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(10, min(y, frame.shape[0] - 1))
            text = f"{spot_id}: {'Occupied' if stable_occupied else 'Free'} ({edge_pixel_count})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return status, canny_edges

    def run(self):
        """Run the detection and visualization loop."""
        drawing = False
        current_spot = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, current_spot
            if drawing and event == cv2.EVENT_LBUTTONDOWN:
                current_spot.append((x, y))

        cv2.namedWindow('Parking Space Detection')
        cv2.setMouseCallback('Parking Space Detection', mouse_callback)
        cv2.namedWindow('Edges Debug')

        print("Press 'd' to start/stop drawing mode | 'c' to complete | 'q' to quit")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("\nFailed to grab frame. (End of video or stream disconnect?)")
                    break

                if self.resize_to:
                    frame_resized = cv2.resize(frame, self.resize_to)
                else:
                    frame_resized = frame.copy()

                frame_copy = frame_resized.copy()
                debug_edges = np.zeros_like(frame_resized, dtype=np.uint8)

                # === Main Detection + Supabase Update Logic ===
                if self.parking_spots:
                    status, canny_edges = self.detect_occupancy(frame_copy)

                    # Send updates only when status changes
                    self.last_status = getattr(self, "last_status", {})
                    for spot_id, occupied in status.items():
                        if self.last_status.get(spot_id) != occupied:
                            send_status_to_supabase(spot_id, occupied)
                    self.last_status = status

                    debug_edges = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
                    print("\r" + " ".join([f"{k}:{'OCC' if v else 'free'}" for k, v in status.items()]), end="")

                # === Drawing Mode ===
                if drawing:
                    for i, point in enumerate(current_spot):
                        cv2.circle(frame_copy, point, 5, (0, 0, 255), -1)
                        if i > 0:
                            cv2.line(frame_copy, current_spot[i - 1], current_spot[i], (0, 0, 255), 2)
                    if len(current_spot) > 2:
                        cv2.line(frame_copy, current_spot[-1], current_spot[0], (0, 0, 255), 2)

                cv2.imshow('Parking Space Detection', frame_copy)
                cv2.imshow('Edges Debug', debug_edges)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    drawing = not drawing
                    if drawing:
                        print("\nDrawing mode ON. Click to add points. Press 'c' to finish spot.")
                        current_spot = []
                    else:
                        print("\nDrawing mode OFF.")
                        current_spot = []
                elif key == ord('c') and drawing and len(current_spot) >= 3:
                    spot_id = input("Enter spot ID (e.g., s1, s2, A1): ").strip()
                    if spot_id:
                        self.add_parking_spot(spot_id, current_spot)
                        print(f"Spot '{spot_id}' saved. Press 'd' to draw another.")
                    current_spot = []
                    drawing = False

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


# ==============================
# 🚗 Main Entry
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parking Space Detection with Supabase Sync")
    parser.add_argument("--source", "-s", default="0", help="Video source: camera index (0) or file path.")
    parser.add_argument("--stability", type=int, default=15, help="Frames required for stable occupancy.")
    parser.add_argument("--threshold", type=int, default=1000, help="Edge pixel threshold for occupancy.")
    parser.add_argument("--width", type=int, default=960, help="Resized frame width.")
    parser.add_argument("--height", type=int, default=540, help="Resized frame height.")
    args = parser.parse_args()

    print("🚗 Parking Space Detection + Supabase Integration")
    print("--------------------------------------------------")

    try:
        detector = ParkingSpaceDetector(
            video_source=args.source,
            stability_frames=args.stability,
            occupancy_threshold=args.threshold,
            resize_to=(args.width, args.height)
        )
        detector.run()
    except Exception as e:
        print("Error:", e)
