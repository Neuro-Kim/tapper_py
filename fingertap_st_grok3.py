import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.ticker import MaxNLocator
import os
import json
import telepot
from io import BytesIO

# Load Telegram configuration
with open('telegram_config.json', 'r') as config_file:
    config = json.load(config_file)
    TOKEN = config['TOKEN']
    CHAT_ID = config['CHAT_ID']

# Define constant for camera settings file
CAMERA_SETTINGS_FILE = 'camera_settings.json'

# Function to load camera settings
def load_camera_settings():
    default_camera = 0
    if os.path.exists(CAMERA_SETTINGS_FILE):
        try:
            with open(CAMERA_SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                return settings.get('camera_index', default_camera)
        except Exception as e:
            st.error(f"Error loading camera settings: {e}")
    return default_camera

# Function to save camera settings
def save_camera_settings(camera_index):
    try:
        with open(CAMERA_SETTINGS_FILE, 'w') as f:
            settings = {'camera_index': camera_index}
            json.dump(settings, f)
        st.success(f"Camera settings saved. Using camera index: {camera_index}")
    except Exception as e:
        st.error(f"Error saving camera settings: {e}")

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Streamlit App
st.title("Finger Tapping Speed Test")
st.write("Measure your finger tapping speed using your webcam!")

# Sidebar for settings
st.sidebar.title("Settings")
st.sidebar.write("Select Camera:")
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("0"):
    save_camera_settings(0)
    camera_index = 0
if col2.button("1"):
    save_camera_settings(1)
    camera_index = 1
if col3.button("2"):
    save_camera_settings(2)
    camera_index = 2
camera_index = load_camera_settings()
st.sidebar.write(f"Current camera: {camera_index}")

test_duration = st.sidebar.slider("Test Duration (seconds)", 5, 30, 15)
distance_threshold = st.sidebar.slider("Tip Distance Threshold", 0, 100, 100)

# Webcam setup
cap = cv2.VideoCapture(camera_index)

# Check if camera opened successfully
if not cap.isOpened():
    st.error(f"Failed to open camera index {camera_index}. Please check your webcam.")
    st.stop()

# State management using session state
if 'test_started' not in st.session_state:
    st.session_state.test_started = False
    st.session_state.test_completed = False
    st.session_state.test_ready = False
    st.session_state.finger_tap_count = 0
    st.session_state.tap_timestamps = []
    st.session_state.distance_data = []
    st.session_state.distance_timestamps = []
    st.session_state.start_time = None
    st.session_state.graph_filename = ""
    st.session_state.is_tapping = False
    st.session_state.prev_tapping_state = False
    st.session_state.tap_effect_time = 0

# Constants
segment_duration = 5  # 5-second segments
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP

# Placeholder for video feed
video_placeholder = st.empty()

# Buttons
start_button = st.button("Start Test" if not st.session_state.test_started else "Test in Progress")
restart_button = st.button("Restart Test", disabled=not st.session_state.test_completed)

# Telegram bot
bot = telepot.Bot(TOKEN)

# Main loop
def process_frame():
    success, image = cap.read()
    if not success:
        st.error("Cannot read frame from webcam.")
        return None

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            if not st.session_state.test_ready and not st.session_state.test_started and not st.session_state.test_completed:
                st.session_state.test_ready = True
                st.write("Hand detected. Click 'Start Test' to begin.")

            thumb_tip = hand_landmarks.landmark[THUMB_TIP]
            index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
            h, w, c = image.shape
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            cv2.circle(image, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
            cv2.circle(image, (index_x, index_y), 10, (0, 0, 255), -1)

            distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)

            if st.session_state.test_started and not st.session_state.test_completed:
                elapsed = current_time - st.session_state.start_time
                if elapsed <= test_duration:
                    st.session_state.distance_data.append(distance)
                    st.session_state.distance_timestamps.append(elapsed)

            st.session_state.is_tapping = distance < distance_threshold
            if (st.session_state.is_tapping and not st.session_state.prev_tapping_state and 
                st.session_state.test_started and not st.session_state.test_completed):
                st.session_state.finger_tap_count += 1
                st.session_state.tap_timestamps.append(current_time)
                st.session_state.tap_effect_time = current_time

            if current_time - st.session_state.tap_effect_time < 0.1:
                cv2.circle(image, (thumb_x, thumb_y), 15, (0, 255, 255), -1)
                cv2.circle(image, (index_x, index_y), 15, (0, 255, 255), -1)

            st.session_state.prev_tapping_state = st.session_state.is_tapping
            cv2.putText(image, f"Distance: {distance:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if st.session_state.test_started and not st.session_state.test_completed:
        elapsed_time = current_time - st.session_state.start_time
        remaining_time = max(0, test_duration - elapsed_time)
        num_segments = int(test_duration / segment_duration)
        section = f"Segment {int(elapsed_time // segment_duration) + 1} of {num_segments}"
        cv2.putText(image, f"Time remaining: {remaining_time:.1f}s - {section}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Current taps: {st.session_state.finger_tap_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if elapsed_time >= test_duration:
            st.session_state.test_completed = True
            display_results()

    if st.session_state.test_completed:
        cv2.putText(image, "Test completed!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if not st.session_state.test_started and not st.session_state.test_completed:
        text = "Hand detected - Click 'Start Test'" if st.session_state.test_ready else "Show your hand to the camera"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Convert BGR to RGB for Streamlit display
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def display_results():
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.graph_filename = f"finger_distance_graph_{timestamp_filename}.png"

    overall_taps_per_second = st.session_state.finger_tap_count / test_duration
    num_segments = int(test_duration / segment_duration)
    segment_counts = [0] * num_segments
    segment_distances = [[] for _ in range(num_segments)]
    for timestamp in st.session_state.tap_timestamps:
        time_offset = timestamp - st.session_state.start_time
        segment_idx = min(int(time_offset // segment_duration), num_segments - 1)
        segment_counts[segment_idx] += 1
    for t, d in zip(st.session_state.distance_timestamps, st.session_state.distance_data):
        segment_idx = min(int(t // segment_duration), num_segments - 1)
        segment_distances[segment_idx].append(d)
    segment_rates = [count / segment_duration for count in segment_counts]
    segment_avg_distances = [sum(seg) / len(seg) if seg else 0 for seg in segment_distances]
    tap_change_percentage = ((segment_rates[-1] - segment_rates[0]) / segment_rates[0]) * 100 if segment_rates[0] > 0 else 0
    distance_change_percentage = ((segment_avg_distances[-1] - segment_avg_distances[0]) / segment_avg_distances[0]) * 100 if segment_avg_distances[0] > 0 else 0

    st.write(f"Total taps: {st.session_state.finger_tap_count}")
    st.write(f"Overall tap rate: {overall_taps_per_second:.2f} taps/second")
    for i in range(num_segments):
        st.write(f"Segment {i+1} ({i*5}-{(i+1)*5}s): {segment_rates[i]:.2f} taps/sec ({segment_counts[i]} taps), Avg Distance: {segment_avg_distances[i]:.1f}")
    st.write(f"Speed change (first to last): {tap_change_percentage:.1f}%")
    st.write(f"Distance change (first to last): {distance_change_percentage:.1f}%")

    # Generate final graph
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.suptitle('Fingertap Analysis', fontsize=16, fontweight='bold')
    plt.title(f'Test Time: {timestamp_str}', fontsize=16)
    ax1.set_xlabel('Time (seconds)', fontsize=16)
    ax1.set_ylabel('Finger Distance', color='blue', fontsize=16)
    # Plot raw distance with light blue line
    ax1.plot(st.session_state.distance_timestamps, st.session_state.distance_data, 'b-', alpha=0.3, label='Raw Distance')
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(0, test_duration)

    # Plot 2-second average distance with blue line
    time_bins = np.arange(0, test_duration + 2, 2)
    avg_distances = []
    for i in range(len(time_bins) - 1):
        mask = (np.array(st.session_state.distance_timestamps) >= time_bins[i]) & (np.array(st.session_state.distance_timestamps) < time_bins[i + 1])
        distances_in_bin = np.array(st.session_state.distance_data)[mask]
        avg_distances.append(np.mean(distances_in_bin) if len(distances_in_bin) > 0 else 0)
    ax1.plot(time_bins[:-1], avg_distances, 'b-', label='2-Second Avg Distance')

    # Plot tap events as green dots
    tap_times = [t - st.session_state.start_time for t in st.session_state.tap_timestamps]
    tap_distances = [st.session_state.distance_data[min(range(len(st.session_state.distance_timestamps)), 
             key=lambda i: abs(st.session_state.distance_timestamps[i] - t))] for t in tap_times]
    ax1.scatter(tap_times, tap_distances, color='green', label='Tap Events', s=50, alpha=0.7)

    # Calculate and plot 2-second tap rate with red line
    ax2 = ax1.twinx()
    ax2.set_ylabel('Tap Rate (taps/second)', color='red', fontsize=16)
    tap_rates = []
    if tap_times:  # Ensure tap_times is not empty
        for i in range(len(time_bins) - 1):
            mask = (np.array(tap_times) >= time_bins[i]) & (np.array(tap_times) < time_bins[i + 1])
            taps_in_bin = len(np.array(tap_times)[mask])
            tap_rates.append(taps_in_bin / 2.0)  # taps per second
        max_tap_rate = max(tap_rates) if tap_rates else 1.0  # Avoid division by zero
        ax2.set_ylim(0, max_tap_rate * 1.1)  # Dynamic range with 10% margin
        ax2.plot(time_bins[:-1], tap_rates, 'r-', label='2-Second Tap Rate', linewidth=2)
    else:
        st.write("No tap events detected to calculate tap rate.")

    # Grid and legend
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)

    # Add segment analysis at the bottom
    avg_dist_text = ", ".join([f"{d:.1f}" for d in segment_avg_distances])
    tap_rate_text = ", ".join([f"{r:.1f}" for r in segment_rates])
    change_dist = f"{distance_change_percentage:.1f}%"
    change_rate = f"{tap_change_percentage:.1f}%"
    plt.figtext(0.5, -0.05, f"Avg Distance (5s segments): {avg_dist_text}, Change: {change_dist}", 
                ha='center', fontsize=12)
    plt.figtext(0.5, -0.10, f"Tap Rate (5s segments): {tap_rate_text}, Change: {change_rate}", 
                ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.08, 1, 0.85])  # Adjust layout to accommodate text
    plt.savefig(st.session_state.graph_filename, dpi=300)
    st.image(st.session_state.graph_filename)
    with open(st.session_state.graph_filename, 'rb') as photo:
        bot.sendPhoto(CHAT_ID, photo)
    st.write(f"Graph sent to Telegram chat ID {CHAT_ID}")

# Button actions
if start_button and st.session_state.test_ready and not st.session_state.test_started and not st.session_state.test_completed:
    st.session_state.start_time = time.time()
    st.session_state.test_started = True
    st.write("Test started!")

if restart_button and st.session_state.test_completed:
    st.session_state.finger_tap_count = 0
    st.session_state.tap_timestamps = []
    st.session_state.distance_data = []
    st.session_state.distance_timestamps = []
    st.session_state.test_started = False
    st.session_state.test_completed = False
    st.session_state.test_ready = False
    st.session_state.graph_filename = ""
    st.session_state.tap_effect_time = 0
    st.write("Test restarted. Show your hand and click 'Start Test'.")

# Video feed loop
while True:
    frame = process_frame()
    if frame is not None:
        video_placeholder.image(frame, channels="RGB")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
hands.close()