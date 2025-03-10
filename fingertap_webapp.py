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
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Finger Tap Speed Test",
    page_icon="👆",
    layout="wide"
)

# Function to load camera settings
def load_camera_settings():
    # Default camera index
    default_camera = 0
    
    # Check if settings file exists
    if os.path.exists('camera_settings.json'):
        try:
            with open('camera_settings.json', 'r') as f:
                settings = json.load(f)
                return settings.get('camera_index', default_camera)
        except Exception as e:
            st.error(f"Error loading camera settings: {e}")
    
    return default_camera

# Function to save camera settings
def save_camera_settings(camera_index):
    try:
        with open('camera_settings.json', 'w') as f:
            settings = {'camera_index': camera_index}
            json.dump(settings, f)
        st.success(f"Camera settings saved. Using camera index: {camera_index}")
    except Exception as e:
        st.error(f"Error saving camera settings: {e}")

# Create a function to generate the analysis graph
def generate_analysis_graph(distance_timestamps, distance_data, tap_timestamps, start_time, test_duration=15, segment_duration=5):
    # Calculate overall results
    finger_tap_count = len(tap_timestamps)
    overall_taps_per_second = finger_tap_count / test_duration
    
    # Calculate results for each 5-second segment
    segment_counts = [0, 0, 0]  # Tap count for each 5-second segment
    
    for timestamp in tap_timestamps:
        time_offset = timestamp - start_time
        if time_offset < segment_duration:
            segment_counts[0] += 1
        elif time_offset < 2 * segment_duration:
            segment_counts[1] += 1
        else:
            segment_counts[2] += 1
    
    segment_rates = [count / segment_duration for count in segment_counts]
    
    # Calculate change rate between first and last 5 seconds for tap rate
    if segment_rates[0] > 0:  # Prevent division by zero
        tap_change_percentage = ((segment_rates[2] - segment_rates[0]) / segment_rates[0]) * 100
    else:
        tap_change_percentage = 0
    
    # Calculate 2-second interval tap rates
    two_sec_intervals = [0, 2, 4, 6, 8, 10, 12, 14]
    two_sec_tap_rates = []
    
    for i in range(len(two_sec_intervals)):
        start_sec = two_sec_intervals[i]
        end_sec = start_sec + 2 if start_sec + 2 <= test_duration else test_duration
        
        # Count taps in this interval
        count = sum(1 for t in tap_timestamps if start_sec <= (t - start_time) < end_sec)
        interval_duration = end_sec - start_sec
        
        # Calculate rate (taps per second)
        if interval_duration > 0:
            two_sec_tap_rates.append(count / interval_duration)
        else:
            two_sec_tap_rates.append(0)
    
    # Calculate average distance for each 5-second segment
    segment_distances = [[], [], []]
    
    for i, (timestamp, distance) in enumerate(zip(distance_timestamps, distance_data)):
        if timestamp < segment_duration:
            segment_distances[0].append(distance)
        elif timestamp < 2 * segment_duration:
            segment_distances[1].append(distance)
        else:
            segment_distances[2].append(distance)
    
    # Calculate average distance for each segment
    segment_avg_distances = []
    for segment in segment_distances:
        if segment:
            segment_avg_distances.append(sum(segment) / len(segment))
        else:
            segment_avg_distances.append(0)
    
    # Calculate distance change percentage
    if segment_avg_distances[0] > 0:
        distance_change_percentage = ((segment_avg_distances[2] - segment_avg_distances[0]) / segment_avg_distances[0]) * 100
    else:
        distance_change_percentage = 0
    
    # Calculate 2-second interval average distances
    two_sec_avg_distances = []
    
    for i in range(len(two_sec_intervals)):
        start_sec = two_sec_intervals[i]
        end_sec = start_sec + 2 if start_sec + 2 <= test_duration else test_duration
        
        # Get distances in this interval
        interval_distances = [d for t, d in zip(distance_timestamps, distance_data) if start_sec <= t < end_sec]
        
        # Calculate average distance
        if interval_distances:
            two_sec_avg_distances.append(sum(interval_distances) / len(interval_distances))
        else:
            two_sec_avg_distances.append(0)
    
    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Generate timestamp for subtitle
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Set figure title and subtitle
    plt.suptitle('Fingertap Analysis', fontsize=16, fontweight='bold')
    plt.title(f'Test Time: {timestamp_str}', fontsize=10)
    
    # First axis for tap rate and finger distance
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Finger Distance', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot raw distance data (light color)
    ax1.plot(distance_timestamps, distance_data, 'b-', alpha=0.3, label='Raw Distance')
    
    # Plot 2-second average distances
    ax1.plot(two_sec_intervals, two_sec_avg_distances, 'b-', linewidth=2, marker='o', 
            label='2-Second Avg Distance')
    
    # Set y-axis to start from 0
    ax1.set_ylim(bottom=0)
    
    # Create second y-axis for tap rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('Tap Rate (taps/second)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Plot 2-second tap rates
    ax2.plot(two_sec_intervals, two_sec_tap_rates, 'r-', linewidth=2, marker='x', 
            label='2-Second Tap Rate')
    
    # Set y-axis to start from 0
    ax2.set_ylim(bottom=0)
    
    # Mark tap events on the primary axis
    tap_times = [t - start_time for t in tap_timestamps if t - start_time <= test_duration]
    if tap_times:
        # For each tap, find the corresponding distance value
        tap_distances = []
        for tap_time in tap_times:
            # Fixed lambda function - using a default parameter to capture the current value
            closest_idx = min(range(len(distance_timestamps)), 
                              key=lambda i, current_time=tap_time: abs(distance_timestamps[i] - current_time))
            if closest_idx < len(distance_data):
                tap_distances.append(distance_data[closest_idx])
            else:
                tap_distances.append(0)
        
        ax1.scatter(tap_times, tap_distances, color='green', zorder=5, 
                   label='Tap Events', s=50, alpha=0.7)
    
    # Add tap rate segment averages and change info as text
    tap_info = f"Tap Rate (5s segments): {segment_rates[0]:.1f} - {segment_rates[1]:.1f} - {segment_rates[2]:.1f} taps/s, Change: {tap_change_percentage:.1f}%"
    fig.text(0.5, 0.02, tap_info, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add distance segment averages and change info as text
    distance_info = f"Avg Distance (5s segments): {segment_avg_distances[0]:.1f} - {segment_avg_distances[1]:.1f} - {segment_avg_distances[2]:.1f}, Change: {distance_change_percentage:.1f}%"
    fig.text(0.5, 0.05, distance_info, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add integer-only x ticks
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjust to make room for the stats at the bottom
    
    # Return the figure and results dictionary
    results = {
        "finger_tap_count": finger_tap_count,
        "overall_taps_per_second": overall_taps_per_second,
        "segment_rates": segment_rates,
        "tap_change_percentage": tap_change_percentage,
        "segment_avg_distances": segment_avg_distances,
        "distance_change_percentage": distance_change_percentage
    }
    
    return fig, results

# Initialize the app layout
st.title("Finger Tap Speed Test")
st.markdown("Test your finger tapping speed by tapping your thumb and index finger together for 15 seconds.")

# Create sidebar for settings
st.sidebar.title("Settings")

# Camera selection
camera_options = ["Camera 0", "Camera 1", "Camera 2", "Camera 3"]
default_camera = load_camera_settings()
selected_camera = st.sidebar.selectbox("Select Camera", camera_options, index=default_camera)
camera_index = int(selected_camera.split()[1])

# Save camera settings if changed
if camera_index != default_camera:
    save_camera_settings(camera_index)

# Initialize session state variables if they don't exist
if 'test_running' not in st.session_state:
    st.session_state.test_running = False
if 'test_complete' not in st.session_state:
    st.session_state.test_complete = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'finger_tap_count' not in st.session_state:
    st.session_state.finger_tap_count = 0
if 'tap_timestamps' not in st.session_state:
    st.session_state.tap_timestamps = []
if 'distance_data' not in st.session_state:
    st.session_state.distance_data = []
if 'distance_timestamps' not in st.session_state:
    st.session_state.distance_timestamps = []
if 'prev_tapping_state' not in st.session_state:
    st.session_state.prev_tapping_state = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'analysis_figure' not in st.session_state:
    st.session_state.analysis_figure = None

# Function to start the test
def start_test():
    st.session_state.test_running = True
    st.session_state.test_complete = False
    st.session_state.start_time = time.time()
    st.session_state.finger_tap_count = 0
    st.session_state.tap_timestamps = []
    st.session_state.distance_data = []
    st.session_state.distance_timestamps = []
    st.session_state.prev_tapping_state = False
    st.session_state.results = None
    st.session_state.analysis_figure = None

# Function to reset the test
def reset_test():
    st.session_state.test_running = False
    st.session_state.test_complete = False
    st.session_state.start_time = None
    st.session_state.finger_tap_count = 0
    st.session_state.tap_timestamps = []
    st.session_state.distance_data = []
    st.session_state.distance_timestamps = []
    st.session_state.prev_tapping_state = False
    st.session_state.results = None
    st.session_state.analysis_figure = None

# Test duration settings
test_duration = 15  # Total test duration in seconds
segment_duration = 5  # Each segment duration in seconds

# Create two columns for the main interface
col1, col2 = st.columns([2, 1])

# Create a placeholder for the video feed
with col1:
    video_placeholder = st.empty()
    info_placeholder = st.empty()

# Create placeholders for controls and results
with col2:
    status_placeholder = st.empty()
    controls_placeholder = st.empty()
    metrics_placeholder = st.empty()

# Add start/reset buttons
with controls_placeholder.container():
    cols = st.columns(2)
    with cols[0]:
        if not st.session_state.test_running and not st.session_state.test_complete:
            if st.button("Start Test", use_container_width=True):
                start_test()
    with cols[1]:
        if st.session_state.test_running or st.session_state.test_complete:
            if st.button("Reset", use_container_width=True):
                reset_test()

# Display analysis graph if test completed
if st.session_state.test_complete and st.session_state.analysis_figure is not None:
    st.pyplot(st.session_state.analysis_figure)

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Main app loop
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    st.error(f"Could not open camera {camera_index}. Please select a different camera.")
else:
    # Process video frame by frame
    while True:
        success, image = cap.read()
        if not success:
            st.error("Failed to read from camera. Please check camera connection and permissions.")
            break
        
        # Flip the image horizontally (mirror effect)
        image = cv2.flip(image, 1)
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        # Current time
        current_time = time.time()
        
        # Draw hand landmarks and process tap detection
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Get thumb and index finger tip coordinates
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Get image dimensions
                h, w, c = image.shape
                
                # Convert coordinates to pixels
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                
                # Draw circles at finger tips
                cv2.circle(image, (thumb_x, thumb_y), 10, (255, 0, 0), -1)  # Blue for thumb
                cv2.circle(image, (index_x, index_y), 10, (0, 0, 255), -1)  # Red for index
                
                # Calculate distance between fingers
                distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                
                # Add distance info to image
                cv2.putText(image, f"Distance: {distance:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Process data if test is running
                if st.session_state.test_running:
                    elapsed = current_time - st.session_state.start_time
                    
                    # Check if test should end
                    if elapsed >= test_duration:
                        st.session_state.test_running = False
                        st.session_state.test_complete = True
                        
                        # Generate analysis graph
                        if len(st.session_state.distance_timestamps) > 0 and len(st.session_state.tap_timestamps) > 0:
                            fig, results = generate_analysis_graph(
                                st.session_state.distance_timestamps,
                                st.session_state.distance_data,
                                st.session_state.tap_timestamps,
                                st.session_state.start_time,
                                test_duration,
                                segment_duration
                            )
                            st.session_state.analysis_figure = fig
                            st.session_state.results = results
                        break
                    
                    # Record distance data
                    st.session_state.distance_data.append(distance)
                    st.session_state.distance_timestamps.append(elapsed)
                    
                    # Detect tap (if distance is close enough)
                    is_tapping = distance < 100  # Threshold value
                    
                    # If tap state changed and currently tapping, increase count
                    if is_tapping and not st.session_state.prev_tapping_state:
                        st.session_state.finger_tap_count += 1
                        st.session_state.tap_timestamps.append(current_time)
                    
                    # Update previous tapping state
                    st.session_state.prev_tapping_state = is_tapping
                    
                    # Display progress info
                    remaining_time = max(0, test_duration - elapsed)
                    
                    if elapsed < segment_duration:
                        section = "First 5 seconds"
                    elif elapsed < 2 * segment_duration:
                        section = "Middle 5 seconds"
                    else:
                        section = "Last 5 seconds"
                    
                    # Add progress text to image
                    cv2.putText(image, f"Time remaining: {remaining_time:.1f}s - {section}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add tap count to image
                    cv2.putText(image, f"Current taps: {st.session_state.finger_tap_count}", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert image to RGB for Streamlit display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display the image in the placeholder
        video_placeholder.image(image_rgb, channels="RGB", use_column_width=True)
        
        # Update status and info
        if not st.session_state.test_running and not st.session_state.test_complete:
            status_placeholder.success("Ready to start test")
            info_placeholder.info("Show your hand to the camera and click 'Start Test'")
        elif st.session_state.test_running:
            elapsed = current_time - st.session_state.start_time
            remaining = max(0, test_duration - elapsed)
            progress = int((elapsed / test_duration) * 100)
            status_placeholder.progress(progress)
            info_placeholder.info(f"Test in progress: {remaining:.1f} seconds remaining")
        elif st.session_state.test_complete:
            status_placeholder.success("Test completed!")
            
            # Display metrics
            if st.session_state.results:
                with metrics_placeholder.container():
                    st.subheader("Test Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Taps", st.session_state.finger_tap_count)
                    with col2:
                        st.metric("Avg Taps/Second", f"{st.session_state.results['overall_taps_per_second']:.2f}")
                    with col3:
                        st.metric("Speed Change", f"{st.session_state.results['tap_change_percentage']:.1f}%")
                    
                    st.markdown("### Segment Analysis")
                    
                    # Create a DataFrame for segment analysis
                    segment_data = {
                        "Segment": ["First 5s", "Middle 5s", "Last 5s"],
                        "Tap Rate (taps/s)": [f"{rate:.2f}" for rate in st.session_state.results['segment_rates']],
                        "Avg Distance": [f"{dist:.1f}" for dist in st.session_state.results['segment_avg_distances']]
                    }
                    
                    st.dataframe(segment_data, use_container_width=True)
            
            # Generate and offer download link for graph
            if st.session_state.analysis_figure:
                buffer = io.BytesIO()
                st.session_state.analysis_figure.savefig(buffer, format='png', dpi=300)
                buffer.seek(0)
                
                # Generate timestamp for filename
                timestamp_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"finger_distance_graph_{timestamp_filename}.png"
                
                st.download_button(
                    label="Download Graph",
                    data=buffer,
                    file_name=filename,
                    mime="image/png"
                )
        
        # Break out of the loop to allow Streamlit to refresh the UI
        break

# Release resources
cap.release()
hands.close()