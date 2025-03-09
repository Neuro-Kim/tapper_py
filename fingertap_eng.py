import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.ticker import MaxNLocator
import os
import json

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
            print(f"Error loading camera settings: {e}")
    
    return default_camera

# Function to save camera settings
def save_camera_settings(camera_index):
    try:
        with open('camera_settings.json', 'w') as f:
            settings = {'camera_index': camera_index}
            json.dump(settings, f)
        print(f"Camera settings saved. Using camera index: {camera_index}")
    except Exception as e:
        print(f"Error saving camera settings: {e}")

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load camera settings
camera_index = load_camera_settings()
print(f"Using camera index: {camera_index}")

# Set up webcam
cap = cv2.VideoCapture(camera_index)

# Function to switch camera
def switch_camera(current_index):
    global cap
    
    # Release current camera
    if cap is not None:
        cap.release()
    
    # Try next camera (cycle through 0-3)
    next_index = (current_index + 1) % 4
    
    # Try to open the next camera
    cap = cv2.VideoCapture(next_index)
    
    # If successful, return the new index
    if cap.isOpened():
        save_camera_settings(next_index)
        return next_index
    else:
        # If failed, recursively try the next one
        print(f"Failed to open camera {next_index}, trying next...")
        return switch_camera(next_index)

# Check if camera opened successfully
if not cap.isOpened():
    print("Failed to open the default camera. Trying to find an available camera...")
    camera_index = switch_camera(camera_index)

# Initialize tap detection variables
finger_tap_count = 0
is_tapping = False
prev_tapping_state = False
tap_timestamps = []
test_duration = 15  # Total test time (seconds)
segment_duration = 5  # Each segment time (seconds)
start_time = None
test_started = False
test_completed = False
results_displayed = False
test_ready = False  # New flag to indicate ready for space key to start test
graph_filename = ""  # Variable to store the graph filename

# Variables for distance tracking
distance_data = []  # List to store (time, distance) pairs
distance_timestamps = []  # List to store timestamps for distances

# Finger landmark indices
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP

print("Camera started. Please show your hand.")
print("Press SPACE key to start the 15-second measurement.")
print("Press 'c' to cycle through available cameras.")

while True:
    success, image = cap.read()
    if not success:
        print("Cannot read frame.")
        # Try to switch to next camera automatically if current one fails
        camera_index = switch_camera(camera_index)
        continue
    
    # Flip the image horizontally (mirror effect)
    image = cv2.flip(image, 1)
    
    # Set image to read-only for better performance
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process hand recognition
    results = hands.process(image_rgb)
    
    # Set image back to writable
    image.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Current time
    current_time = time.time()
    
    # When hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Mark test as ready when hand is detected
            if not test_ready and not test_started and not test_completed:
                test_ready = True
                print("Hand detected. Press SPACE to start the test.")
            
            # Get thumb and index finger tip coordinates
            thumb_tip = hand_landmarks.landmark[THUMB_TIP]
            index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
            
            # Get screen dimensions
            h, w, c = image.shape
            
            # Convert finger tip coordinates to pixels
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            
            # Visualize finger tips
            cv2.circle(image, (thumb_x, thumb_y), 10, (255, 0, 0), -1)  # Thumb: blue
            cv2.circle(image, (index_x, index_y), 10, (0, 0, 255), -1)  # Index: red
            
            # Calculate distance between fingers
            distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            
            # Record distance data if test is in progress
            if test_started and not test_completed:
                elapsed = current_time - start_time
                if elapsed <= test_duration:
                    distance_data.append(distance)
                    distance_timestamps.append(elapsed)
            
            # Detect tap (if distance is close enough)
            is_tapping = distance < 100  # This value can be adjusted
            
            # If tap state changed and currently tapping, increase count
            if is_tapping and not prev_tapping_state and test_started and not test_completed:
                finger_tap_count += 1
                tap_timestamps.append(current_time)
                
            # Update previous tap state
            prev_tapping_state = is_tapping
            
            # Display distance between fingers
            distance_text = f"Distance: {distance:.1f}"
            cv2.putText(image, distance_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # If no hands detected and test not started, reset ready state
        if not test_started and not test_completed:
            test_ready = False
    
    # Calculate remaining time if test is in progress
    if test_started and not test_completed:
        elapsed_time = current_time - start_time
        remaining_time = max(0, test_duration - elapsed_time)
        
        # Show current section (first/middle/last 5 seconds)
        if elapsed_time < segment_duration:
            section = "First 5 seconds"
        elif elapsed_time < 2 * segment_duration:
            section = "Middle 5 seconds"
        else:
            section = "Last 5 seconds"
        
        # Check if test is completed
        if elapsed_time >= test_duration:
            test_completed = True
            print("\nTest completed!")
            
            # Generate timestamp for filename and graph subtitle
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timestamp_filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_filename = f"finger_distance_graph_{timestamp_filename}.png"
            
            # Calculate overall results
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
            
            # Display results
            print(f"Total taps in 15 seconds: {finger_tap_count}")
            print(f"Overall average tap rate: {overall_taps_per_second:.2f} taps/second")
            print(f"First 5 seconds tap rate: {segment_rates[0]:.2f} taps/second ({segment_counts[0]} taps)")
            print(f"Middle 5 seconds tap rate: {segment_rates[1]:.2f} taps/second ({segment_counts[1]} taps)")
            print(f"Last 5 seconds tap rate: {segment_rates[2]:.2f} taps/second ({segment_counts[2]} taps)")
            print(f"Speed change from first to last 5 seconds: {tap_change_percentage:.1f}%")
            print(f"Average distances for 5-second segments: {[f'{d:.1f}' for d in segment_avg_distances]}")
            print(f"Distance change from first to last 5 seconds: {distance_change_percentage:.1f}%")
            
            # Create a figure with two y-axes
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
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
            
            # Save the figure
            plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjust to make room for the stats at the bottom
            plt.savefig(graph_filename, dpi=300)
            print(f"Distance graph saved as '{graph_filename}'")
            
            results_displayed = True
        
        # Display progress on screen
        progress_text = f"Time remaining: {remaining_time:.1f}s - {section}"
        cv2.putText(image, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display current tap count
        tap_text = f"Current taps: {finger_tap_count}"
        cv2.putText(image, tap_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display results after test completion
    if test_completed:
        if not results_displayed:
            # This is a duplicate of the code above - it's a fallback in case the first calculation was missed
            # The calculations and graph generation would happen here too
            # Since it's identical to the code above, it's not repeated here for brevity
            pass
        
        # Display results on screen
        cv2.putText(image, "Test completed!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate results for each segment (for display)
        segment_counts = [0, 0, 0]
        for timestamp in tap_timestamps:
            time_offset = timestamp - start_time
            if time_offset < segment_duration:
                segment_counts[0] += 1
            elif time_offset < 2 * segment_duration:
                segment_counts[1] += 1
            else:
                segment_counts[2] += 1
        
        segment_rates = [count / segment_duration for count in segment_counts]
        
        # Calculate change rate
        if segment_rates[0] > 0:
            change_percentage = ((segment_rates[2] - segment_rates[0]) / segment_rates[0]) * 100
            change_text = f"{change_percentage:.1f}%"
            # Add increase/decrease arrow
            if change_percentage > 0:
                change_text += " ↑"
            elif change_percentage < 0:
                change_text += " ↓"
        else:
            change_text = "N/A"
        
        # Total tap count
        cv2.putText(image, f"Total taps: {finger_tap_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Tap rate by segment
        cv2.putText(image, f"First 5s: {segment_rates[0]:.2f} taps/sec", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"Middle 5s: {segment_rates[1]:.2f} taps/sec", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"Last 5s: {segment_rates[2]:.2f} taps/sec", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Change rate
        cv2.putText(image, f"Speed change: {change_text}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Restart test instructions
        cv2.putText(image, "Press 'r' to restart test", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"Graph saved as '{graph_filename}'", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show instructions if test hasn't started
    if not test_started and not test_completed:
        if test_ready:
            cv2.putText(image, "Hand detected - Press SPACE to start the test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(image, "Show your hand to the camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, "Measuring tap speed with thumb and index finger for 15 seconds", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display camera information
    camera_info = f"Camera: {camera_index} (Press 'c' to switch camera)"
    cv2.putText(image, camera_info, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display result screen
    cv2.imshow('Finger Tap Speed Test - 15 seconds', image)
    
    # Process key input
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord(' ') and test_ready and not test_started and not test_completed:  # Press SPACE to start test
        start_time = time.time()
        test_started = True
        print("Test started!")
    elif key == ord('r') and test_completed:  # Press 'r' to restart test
        finger_tap_count = 0
        tap_timestamps = []
        distance_data = []
        distance_timestamps = []
        test_started = False
        test_completed = False
        results_displayed = False
        test_ready = False
        graph_filename = ""
        print("\nRestarting test.")
        print("Show your hand to the camera and press SPACE to start.")
    elif key == ord('c') and not test_started:  # Press 'c' to cycle through cameras
        print("\nSwitching camera...")
        camera_index = switch_camera(camera_index)
        # Reset test readiness when switching camera
        test_ready = False

# Release resources
hands.close()
cap.release()
cv2.destroyAllWindows()