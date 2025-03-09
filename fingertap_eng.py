import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Set up webcam
cap = cv2.VideoCapture(1)

# Check if camera opened successfully
if not cap.isOpened():
    print("Cannot open camera. Please check camera permissions.")
    exit()

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

while True:
    success, image = cap.read()
    if not success:
        print("Cannot read frame.")
        break
    
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
            
            # Generate timestamp for filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_filename = f"finger_distance_graph_{timestamp}.png"
            
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
            
            # Calculate change rate between first and last 5 seconds
            if segment_rates[0] > 0:  # Prevent division by zero
                change_percentage = ((segment_rates[2] - segment_rates[0]) / segment_rates[0]) * 100
            else:
                change_percentage = 0
            
            # Display results
            print(f"Total taps in 15 seconds: {finger_tap_count}")
            print(f"Overall average tap rate: {overall_taps_per_second:.2f} taps/second")
            print(f"First 5 seconds tap rate: {segment_rates[0]:.2f} taps/second ({segment_counts[0]} taps)")
            print(f"Middle 5 seconds tap rate: {segment_rates[1]:.2f} taps/second ({segment_counts[1]} taps)")
            print(f"Last 5 seconds tap rate: {segment_rates[2]:.2f} taps/second ({segment_counts[2]} taps)")
            print(f"Speed change from first to last 5 seconds: {change_percentage:.1f}%")
            
            # Generate and save distance graph
            plt.figure(figsize=(10, 6))
            plt.plot(distance_timestamps, distance_data, 'b-')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Distance between fingers')
            plt.title('Finger Distance over 15 Seconds')
            plt.grid(True)
            
            # Mark tap events on the graph
            tap_times = [t - start_time for t in tap_timestamps if t - start_time <= test_duration]
            if tap_times:
                # Find the corresponding distance values for tap times
                tap_distances = []
                for tap_time in tap_times:
                    # Find the closest timestamp in distance_timestamps
                    closest_idx = min(range(len(distance_timestamps)), 
                                      key=lambda i: abs(distance_timestamps[i] - tap_time))
                    if closest_idx < len(distance_data):
                        tap_distances.append(distance_data[closest_idx])
                    else:
                        tap_distances.append(0)  # Fallback value
                
                plt.scatter(tap_times, tap_distances, color='red', zorder=5, 
                           label='Tap Events', s=50)
                plt.legend()
            
            plt.savefig(graph_filename)
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
            # Generate timestamp for filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_filename = f"finger_distance_graph_{timestamp}.png"
            
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
            
            # Calculate change rate between first and last 5 seconds
            if segment_rates[0] > 0:  # Prevent division by zero
                change_percentage = ((segment_rates[2] - segment_rates[0]) / segment_rates[0]) * 100
            else:
                change_percentage = 0
            
            # Display results
            print(f"\nTest completed!")
            print(f"Total taps in 15 seconds: {finger_tap_count}")
            print(f"Overall average tap rate: {overall_taps_per_second:.2f} taps/second")
            print(f"First 5 seconds tap rate: {segment_rates[0]:.2f} taps/second ({segment_counts[0]} taps)")
            print(f"Middle 5 seconds tap rate: {segment_rates[1]:.2f} taps/second ({segment_counts[1]} taps)")
            print(f"Last 5 seconds tap rate: {segment_rates[2]:.2f} taps/second ({segment_counts[2]} taps)")
            print(f"Speed change from first to last 5 seconds: {change_percentage:.1f}%")
            
            # Generate and save distance graph
            plt.figure(figsize=(10, 6))
            plt.plot(distance_timestamps, distance_data, 'b-')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Distance between fingers')
            plt.title('Finger Distance over 15 Seconds')
            plt.grid(True)
            
            # Mark tap events on the graph
            tap_times = [t - start_time for t in tap_timestamps if t - start_time <= test_duration]
            if tap_times:
                # Find the corresponding distance values for tap times
                tap_distances = []
                for tap_time in tap_times:
                    # Find the closest timestamp in distance_timestamps
                    closest_idx = min(range(len(distance_timestamps)), 
                                      key=lambda i: abs(distance_timestamps[i] - tap_time))
                    if closest_idx < len(distance_data):
                        tap_distances.append(distance_data[closest_idx])
                    else:
                        tap_distances.append(0)  # Fallback value
                
                plt.scatter(tap_times, tap_distances, color='red', zorder=5, 
                           label='Tap Events', s=50)
                plt.legend()
            
            plt.savefig(graph_filename)
            print(f"Distance graph saved as '{graph_filename}'")
            
            results_displayed = True
        
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

# Release resources
hands.close()
cap.release()
cv2.destroyAllWindows()