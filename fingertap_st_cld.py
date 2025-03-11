import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import threading
import base64
from io import BytesIO

# 페이지 설정
st.set_page_config(
    page_title="Fingertap Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일링
st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 0px;
}
.sub-title {
    font-size: 20px;
    color: #7f8c8d;
    text-align: center;
    margin-bottom: 20px;
}
.result-text {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}
.stButton button {
    width: 100%;
    border-radius: 5px;
    background-color: #3498db;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'camera_index' not in st.session_state:
    st.session_state.camera_index = 0
if 'test_started' not in st.session_state:
    st.session_state.test_started = False
if 'test_completed' not in st.session_state:
    st.session_state.test_completed = False
if 'test_ready' not in st.session_state:
    st.session_state.test_ready = False
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
if 'results_data' not in st.session_state:
    st.session_state.results_data = None
if 'graph_image' not in st.session_state:
    st.session_state.graph_image = None
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = None
if 'status_placeholder' not in st.session_state:
    st.session_state.status_placeholder = None
if 'results_placeholder' not in st.session_state:
    st.session_state.results_placeholder = None
if 'graph_placeholder' not in st.session_state:
    st.session_state.graph_placeholder = None
if 'cap' not in st.session_state:
    st.session_state.cap = None

# 테스트 상수
TEST_DURATION = 15  # 전체 테스트 시간 (초)
SEGMENT_DURATION = 5  # 각 세그먼트 시간 (초)

# MediaPipe 핸드 모듈 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 손가락 랜드마크 인덱스
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP

# 카메라 전환 함수
def switch_camera(camera_index):
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    
    # 새 카메라 열기
    cap = cv2.VideoCapture(camera_index)
    
    # 성공 여부 확인
    if cap.isOpened():
        st.session_state.camera_index = camera_index
        st.session_state.cap = cap
        return True
    else:
        st.error(f"카메라 {camera_index}를 열 수 없습니다.")
        return False

# 테스트 재시작 함수
def restart_test():
    st.session_state.finger_tap_count = 0
    st.session_state.tap_timestamps = []
    st.session_state.distance_data = []
    st.session_state.distance_timestamps = []
    st.session_state.test_started = False
    st.session_state.test_completed = False
    st.session_state.test_ready = False
    st.session_state.start_time = None
    st.session_state.results_data = None
    st.session_state.graph_image = None

# 테스트 시작 함수
def start_test():
    if st.session_state.test_ready and not st.session_state.test_started and not st.session_state.test_completed:
        st.session_state.start_time = time.time()
        st.session_state.test_started = True
        st.toast("테스트가 시작되었습니다!")

# 결과 계산 함수
def calculate_results():
    # 테스트가 완료되지 않았으면 계산하지 않음
    if not st.session_state.test_completed:
        return None
    
    # 타임스탬프 생성
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 전체 결과 계산
    tap_count = st.session_state.finger_tap_count
    overall_taps_per_second = tap_count / TEST_DURATION
    
    # 각 5초 구간별 결과 계산
    segment_counts = [0, 0, 0]  # 각 5초 구간의 탭 횟수
    
    for timestamp in st.session_state.tap_timestamps:
        time_offset = timestamp - st.session_state.start_time
        if time_offset < SEGMENT_DURATION:
            segment_counts[0] += 1
        elif time_offset < 2 * SEGMENT_DURATION:
            segment_counts[1] += 1
        else:
            segment_counts[2] += 1
    
    segment_rates = [count / SEGMENT_DURATION for count in segment_counts]
    
    # 첫 5초와 마지막 5초의 속도 변화율 계산
    if segment_rates[0] > 0:
        tap_change_percentage = ((segment_rates[2] - segment_rates[0]) / segment_rates[0]) * 100
    else:
        tap_change_percentage = 0
    
    # 2초 간격의 탭 속도 계산
    two_sec_intervals = [0, 2, 4, 6, 8, 10, 12, 14]
    two_sec_tap_rates = []
    
    for i in range(len(two_sec_intervals)):
        start_sec = two_sec_intervals[i]
        end_sec = start_sec + 2 if start_sec + 2 <= TEST_DURATION else TEST_DURATION
        
        # 이 구간의 탭 개수 계산
        count = sum(1 for t in st.session_state.tap_timestamps 
                   if start_sec <= (t - st.session_state.start_time) < end_sec)
        interval_duration = end_sec - start_sec
        
        # 속도 계산 (초당 탭 수)
        if interval_duration > 0:
            two_sec_tap_rates.append(count / interval_duration)
        else:
            two_sec_tap_rates.append(0)
    
    # 각 5초 구간별 평균 거리 계산
    segment_distances = [[], [], []]
    
    for i, (timestamp, distance) in enumerate(zip(st.session_state.distance_timestamps, 
                                                st.session_state.distance_data)):
        if timestamp < SEGMENT_DURATION:
            segment_distances[0].append(distance)
        elif timestamp < 2 * SEGMENT_DURATION:
            segment_distances[1].append(distance)
        else:
            segment_distances[2].append(distance)
    
    # 각 구간별 평균 거리 계산
    segment_avg_distances = []
    for segment in segment_distances:
        if segment:
            segment_avg_distances.append(sum(segment) / len(segment))
        else:
            segment_avg_distances.append(0)
    
    # 거리 변화 퍼센트 계산
    if segment_avg_distances[0] > 0:
        distance_change_percentage = ((segment_avg_distances[2] - segment_avg_distances[0]) / 
                                     segment_avg_distances[0]) * 100
    else:
        distance_change_percentage = 0
    
    # 2초 간격의 평균 거리 계산
    two_sec_avg_distances = []
    
    for i in range(len(two_sec_intervals)):
        start_sec = two_sec_intervals[i]
        end_sec = start_sec + 2 if start_sec + 2 <= TEST_DURATION else TEST_DURATION
        
        # 이 구간의 거리 가져오기
        interval_distances = [d for t, d in zip(st.session_state.distance_timestamps, 
                                              st.session_state.distance_data) 
                             if start_sec <= t < end_sec]
        
        # 평균 거리 계산
        if interval_distances:
            two_sec_avg_distances.append(sum(interval_distances) / len(interval_distances))
        else:
            two_sec_avg_distances.append(0)
    
    # 결과 데이터 반환
    return {
        'timestamp': timestamp_str,
        'tap_count': tap_count,
        'overall_taps_per_second': overall_taps_per_second,
        'segment_counts': segment_counts,
        'segment_rates': segment_rates,
        'tap_change_percentage': tap_change_percentage,
        'segment_avg_distances': segment_avg_distances,
        'distance_change_percentage': distance_change_percentage,
        'two_sec_intervals': two_sec_intervals,
        'two_sec_tap_rates': two_sec_tap_rates,
        'two_sec_avg_distances': two_sec_avg_distances,
        'tap_timestamps': [t - st.session_state.start_time for t in st.session_state.tap_timestamps 
                          if t - st.session_state.start_time <= TEST_DURATION],
        'distance_timestamps': st.session_state.distance_timestamps,
        'distance_data': st.session_state.distance_data
    }

# 그래프 생성 함수
def create_results_graph(results):
    if not results:
        return None
    
    # 그림 생성
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 그림 제목과 부제목 설정
    plt.suptitle('Fingertap Analysis', fontsize=16, fontweight='bold')
    plt.title(f'Test Time: {results["timestamp"]}', fontsize=14)
    
    # 첫 번째 축: 탭 속도와 손가락 거리
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Finger Distance', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # 원시 거리 데이터 그리기 (연한 색상)
    ax1.plot(results['distance_timestamps'], results['distance_data'], 'b-', 
            alpha=0.3, label='Raw Distance')
    
    # 2초 평균 거리 그리기
    ax1.plot(results['two_sec_intervals'], results['two_sec_avg_distances'], 'b-', 
            linewidth=2, marker='o', label='2-Second Avg Distance')
    
    # y축을 0에서 시작하도록 설정
    ax1.set_ylim(bottom=0)
    
    # 두 번째 y축 생성 (탭 속도용)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Tap Rate (taps/second)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 2초 탭 속도 그리기
    ax2.plot(results['two_sec_intervals'], results['two_sec_tap_rates'], 'r-', 
            linewidth=2, marker='x', label='2-Second Tap Rate')
    
    # y축을 0에서 시작하도록 설정
    ax2.set_ylim(bottom=0)
    
    # 탭 이벤트 표시
    if results['tap_timestamps']:
        # 각 탭에 대해 해당 거리 값 찾기
        tap_distances = []
        for tap_time in results['tap_timestamps']:
            closest_idx = min(range(len(results['distance_timestamps'])), 
                             key=lambda i: abs(results['distance_timestamps'][i] - tap_time))
            if closest_idx < len(results['distance_data']):
                tap_distances.append(results['distance_data'][closest_idx])
            else:
                tap_distances.append(0)
        
        ax1.scatter(results['tap_timestamps'], tap_distances, color='green', 
                   zorder=5, label='Tap Events', s=50, alpha=0.7)
    
    # 탭 속도 구간 평균 및 변화 정보 텍스트 추가
    segment_rates = results['segment_rates']
    tap_change_percentage = results['tap_change_percentage']
    tap_info = f"Tap Rate (5s segments): {segment_rates[0]:.1f} - {segment_rates[1]:.1f} - {segment_rates[2]:.1f} taps/s, Change: {tap_change_percentage:.1f}%"
    fig.text(0.5, 0.02, tap_info, ha='center', fontsize=16)
    
    # 거리 구간 평균 및 변화 정보 텍스트 추가
    segment_avg_distances = results['segment_avg_distances']
    distance_change_percentage = results['distance_change_percentage']
    distance_info = f"Avg Distance (5s segments): {segment_avg_distances[0]:.1f} - {segment_avg_distances[1]:.1f} - {segment_avg_distances[2]:.1f}, Change: {distance_change_percentage:.1f}%"
    fig.text(0.5, 0.05, distance_info, ha='center', fontsize=16)
    
    # 그리드 추가
    ax1.grid(True, alpha=0.3)
    
    # 정수만 표시하는 x 눈금 설정
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 통합 범례 생성
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 그림 조정
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # 하단 통계용 공간 확보
    
    # 이미지를 바이트 스트림으로 변환
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    
    # 그림 닫기
    plt.close(fig)
    
    return buffer

# 카메라 처리 함수
def process_camera_feed():
    # 캡처 객체 확인
    if st.session_state.cap is None:
        return None, "카메라가 연결되지 않았습니다. 카메라를 선택해주세요."
    
    # 프레임 읽기
    success, image = st.session_state.cap.read()
    if not success:
        return None, "프레임을 읽을 수 없습니다. 카메라를 다시 선택해주세요."
    
    # 이미지 좌우 반전 (거울 효과)
    image = cv2.flip(image, 1)
    
    # 현재 시간
    current_time = time.time()
    
    # MediaPipe 손 인식 처리
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        # 이미지를 읽기 전용으로 설정
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 손 인식 처리
        results = hands.process(image_rgb)
        
        # 이미지를 다시 쓰기 가능하게 설정
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # 손이 감지되었을 때
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # 손이 감지되면 테스트 준비 상태로 설정
                if not st.session_state.test_ready and not st.session_state.test_started and not st.session_state.test_completed:
                    st.session_state.test_ready = True
                
                # 엄지와 검지 손가락 끝 좌표 가져오기
                thumb_tip = hand_landmarks.landmark[THUMB_TIP]
                index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
                
                # 화면 크기 가져오기
                h, w, c = image.shape
                
                # 손가락 끝 좌표를 픽셀로 변환
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                
                # 손가락 끝 시각화
                cv2.circle(image, (thumb_x, thumb_y), 10, (255, 0, 0), -1)  # 엄지: 파란색
                cv2.circle(image, (index_x, index_y), 10, (0, 0, 255), -1)  # 검지: 빨간색
                
                # 손가락 사이 거리 계산
                distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                
                # 테스트 진행 중이면 거리 데이터 기록
                if st.session_state.test_started and not st.session_state.test_completed:
                    elapsed = current_time - st.session_state.start_time
                    if elapsed <= TEST_DURATION:
                        st.session_state.distance_data.append(distance)
                        st.session_state.distance_timestamps.append(elapsed)
                
                # 탭 감지 (거리가 충분히 가까우면)
                is_tapping = distance < 100  # 이 값은 조정 가능
                
                # 탭 상태가 변했고 현재 탭 중이면 카운트 증가
                if is_tapping and not st.session_state.prev_tapping_state and st.session_state.test_started and not st.session_state.test_completed:
                    st.session_state.finger_tap_count += 1
                    st.session_state.tap_timestamps.append(current_time)
                
                # 이전 탭 상태 업데이트
                st.session_state.prev_tapping_state = is_tapping
                
                # 손가락 사이 거리 표시
                distance_text = f"Distance: {distance:.1f}"
                cv2.putText(image, distance_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 손이 감지되지 않고 테스트가 시작되지 않았으면 준비 상태 리셋
            if not st.session_state.test_started and not st.session_state.test_completed:
                st.session_state.test_ready = False
        
        # 테스트 진행 중일 때 처리
        if st.session_state.test_started and not st.session_state.test_completed:
            elapsed_time = current_time - st.session_state.start_time
            remaining_time = max(0, TEST_DURATION - elapsed_time)
            
            # 현재 구간 표시 (첫/중간/마지막 5초)
            if elapsed_time < SEGMENT_DURATION:
                section = "First 5 seconds"
            elif elapsed_time < 2 * SEGMENT_DURATION:
                section = "Middle 5 seconds"
            else:
                section = "Last 5 seconds"
            
            # 테스트 완료 확인
            if elapsed_time >= TEST_DURATION:
                st.session_state.test_completed = True
                
                # 결과 계산 및 그래프 생성
                st.session_state.results_data = calculate_results()
                graph_buffer = create_results_graph(st.session_state.results_data)
                if graph_buffer:
                    st.session_state.graph_image = graph_buffer
            
            # 화면에 진행 상황 표시
            progress_text = f"Time remaining: {remaining_time:.1f}s - {section}"
            cv2.putText(image, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 현재 탭 횟수 표시
            tap_text = f"Current taps: {st.session_state.finger_tap_count}"
            cv2.putText(image, tap_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 테스트 완료 후 처리
        elif st.session_state.test_completed:
            # 결과가 없으면 계산
            if st.session_state.results_data is None:
                st.session_state.results_data = calculate_results()
                graph_buffer = create_results_graph(st.session_state.results_data)
                if graph_buffer:
                    st.session_state.graph_image = graph_buffer
            
            # 화면에 결과 표시
            cv2.putText(image, "Test completed!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 결과 데이터 가져오기
            results = st.session_state.results_data
            if results:
                segment_rates = results['segment_rates']
                tap_change_percentage = results['tap_change_percentage']
                
                # 변화율 텍스트 계산
                change_text = f"{tap_change_percentage:.1f}%"
                if tap_change_percentage > 0:
                    change_text += " ↑"
                elif tap_change_percentage < 0:
                    change_text += " ↓"
                
                # 화면에 결과 표시
                cv2.putText(image, f"Total taps: {results['tap_count']}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, f"First 5s: {segment_rates[0]:.2f} taps/sec", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, f"Middle 5s: {segment_rates[1]:.2f} taps/sec", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, f"Last 5s: {segment_rates[2]:.2f} taps/sec", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, f"Speed change: {change_text}", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 테스트 재시작 안내
            cv2.putText(image, "Press 'r' to restart test", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 테스트 시작 전 안내
        elif not st.session_state.test_started:
            if st.session_state.test_ready:
                cv2.putText(image, "Hand detected - Press SPACE to start the test", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(image, "Show your hand to the camera", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, "Measuring tap speed with thumb and index finger for 15 seconds", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 카메라 정보 표시
        camera_info = f"Camera: {st.session_state.camera_index}"
        cv2.putText(image, camera_info, (10, image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 이미지를 RGB로 변환 (Streamlit 표시용)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None

# 키 입력 처리 함수
def handle_key_press():
    # 키 입력 처리
    pressed_key = cv2.waitKey(1) & 0xFF
    
    # r 키: 테스트 재시작
    if pressed_key == ord('r') and st.session_state.test_completed:
        restart_test()
    
    # 스페이스바: 테스트 시작
    elif pressed_key == ord(' ') and st.session_state.test_ready and not st.session_state.test_started and not st.session_state.test_completed:
        start_test()

# 메인 UI 구성
def main():
    # 사이드바 설정
    with st.sidebar:
        st.title("카메라 설정")
        st.markdown("사용할 카메라를 선택하세요:")
        
        # 카메라 버튼
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("카메라 1"):
                switch_camera(0)
                restart_test()
        
        with col2:
            if st.button("카메라 2"):
                switch_camera(1)
                restart_test()
        
        with col3:
            if st.button("카메라 3"):
                switch_camera(2)
                restart_test()
                
        # 테스트 제어 버튼
        st.markdown("---")
        if st.button("테스트 시작", key="start_test", disabled=not st.session_state.test_ready or st.session_state.test_started or st.session_state.test_completed):
            start_test()
            
        if st.button("테스트 재시작", key="restart_test", disabled=not st.session_state.test_completed):
            restart_test()
        
        # 키보드 단축키 안내
        st.markdown("---")
        st.markdown("### 단축키 안내")
        st.markdown("- **스페이스바**: 테스트 시작")
        st.markdown("- **R 키**: 테스트 재시작")
    
    # 메인 화면 설정
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 타이틀과 서브타이틀
        st.markdown('<p class="main-title">핑거탭 분석</p>', unsafe_allow_html=True)
        
        if st.session_state.test_started and not st.session_state.test_completed:
            remaining_time = max(0, TEST_DURATION - (time.time() - st.session_state.start_time))
            st.markdown(f'<p class="sub-title">테스트 진행 중... {remaining_time:.1f}초 남음</p>', unsafe_allow_html=True)
        elif st.session_state.test_completed:
            st.markdown('<p class="sub-title">테스트 완료!</p>', unsafe_allow_html=True)
        elif st.session_state.test_ready:
            st.markdown('<p class="sub-title">스페이스 키를 눌러 시작하세요</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="sub-title">카메라에 손을 보여주세요</p>', unsafe_allow_html=True)
        
        # 카메라 프레임용 placeholder
        frame_placeholder = st.empty()
        
        # 상태 메시지용 placeholder
        status_placeholder = st.empty()
    
    with col2:
        # 결과 표시용 placeholder
        if st.session_state.test_completed and st.session_state.results_data:
            results = st.session_state.results_data
            
            st.markdown("### 분석 결과")
            st.markdown('<div class="result-text">', unsafe_allow_html=True)
            
            st.markdown(f"**총 탭 횟수:** {results['tap_count']}회")
            st.markdown(f"**평균 탭 속도:** {results['overall_taps_per_second']:.2f} 탭/초")
            
            st.markdown("**5초 구간별 탭 속도:**")
            st.markdown(f"- 첫 5초: {results['segment_rates'][0]:.2f} 탭/초 ({results['segment_counts'][0]}회)")
            st.markdown(f"- 중간 5초: {results['segment_rates'][1]:.2f} 탭/초 ({results['segment_counts'][1]}회)")
            st.markdown(f"- 마지막 5초: {results['segment_rates'][2]:.2f} 탭/초 ({results['segment_counts'][2]}회)")
            
            # 변화율 및 화살표 표시
            tap_change = results['tap_change_percentage']
            arrow = "↑" if tap_change > 0 else "↓" if tap_change < 0 else "→"
            st.markdown(f"**속도 변화율:** {tap_change:.1f}% {arrow}")
            
            # 거리 정보
            st.markdown("**손가락 사이 평균 거리:**")
            for i, dist in enumerate(results['segment_avg_distances']):
                segment_name = ["첫 5초", "중간 5초", "마지막 5초"][i]
                st.markdown(f"- {segment_name}: {dist:.1f}")
            
            # 거리 변화율 및 화살표 표시
            dist_change = results['distance_change_percentage']
            arrow = "↑" if dist_change > 0 else "↓" if dist_change < 0 else "→"
            st.markdown(f"**거리 변화율:** {dist_change:.1f}% {arrow}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
    # 그래프 표시용 placeholder
    graph_placeholder = st.empty()
    
    # 결과 그래프 표시
    if st.session_state.test_completed and st.session_state.graph_image:
        with graph_placeholder:
            st.image(st.session_state.graph_image, caption="핑거탭 분석 그래프", use_column_width=True)
    
    # 카메라 관련 함수 실행
    if 'cap' not in st.session_state or st.session_state.cap is None:
        switch_camera(st.session_state.camera_index)
    
    # 프레임 처리 및 표시
    if st.session_state.cap is not None:
        # 프레임 처리
        frame, error_message = process_camera_feed()
        
        if frame is not None:
            # 프레임 표시
            with frame_placeholder:
                st.image(frame, channels="RGB", use_column_width=True)
        elif error_message:
            # 에러 메시지 표시
            with status_placeholder:
                st.error(error_message)
    
    # 키 처리 - Streamlit에서는 직접적인 키 이벤트를 처리할 수 없으므로
    # 버튼 인터페이스에 의존하거나 JavaScript를 사용해야 함
    # 여기서는 버튼 인터페이스를 사용

# 앱 실행
if __name__ == "__main__":
    main()