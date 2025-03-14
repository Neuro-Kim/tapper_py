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
from PIL import Image

# 세션 상태 초기화
if 'test_started' not in st.session_state:
    st.session_state.test_started = False
    st.session_state.test_completed = False
    st.session_state.finger_tap_count = 0
    st.session_state.tap_timestamps = []
    st.session_state.distance_data = []
    st.session_state.distance_timestamps = []
    st.session_state.start_time = None
    st.session_state.graph_filename = ""
    st.session_state.is_tapping = False
    st.session_state.prev_tapping_state = False
    st.session_state.tap_effect_time = 0
    st.session_state.last_processed_image = None
    st.session_state.current_frame = None

# 텔레그램 설정 로드
try:
    with open('telegram_config.json', 'r') as config_file:
        config = json.load(config_file)
        TOKEN = config['TOKEN']
        CHAT_ID = config['CHAT_ID']
    # 텔레그램 봇 초기화
    bot = telepot.Bot(TOKEN)
except Exception as e:
    st.sidebar.warning(f"텔레그램 설정을 로드할 수 없습니다: {e}")
    st.sidebar.info("텔레그램 알림을 사용하지 않고 계속합니다.")
    TOKEN = None
    CHAT_ID = None
    bot = None

# MediaPipe 손 모듈 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # 이미지 입력에 맞게 True로 변경
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 상수
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
segment_duration = 5  # 5초 세그먼트

# 앱 제목
st.title("손가락 태핑 속도 테스트")
st.write("웹캠을 사용하여 손가락 태핑 속도를 측정하세요!")

# 사이드바 설정
st.sidebar.title("설정")
test_duration = st.sidebar.slider("테스트 시간 (초)", 5, 30, 15)
distance_threshold = st.sidebar.slider("팁 거리 임계값", 0, 100, 50)

# 이미지 처리 함수
def process_image(image):
    if image is None:
        return None
    
    # PIL 이미지를 NumPy 배열로 변환
    image_np = np.array(image)
    
    # RGB에서 BGR로 변환 (MediaPipe는 RGB 입력 필요)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGB)
    
    # 손 감지
    results = hands.process(image_rgb)
    
    # 처리 결과를 이미지에 그리기
    annotated_image = image_np.copy()
    current_time = time.time()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(
                annotated_image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # 엄지와 검지 위치 계산
            thumb_tip = hand_landmarks.landmark[THUMB_TIP]
            index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
            h, w, _ = annotated_image.shape
            
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            
            # 원 그리기
            cv2.circle(annotated_image, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
            cv2.circle(annotated_image, (index_x, index_y), 10, (0, 0, 255), -1)
            
            # 거리 계산
            distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            
            # 테스트 중인 경우 데이터 기록
            if st.session_state.test_started and not st.session_state.test_completed:
                elapsed = current_time - st.session_state.start_time
                if elapsed <= test_duration:
                    st.session_state.distance_data.append(distance)
                    st.session_state.distance_timestamps.append(elapsed)
            
            # 태핑 감지
            st.session_state.is_tapping = distance < distance_threshold
            if (st.session_state.is_tapping and not st.session_state.prev_tapping_state and 
                st.session_state.test_started and not st.session_state.test_completed):
                st.session_state.finger_tap_count += 1
                st.session_state.tap_timestamps.append(current_time)
                st.session_state.tap_effect_time = current_time
            
            # 태핑 효과
            if current_time - st.session_state.tap_effect_time < 0.1:
                cv2.circle(annotated_image, (thumb_x, thumb_y), 15, (0, 255, 255), -1)
                cv2.circle(annotated_image, (index_x, index_y), 15, (0, 255, 255), -1)
            
            st.session_state.prev_tapping_state = st.session_state.is_tapping
            cv2.putText(annotated_image, f"거리: {distance:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 테스트 상태 표시
    if st.session_state.test_started and not st.session_state.test_completed:
        elapsed_time = current_time - st.session_state.start_time
        remaining_time = max(0, test_duration - elapsed_time)
        num_segments = int(test_duration / segment_duration)
        section = f"세그먼트 {int(elapsed_time // segment_duration) + 1}/{num_segments}"
        cv2.putText(annotated_image, f"남은 시간: {remaining_time:.1f}초 - {section}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_image, f"현재 탭 수: {st.session_state.finger_tap_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if elapsed_time >= test_duration:
            st.session_state.test_completed = True
    
    if st.session_state.test_completed:
        cv2.putText(annotated_image, "테스트 완료!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if not st.session_state.test_started and not st.session_state.test_completed:
        cv2.putText(annotated_image, "손을 보여주고 '테스트 시작'을 클릭하세요", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return annotated_image

# 결과 표시 함수
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

    st.write(f"총 탭 수: {st.session_state.finger_tap_count}")
    st.write(f"전체 탭 속도: {overall_taps_per_second:.2f} 탭/초")
    
    for i in range(num_segments):
        st.write(f"세그먼트 {i+1} ({i*5}-{(i+1)*5}초): {segment_rates[i]:.2f} 탭/초 ({segment_counts[i]} 탭), 평균 거리: {segment_avg_distances[i]:.1f}")
    
    st.write(f"속도 변화 (처음에서 마지막까지): {tap_change_percentage:.1f}%")
    st.write(f"거리 변화 (처음에서 마지막까지): {distance_change_percentage:.1f}%")

    # 최종 그래프 생성
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.suptitle('손가락 태핑 분석', fontsize=16, fontweight='bold')
    plt.title(f'테스트 시간: {timestamp_str}', fontsize=16)
    ax1.set_xlabel('시간 (초)', fontsize=16)
    ax1.set_ylabel('손가락 거리', color='blue', fontsize=16)
    
    # 원시 거리 데이터 그리기
    ax1.plot(st.session_state.distance_timestamps, st.session_state.distance_data, 'b-', alpha=0.3, label='원시 거리')
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(0, test_duration)

    # 2초 평균 거리 그리기
    time_bins = np.arange(0, test_duration + 2, 2)
    avg_distances = []
    for i in range(len(time_bins) - 1):
        mask = (np.array(st.session_state.distance_timestamps) >= time_bins[i]) & (np.array(st.session_state.distance_timestamps) < time_bins[i + 1])
        distances_in_bin = np.array(st.session_state.distance_data)[mask]
        avg_distances.append(np.mean(distances_in_bin) if len(distances_in_bin) > 0 else 0)
    ax1.plot(time_bins[:-1], avg_distances, 'b-', label='2초 평균 거리')

    # 탭 이벤트를 녹색 점으로 표시
    tap_times = [t - st.session_state.start_time for t in st.session_state.tap_timestamps]
    if tap_times and st.session_state.distance_timestamps:
        tap_distances = []
        for t in tap_times:
            closest_idx = min(range(len(st.session_state.distance_timestamps)), 
                              key=lambda i: abs(st.session_state.distance_timestamps[i] - t))
            tap_distances.append(st.session_state.distance_data[closest_idx])
        ax1.scatter(tap_times, tap_distances, color='green', label='탭 이벤트', s=50, alpha=0.7)

    # 2초 탭 속도 계산 및 빨간색 선으로 표시
    ax2 = ax1.twinx()
    ax2.set_ylabel('탭 속도 (탭/초)', color='red', fontsize=16)
    tap_rates = []
    if tap_times:  # tap_times가 비어있지 않은지 확인
        for i in range(len(time_bins) - 1):
            mask = (np.array(tap_times) >= time_bins[i]) & (np.array(tap_times) < time_bins[i + 1])
            taps_in_bin = len(np.array(tap_times)[mask])
            tap_rates.append(taps_in_bin / 2.0)  # 초당 탭 수
        max_tap_rate = max(tap_rates) if tap_rates else 1.0  # 0으로 나누기 방지
        ax2.set_ylim(0, max_tap_rate * 1.1)  # 10% 여유를 둔 동적 범위
        ax2.plot(time_bins[:-1], tap_rates, 'r-', label='2초 탭 속도', linewidth=2)
    else:
        st.write("탭 이벤트가 감지되지 않아 탭 속도를 계산할 수 없습니다.")

    # 그리드와 범례
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)

    # 하단에 세그먼트 분석 추가
    avg_dist_text = ", ".join([f"{d:.1f}" for d in segment_avg_distances])
    tap_rate_text = ", ".join([f"{r:.1f}" for r in segment_rates])
    change_dist = f"{distance_change_percentage:.1f}%"
    change_rate = f"{tap_change_percentage:.1f}%"
    plt.figtext(0.5, -0.05, f"평균 거리 (5초 세그먼트): {avg_dist_text}, 변화: {change_dist}", 
                ha='center', fontsize=12)
    plt.figtext(0.5, -0.10, f"탭 속도 (5초 세그먼트): {tap_rate_text}, 변화: {change_rate}", 
                ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.08, 1, 0.85])  # 텍스트 공간 확보를 위한 레이아웃 조정
    
    # 그래프 저장 및 표시
    fig.savefig(st.session_state.graph_filename, dpi=300)
    st.image(st.session_state.graph_filename)
    
    # 텔레그램으로 전송
    if bot and CHAT_ID:
        try:
            with open(st.session_state.graph_filename, 'rb') as photo:
                bot.sendPhoto(CHAT_ID, photo)
            st.write(f"그래프가 텔레그램 채팅 ID {CHAT_ID}로 전송되었습니다")
        except Exception as e:
            st.error(f"텔레그램으로 전송 중 오류 발생: {e}")

# UI 구성
col1, col2 = st.columns(2)
start_button = col1.button("테스트 시작" if not st.session_state.test_started else "테스트 진행 중")
restart_button = col2.button("테스트 재시작", disabled=not st.session_state.test_completed)

# 카메라 입력 위젯
img_file_buffer = st.camera_input("카메라로 손을 보여주세요")

# 버튼 동작
if start_button and not st.session_state.test_started and not st.session_state.test_completed:
    if img_file_buffer is not None:
        st.session_state.start_time = time.time()
        st.session_state.test_started = True
        st.write("테스트가 시작되었습니다!")
    else:
        st.warning("테스트를 시작하기 전에 카메라를 활성화하고 이미지를 캡처해주세요.")

if restart_button and st.session_state.test_completed:
    st.session_state.finger_tap_count = 0
    st.session_state.tap_timestamps = []
    st.session_state.distance_data = []
    st.session_state.distance_timestamps = []
    st.session_state.test_started = False
    st.session_state.test_completed = False
    st.session_state.graph_filename = ""
    st.session_state.tap_effect_time = 0
    st.write("테스트가 재시작되었습니다. 손을 보여주고 '테스트 시작'을 클릭하세요.")

# 이미지 처리 및 결과 표시
if img_file_buffer is not None:
    # 이미지 열기
    image = Image.open(img_file_buffer)
    st.session_state.current_frame = image
    
    # 이미지 처리
    processed_image = process_image(image)
    
    if processed_image is not None:
        st.session_state.last_processed_image = processed_image
        
        # 프레임 표시
        st.image(processed_image, channels="RGB", caption="처리된 이미지")
        
        # 테스트 완료 시 결과 표시
        if st.session_state.test_completed and not hasattr(st.session_state, 'results_displayed'):
            display_results()
            st.session_state.results_displayed = True

# 자동 재캡처 안내
if st.session_state.test_started and not st.session_state.test_completed:
    elapsed_time = time.time() - st.session_state.start_time
    remaining_time = max(0, test_duration - elapsed_time)
    
    # 진행 표시줄 추가
    progress_bar = st.progress(int((elapsed_time / test_duration) * 100))
    
    st.info(f"테스트 진행 중: {remaining_time:.1f}초 남음. 계속해서 새 이미지를 캡처하여 손가락 태핑을 기록하세요.")
    
    if elapsed_time >= test_duration and not st.session_state.test_completed:
        st.session_state.test_completed = True
        display_results()
        st.session_state.results_displayed = True
        st.success("테스트가 완료되었습니다!")

# 손 감지 모듈 정리
hands.close()