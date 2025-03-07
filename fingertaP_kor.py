import cv2
import mediapipe as mp
import time
import numpy as np

# MediaPipe hands 모듈 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 웹캠 설정
cap = cv2.VideoCapture(1)

# 카메라가 제대로 열렸는지 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다. 카메라 권한을 확인해주세요.")
    exit()

# 탭 감지 변수 초기화
finger_tap_count = 0
is_tapping = False
prev_tapping_state = False
tap_timestamps = []
test_duration = 15  # 테스트 총 시간 (초)
segment_duration = 5  # 각 세그먼트 시간 (초)
start_time = None
test_started = False
test_completed = False
results_displayed = False

# 손가락 랜드마크 인덱스
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP

print("카메라가 시작되었습니다. 손을 보여주세요.")
print("15초 동안 엄지와 검지로 탭 속도를 측정합니다.")
print("테스트를 시작하려면 손을 카메라에 보여주세요.")

while True:
    success, image = cap.read()
    if not success:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # 이미지 좌우 반전 (거울 효과)
    image = cv2.flip(image, 1)
    
    # 성능 향상을 위해 이미지를 읽기 전용으로 설정
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 손 인식 처리
    results = hands.process(image_rgb)
    
    # 다시 이미지를 쓰기 가능하도록 설정
    image.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # 현재 시간
    current_time = time.time()
    
    # 손이 감지되면
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # 테스트가 아직 시작되지 않았으면 시작
            if not test_started and not test_completed:
                start_time = current_time
                test_started = True
                print("테스트 시작!")
            
            # 엄지와 검지 손가락 끝 좌표 얻기
            thumb_tip = hand_landmarks.landmark[THUMB_TIP]
            index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
            
            # 화면 크기 얻기
            h, w, c = image.shape
            
            # 손가락 끝 좌표를 픽셀로 변환
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            
            # 엄지와 검지 끝 시각화
            cv2.circle(image, (thumb_x, thumb_y), 10, (255, 0, 0), -1)  # 엄지: 파란색
            cv2.circle(image, (index_x, index_y), 10, (0, 0, 255), -1)  # 검지: 빨간색
            
            # 두 손가락 사이 거리 계산
            distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            
            # 탭 감지 (거리가 충분히 가까우면)
            is_tapping = distance < 100  # 이 값은 조정 가능
            
            # 탭 상태가 변경되었고, 현재 탭 중이면 카운트 증가
            if is_tapping and not prev_tapping_state and test_started and not test_completed:
                finger_tap_count += 1
                tap_timestamps.append(current_time)
                
            # 이전 탭 상태 업데이트
            prev_tapping_state = is_tapping
            
            # 손가락 사이 거리 표시
            distance_text = f"거리: {distance:.1f}"
            cv2.putText(image, distance_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 테스트 진행 중이면 남은 시간 계산
    if test_started and not test_completed:
        elapsed_time = current_time - start_time
        remaining_time = max(0, test_duration - elapsed_time)
        
        # 현재 섹션 표시 (처음/중간/마지막 5초)
        if elapsed_time < segment_duration:
            section = "처음 5초"
        elif elapsed_time < 2 * segment_duration:
            section = "중간 5초"
        else:
            section = "마지막 5초"
        
        # 테스트 완료 확인
        if elapsed_time >= test_duration:
            test_completed = True
            print("\n테스트 완료!")
            
            # 전체 결과 계산
            overall_taps_per_second = finger_tap_count / test_duration
            
            # 5초 구간별 결과 계산
            segment_counts = [0, 0, 0]  # 각 5초 구간의 탭 수
            
            for timestamp in tap_timestamps:
                time_offset = timestamp - start_time
                if time_offset < segment_duration:
                    segment_counts[0] += 1
                elif time_offset < 2 * segment_duration:
                    segment_counts[1] += 1
                else:
                    segment_counts[2] += 1
            
            segment_rates = [count / segment_duration for count in segment_counts]
            
            # 처음 5초와 마지막 5초 변화율 계산
            if segment_rates[0] > 0:  # 0으로 나누기 방지
                change_percentage = ((segment_rates[2] - segment_rates[0]) / segment_rates[0]) * 100
            else:
                change_percentage = 0
            
            # 결과 표시
            print(f"15초 동안 총 탭 횟수: {finger_tap_count}")
            print(f"전체 평균 탭 속도: {overall_taps_per_second:.2f} 탭/초")
            print(f"처음 5초 탭 속도: {segment_rates[0]:.2f} 탭/초 ({segment_counts[0]}회)")
            print(f"중간 5초 탭 속도: {segment_rates[1]:.2f} 탭/초 ({segment_counts[1]}회)")
            print(f"마지막 5초 탭 속도: {segment_rates[2]:.2f} 탭/초 ({segment_counts[2]}회)")
            print(f"처음 5초 대비 마지막 5초 속도 변화: {change_percentage:.1f}%")
            
            results_displayed = True
        
        # 화면에 진행 상황 표시
        progress_text = f"남은 시간: {remaining_time:.1f}초 - {section}"
        cv2.putText(image, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 현재 탭 수 표시
        tap_text = f"현재 탭 횟수: {finger_tap_count}"
        cv2.putText(image, tap_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 테스트 완료 후 결과 표시
    if test_completed:
        if not results_displayed:
            # 전체 결과 계산
            overall_taps_per_second = finger_tap_count / test_duration
            
            # 5초 구간별 결과 계산
            segment_counts = [0, 0, 0]  # 각 5초 구간의 탭 수
            
            for timestamp in tap_timestamps:
                time_offset = timestamp - start_time
                if time_offset < segment_duration:
                    segment_counts[0] += 1
                elif time_offset < 2 * segment_duration:
                    segment_counts[1] += 1
                else:
                    segment_counts[2] += 1
            
            segment_rates = [count / segment_duration for count in segment_counts]
            
            # 처음 5초와 마지막 5초 변화율 계산
            if segment_rates[0] > 0:  # 0으로 나누기 방지
                change_percentage = ((segment_rates[2] - segment_rates[0]) / segment_rates[0]) * 100
            else:
                change_percentage = 0
            
            # 결과 표시
            print(f"\n테스트 완료!")
            print(f"15초 동안 총 탭 횟수: {finger_tap_count}")
            print(f"전체 평균 탭 속도: {overall_taps_per_second:.2f} 탭/초")
            print(f"처음 5초 탭 속도: {segment_rates[0]:.2f} 탭/초 ({segment_counts[0]}회)")
            print(f"중간 5초 탭 속도: {segment_rates[1]:.2f} 탭/초 ({segment_counts[1]}회)")
            print(f"마지막 5초 탭 속도: {segment_rates[2]:.2f} 탭/초 ({segment_counts[2]}회)")
            print(f"처음 5초 대비 마지막 5초 속도 변화: {change_percentage:.1f}%")
            
            results_displayed = True
        
        # 화면에 결과 표시
        cv2.putText(image, "테스트 완료!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 5초 구간별 결과 계산 (화면 표시용)
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
        
        # 변화율 계산
        if segment_rates[0] > 0:
            change_percentage = ((segment_rates[2] - segment_rates[0]) / segment_rates[0]) * 100
            change_text = f"{change_percentage:.1f}%"
            # 증가/감소 화살표 추가
            if change_percentage > 0:
                change_text += " ↑"
            elif change_percentage < 0:
                change_text += " ↓"
        else:
            change_text = "N/A"
        
        # 총 탭 횟수
        cv2.putText(image, f"총 탭 횟수: {finger_tap_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 구간별 탭 속도
        cv2.putText(image, f"처음 5초: {segment_rates[0]:.2f} 탭/초", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"중간 5초: {segment_rates[1]:.2f} 탭/초", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"마지막 5초: {segment_rates[2]:.2f} 탭/초", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 변화율
        cv2.putText(image, f"속도 변화율: {change_text}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 다시 테스트 안내
        cv2.putText(image, "다시 테스트하려면 'r'을 누르세요", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 아직 테스트가 시작되지 않았으면 안내 메시지 표시
    if not test_started and not test_completed:
        cv2.putText(image, "손을 카메라에 보여주세요", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, "15초 동안 엄지와 검지로 탭하는 속도를 측정합니다", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 결과 화면 표시
    cv2.imshow('Finger Tap Speed Test - 15 seconds', image)
    
    # 키 입력 처리
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):  # 'q'를 누르면 종료
        break
    elif key == ord('r') and test_completed:  # 'r'를 누르면 테스트 재시작
        finger_tap_count = 0
        tap_timestamps = []
        test_started = False
        test_completed = False
        results_displayed = False
        print("\n테스트를 다시 시작합니다.")
        print("손을 카메라에 보여주세요.")

# 자원 해제
hands.close()
cap.release()
cv2.destroyAllWindows()