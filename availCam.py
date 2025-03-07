import cv2
import numpy as np

def list_available_cameras(max_cameras_to_check=10):
    """
    사용 가능한 카메라 장치를 확인합니다.
    
    Args:
        max_cameras_to_check (int): 확인할 최대 카메라 수
        
    Returns:
        list: 사용 가능한 카메라 인덱스 목록
    """
    available_cameras = []
    
    for camera_idx in range(max_cameras_to_check):
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"카메라 인덱스 {camera_idx} 사용 가능 (해상도: {frame.shape[1]}x{frame.shape[0]})")
                available_cameras.append(camera_idx)
            else:
                print(f"카메라 인덱스 {camera_idx} 감지되었지만 프레임을 읽을 수 없음")
            cap.release()
        else:
            print(f"카메라 인덱스 {camera_idx} 사용 불가")
    
    return available_cameras

def test_camera(camera_idx, display_time=3):
    """
    선택한 카메라로 짧은 테스트를 실행합니다.
    
    Args:
        camera_idx (int): 테스트할 카메라 인덱스
        display_time (int): 테스트 지속 시간(초)
    """
    import time
    
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"카메라 {camera_idx}를 열 수 없습니다.")
        return
    
    print(f"카메라 {camera_idx} 테스트 중... {display_time}초 동안 표시됩니다.")
    
    # 카메라 정보 가져오기
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"카메라 {camera_idx} 속성: {width}x{height} @ {fps}fps")
    
    start_time = time.time()
    while (time.time() - start_time) < display_time:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
            
        # 카메라 인덱스와 해상도 정보 표시
        cv2.putText(frame, f"Camera {camera_idx}: {int(width)}x{int(height)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(f'Camera {camera_idx} Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("사용 가능한 카메라 목록 확인 중...")
    available_cameras = list_available_cameras()
    
    if available_cameras:
        print(f"\n사용 가능한 카메라 수: {len(available_cameras)}")
        print(f"사용 가능한 카메라 인덱스: {available_cameras}")
        
        # 각 카메라 테스트
        for camera_idx in available_cameras:
            test_camera(camera_idx)
    else:
        print("\n사용 가능한 카메라가 없습니다.")