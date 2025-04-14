import os
import requests
import argparse
import sys

# 다운로드할 샘플 파일 목록 (run_csm.py에서 정의된 파일들)
VOICE_CLONE_FILES = [
    "joy_concert_gift.wav",
    "sad_trip_cancel.wav",
    "angry_lazy_teammate.wav",
    "nervous_interview_eve.wav",
    "touched_by_friend.wav",
    "embarrassed_presentation.wav",
    "dilemma_vacation_choice.wav",
    "emotional_parents_support.wav",
    "surprised_friend_visit.wav",
    "relieved_after_exam.wav",
    "proud_first_dinner.wav",
    "confused_firstday_work.wav",
    "tired_after_overtime.wav",
    "tearful_reunion_friend.wav",
    "disappointed_movie_night.wav",
    "excited_first_date.wav",
    "awkward_failed_joke.wav"
]

def download_file(url, destination):
    """파일 다운로드 함수"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"다운로드 오류 ({url}): {e}")
        return False

def download_samples(output_dir='.', base_url=None):
    """샘플 파일 다운로드"""
    # 기본 URL 설정 (GitHub 또는 Hugging Face에서 호스팅)
    if not base_url:
        base_url = "https://huggingface.co/sesame/csm-1b/resolve/main/samples/"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"샘플 파일 다운로드 중... ({len(VOICE_CLONE_FILES)}개 파일)")
    print(f"저장 위치: {os.path.abspath(output_dir)}")
    
    success_count = 0
    failed_files = []
    
    for i, filename in enumerate(VOICE_CLONE_FILES, 1):
        destination = os.path.join(output_dir, filename)
        
        # 이미 존재하는 파일 건너뛰기
        if os.path.exists(destination):
            print(f"[{i}/{len(VOICE_CLONE_FILES)}] {filename} - 이미 존재함, 건너뜀")
            success_count += 1
            continue
        
        # 다운로드 URL 생성
        url = f"{base_url}{filename}"
        
        print(f"[{i}/{len(VOICE_CLONE_FILES)}] {filename} 다운로드 중...")
        if download_file(url, destination):
            print(f"  ✓ 완료: {filename}")
            success_count += 1
        else:
            print(f"  ✗ 실패: {filename}")
            failed_files.append(filename)
    
    # 결과 요약
    print("\n다운로드 완료!")
    print(f"성공: {success_count}/{len(VOICE_CLONE_FILES)} 파일")
    
    if failed_files:
        print(f"실패: {len(failed_files)} 파일:")
        for filename in failed_files:
            print(f"  - {filename}")
    
    return success_count, failed_files

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="CSM-1B 샘플 음성 파일 다운로더")
    parser.add_argument("--output-dir", "-o", default=".", help="샘플 파일 저장 디렉토리 (기본: 현재 디렉토리)")
    parser.add_argument("--url", help="샘플 파일 기본 URL (기본: Hugging Face 저장소)")
    
    args = parser.parse_args()
    
    print("CSM-1B 샘플 음성 파일 다운로더")
    print("============================")
    
    # 다운로드 시작
    success_count, failed_files = download_samples(args.output_dir, args.url)
    
    # 성공 여부에 따른 종료 코드
    if failed_files:
        print("\n일부 파일 다운로드에 실패했습니다.")
        print("다시 실행하여 다운로드를 다시 시도하거나, 수동으로 파일을 다운로드하세요.")
        sys.exit(1)
    else:
        print("\n모든 파일이 성공적으로 다운로드되었습니다!")
        print("이제 CSM-1B 서버를 시작할 수 있습니다.")
        sys.exit(0)

if __name__ == "__main__":
    main() 