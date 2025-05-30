#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM (Segment Anything Model) 체크포인트 다운로더

이 스크립트는 Meta의 SAM 모델 체크포인트 파일들을 자동으로 다운로드합니다.
다운로드된 파일은 'checkpoints' 폴더에 저장됩니다.
"""

import urllib.request
import os
import sys
from pathlib import Path

def download_with_progress(url, filename):
    """진행률을 표시하면서 파일을 다운로드합니다."""
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            progress = f"\r{filename}: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)"
            print(progress, end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print()  # 새 줄
        return True
    except Exception as e:
        print(f"\n다운로드 실패: {e}")
        return False

def download_sam_models():
    """SAM 모델 체크포인트를 다운로드합니다."""
    
    # 모델 다운로드 정보 (크기 순서로 정렬)
    models = {
        "sam_vit_b_01ec64.pth": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "size_mb": 375,
            "description": "SAM ViT-Base (가장 빠름, 적은 메모리 사용)"
        },
        "sam_vit_l_0b3195.pth": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
            "size_mb": 1249,
            "description": "SAM ViT-Large (균형잡힌 성능)"
        },
        "sam_vit_h_4b8939.pth": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "size_mb": 2564,
            "description": "SAM ViT-Huge (최고 성능, 많은 메모리 필요)"
        }
    }
    
    # checkpoints 디렉토리 생성
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("SAM (Segment Anything Model) 체크포인트 다운로더")
    print("=" * 70)
    print()
    
    # 사용자에게 다운로드할 모델 선택 옵션 제공
    print("사용 가능한 SAM 모델:")
    print()
    
    for i, (model_name, info) in enumerate(models.items(), 1):
        print(f"{i}. {model_name}")
        print(f"   - 크기: ~{info['size_mb']}MB")
        print(f"   - 설명: {info['description']}")
        print()
    
    print("다운로드 옵션:")
    print("1: ViT-Base만 다운로드 (권장 - 빠른 시작)")
    print("2: ViT-Large만 다운로드 (균형잡힌 성능)")
    print("3: ViT-Huge만 다운로드 (최고 성능)")
    print("4: 모든 모델 다운로드 (약 4.2GB)")
    print("q: 종료")
    print()
    
    try:
        choice = input("선택하세요 (1-4, q): ").strip().lower()
    except KeyboardInterrupt:
        print("\n다운로드가 취소되었습니다.")
        return
    
    if choice == 'q':
        print("다운로드가 취소되었습니다.")
        return
    
    # 다운로드할 모델 결정
    models_to_download = []
    model_list = list(models.items())
    
    if choice == '1':
        models_to_download = [model_list[0]]  # ViT-Base
    elif choice == '2':
        models_to_download = [model_list[1]]  # ViT-Large
    elif choice == '3':
        models_to_download = [model_list[2]]  # ViT-Huge
    elif choice == '4':
        models_to_download = model_list  # 모든 모델
    else:
        print("잘못된 선택입니다. ViT-Base 모델을 다운로드합니다.")
        models_to_download = [model_list[0]]
    
    print()
    print("다운로드 시작...")
    print("-" * 50)
    
    success_count = 0
    total_models = len(models_to_download)
    
    for model_name, info in models_to_download:
        file_path = checkpoint_dir / model_name
        
        if file_path.exists():
            print(f"✓ {model_name} 이미 존재함 (건너뛰기)")
            success_count += 1
            continue
        
        print(f"\n다운로드 중: {model_name}")
        print(f"예상 크기: {info['size_mb']}MB")
        
        if download_with_progress(info["url"], str(file_path)):
            print(f"✓ {model_name} 다운로드 완료")
            success_count += 1
        else:
            print(f"✗ {model_name} 다운로드 실패")
            # 실패한 파일이 있다면 삭제
            if file_path.exists():
                file_path.unlink()
    
    print()
    print("-" * 50)
    print(f"다운로드 완료: {success_count}/{total_models} 모델")
    
    if success_count > 0:
        print()
        print("다운로드된 모델 파일:")
        for file in checkpoint_dir.glob("*.pth"):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name} ({size_mb:.1f}MB)")
        
        print()
        print("사용 방법:")
        print("1. Si_SiGe_Selectivity_측정_Tool.py를 실행하세요.")
        print("2. 첫 실행 시 SAM 모델이 자동으로 로드됩니다.")
        print("3. 'Load Image' 버튼으로 분석할 이미지를 불러오세요.")
        
    print()
    print("완료!")

def verify_downloads():
    """다운로드된 파일의 유효성을 확인합니다."""
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        print("checkpoints 폴더가 없습니다.")
        return False
    
    pth_files = list(checkpoint_dir.glob("*.pth"))
    
    if not pth_files:
        print("다운로드된 SAM 모델이 없습니다.")
        return False
    
    print("발견된 SAM 모델:")
    for file in pth_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  ✓ {file.name} ({size_mb:.1f}MB)")
    
    return True

if __name__ == "__main__":
    try:
        # 기존 파일 확인
        if verify_downloads():
            print()
            response = input("기존 모델이 발견되었습니다. 추가 다운로드를 진행하시겠습니까? (y/n): ").strip().lower()
            if response not in ['y', 'yes']:
                print("다운로드를 취소합니다.")
                sys.exit(0)
        
        download_sam_models()
        
    except KeyboardInterrupt:
        print("\n\n다운로드가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        print("인터넷 연결을 확인하고 다시 시도해주세요.") 