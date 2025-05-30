## Latest updates -- SAM 2: Segment Anything in Images and Videos

Please check out our new release on [**Segment Anything Model 2 (SAM 2)**](https://github.com/facebookresearch/segment-anything-2).

* SAM 2 code: https://github.com/facebookresearch/segment-anything-2
* SAM 2 demo: https://sam2.metademolab.com/
* SAM 2 paper: https://arxiv.org/abs/2408.00714

 ![SAM 2 architecture](https://github.com/facebookresearch/segment-anything-2/blob/main/assets/model_diagram.png?raw=true)

**Segment Anything Model 2 (SAM 2)** is a foundation model towards solving promptable visual segmentation in images and videos. We extend SAM to video by considering images as a video with a single frame. The model design is a simple transformer architecture with streaming memory for real-time video processing. We build a model-in-the-loop data engine, which improves model and data via user interaction, to collect [**our SA-V dataset**](https://ai.meta.com/datasets/segment-anything-video), the largest video segmentation dataset to date. SAM 2 trained on our data provides strong performance across a wide range of tasks and visual domains.

# Segment Anything

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

![SAM design](assets/model_diagram.png?raw=true)

The **Segment Anything Model (SAM)** produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a [dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](#model-checkpoints). Then the model can be used in just a few lines to get masks from a given prompt:

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

Additionally, masks can be generated for images from the command line:

```
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```

See the examples notebooks on [using SAM with prompts](/notebooks/predictor_example.ipynb) and [automatically generating masks](/notebooks/automatic_mask_generator_example.ipynb) for more details.

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>

## ONNX Export

SAM's lightweight mask decoder can be exported to ONNX format so that it can be run in any environment that supports ONNX runtime, such as in-browser as showcased in the [demo](https://segment-anything.com/demo). Export the model with

```
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

See the [example notebook](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb) for details on how to combine image preprocessing via SAM's backbone with mask prediction using the ONNX model. It is recommended to use the latest stable version of PyTorch for ONNX export.

### Web demo

The `demo/` folder has a simple one page React app which shows how to run mask prediction with the exported ONNX model in a web browser with multithreading. Please see [`demo/README.md`](https://github.com/facebookresearch/segment-anything/blob/main/demo/README.md) for more details.

## <a name="Models"></a>Model Checkpoints

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Dataset

See [here](https://ai.facebook.com/datasets/segment-anything/) for an overview of the datastet. The dataset can be downloaded [here](https://ai.facebook.com/datasets/segment-anything-downloads/). By downloading the datasets you agree that you have read and accepted the terms of the SA-1B Dataset Research License.

We save masks per image as a json file. It can be loaded as a dictionary in python in the below format.

```python
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # Image id
    "width"                 : int,              # Image width
    "height"                : int,              # Image height
    "file_name"             : str,              # Image filename
}

annotation {
    "id"                    : int,              # Annotation id
    "segmentation"          : dict,             # Mask saved in COCO RLE format.
    "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    "area"                  : int,              # The area in pixels of the mask
    "predicted_iou"         : float,            # The model's own prediction of the mask's quality
    "stability_score"       : float,            # A measure of the mask's quality
    "crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
    "point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
}
```

Image ids can be found in sa_images_ids.txt which can be downloaded using the above [link](https://ai.facebook.com/datasets/segment-anything-downloads/) as well.

To decode a mask in COCO RLE format into binary:

```
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```

See [here](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py) for more instructions to manipulate masks stored in RLE format.

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The Segment Anything project was made possible with the help of many contributors (alphabetical):

Aaron Adcock, Vaibhav Aggarwal, Morteza Behrooz, Cheng-Yang Fu, Ashley Gabriel, Ahuva Goldstand, Allen Goodman, Sumanth Gurram, Jiabo Hu, Somya Jain, Devansh Kukreja, Robert Kuo, Joshua Lane, Yanghao Li, Lilian Luong, Jitendra Malik, Mallika Malhotra, William Ngan, Omkar Parkhi, Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala Varadarajan, Bram Wasti, Zachary Winstrom

## Citing Segment Anything

If you use SAM or SA-1B in your research, please use the following BibTeX entry.

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

# Si/SiGe 선택성 측정 도구 (SAM 기반)

이 도구는 Meta의 Segment Anything Model(SAM)을 활용하여 반도체 웨이퍼 이미지에서 Si와 SiGe 영역을 자동으로 분할하고 선택성을 측정하는 GUI 애플리케이션입니다.

## 주요 기능

- **자동 세그멘테이션**: SAM 모델을 사용한 정밀한 이미지 분할
- **레이어 관리**: 여러 세그멘테이션 레이어를 동시에 관리
- **두께 측정**: 픽셀 단위 및 실제 나노미터 단위 두께 측정
- **선택성 계산**: Si/SiGe 층간 선택성 자동 계산
- **브러시 도구**: 수동 편집을 위한 브러시 추가/제거 기능
- **다양한 분석**: 교차점 분석, 구조 위치 결정
- **결과 출력**: CSV, PDF 보고서 및 시각화 이미지 생성

## 시스템 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장)
- 최소 8GB RAM
- Windows 10/11 또는 Linux

## 설치 방법

### 1. 리포지토리 클론

```bash
git clone <repository-url>
cd segment-anything
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv sam_env
# Windows
sam_env\Scripts\activate
# Linux/Mac
source sam_env/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. SAM 모델 다운로드

SAM 체크포인트 파일을 다운로드하기 위해 아래 스크립트를 실행하세요:

```python
import urllib.request
import os

def download_sam_model():
    """SAM 모델 체크포인트를 다운로드합니다."""
    
    # 모델 다운로드 정보
    models = {
        "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    # checkpoints 디렉토리 생성
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    print("SAM 모델 다운로드 중...")
    
    for model_name, url in models.items():
        file_path = os.path.join(checkpoint_dir, model_name)
        
        if os.path.exists(file_path):
            print(f"✓ {model_name} 이미 존재함")
            continue
            
        print(f"다운로드 중: {model_name}")
        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"✓ {model_name} 다운로드 완료")
        except Exception as e:
            print(f"✗ {model_name} 다운로드 실패: {e}")
    
    print("모든 모델 다운로드 완료!")

if __name__ == "__main__":
    download_sam_model()
```

위 코드를 `download_sam.py`로 저장한 후 실행하세요:

```bash
python download_sam.py
```

### 대안: 수동 다운로드

다음 링크에서 모델을 직접 다운로드할 수도 있습니다:

- **SAM ViT-H (기본 권장)**: [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- **SAM ViT-L**: [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- **SAM ViT-B**: [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

다운로드한 파일을 프로젝트의 `checkpoints/` 폴더에 저장하세요.

## 사용 방법

### 1. 애플리케이션 실행

```bash
python Si_SiGe_Selectivity_측정_Tool.py
```

### 2. 기본 워크플로우

1. **이미지 로드**: 'Load Image' 버튼으로 SEM 이미지 불러오기
2. **스케일 설정**: 픽셀-나노미터 변환 비율 설정
3. **영역 분할**: 
   - 포인트 클릭으로 관심 영역 선택
   - 박스 도구로 영역 지정
   - 브러시로 세밀한 편집
4. **레이어 관리**: 여러 레이어로 다양한 영역 분할
5. **분석 실행**: 선택성 계산 및 두께 측정
6. **결과 저장**: CSV, PDF, 이미지 형태로 결과 출력

### 3. 주요 도구 사용법

#### 포인트 모드
- 좌클릭: 포함할 영역 지정
- 우클릭: 제외할 영역 지정
- Enter: 세그멘테이션 실행

#### 박스 모드
- 드래그로 관심 영역 박스 생성
- Enter: 박스 영역 세그멘테이션

#### 브러시 모드
- 좌클릭 드래그: 마스크 추가/제거
- 마우스 휠: 브러시 크기 조절
- Shift + 휠: 빠른 크기 조절

### 4. 단축키

- `Ctrl + Z`: 실행 취소
- `Ctrl + S`: 결과 저장
- `Ctrl + O`: 이미지 열기
- `Space`: 모든 레이어 표시/숨김
- `1-9`: 브러시 크기 조절
- `+/-`: 확대/축소

## 주요 매개변수

### SAM 모델 설정
- **모델 타입**: vit_h (기본), vit_l, vit_b
- **신뢰도 임계값**: 0.5 (기본)
- **다중 마스크**: True (기본)

### 분석 매개변수
- **윈도우 크기**: 100픽셀 (기본)
- **두께 측정 간격**: 사용자 정의
- **선택성 계산 방법**: 자동 감지

### 시각화 설정
- **마스크 투명도**: 50% (기본)
- **선 두께**: 2픽셀 (기본)
- **폰트 크기**: 60 (기본)

## 출력 파일

### CSV 데이터
- 레이어별 측정값
- 좌표 정보
- 두께 및 선택성 데이터

### PDF 보고서
- 분석 결과 요약
- 시각화 이미지
- 통계 그래프
- 상세 측정 데이터

### 이미지 파일
- 세그멘테이션 결과
- 오버레이 이미지
- 분석 시각화

## 문제 해결

### 일반적인 문제

1. **CUDA 메모리 부족**
   - 더 작은 SAM 모델 사용 (vit_b)
   - 이미지 크기 축소
   - 배치 크기 감소

2. **세그멘테이션 품질 낮음**
   - 신뢰도 임계값 조정
   - 더 정확한 포인트 선택
   - 브러시로 수동 편집

3. **성능 느림**
   - GPU 가속 확인
   - 이미지 해상도 조정
   - 불필요한 레이어 제거

### 로그 및 디버깅

애플리케이션 실행 시 콘솔에서 상세한 로그를 확인할 수 있습니다. 오류 발생 시 traceback 정보가 표시됩니다.

## 기술 사양

### 지원 이미지 형식
- PNG, JPG, JPEG, BMP, TIF, TIFF

### 처리 성능
- 이미지 크기: 최대 4K 해상도
- 메모리 사용량: 4-8GB (이미지 크기에 따라)
- 처리 시간: 1-10초 (GPU 성능에 따라)

## 기여 및 개발

### 코드 구조
```
Si_SiGe_Selectivity_측정_Tool.py
├── SAMApp (메인 애플리케이션 클래스)
├── UI 구성 요소
├── SAM 모델 인터페이스
├── 이미지 처리 함수
├── 분석 및 계산 함수
└── 결과 출력 함수
```

### 개발 환경 설정
```bash
pip install -r requirements.txt
pip install pytest  # 테스트용
pip install black   # 코드 포맷팅
```

## 라이선스

이 프로젝트는 [라이선스 정보]에 따라 배포됩니다.

## 참고 자료

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [PySide2 Documentation](https://doc.qt.io/qtforpython/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

## 문의 및 지원

기술적 문제나 기능 개선 제안은 [Issues](링크)를 통해 문의하시기 바랍니다.
