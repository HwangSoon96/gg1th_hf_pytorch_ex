<div align="center">

# gg1th_hf_pytorch_ex

**PyTorch 기초 · Transformer Self-Attention 원리 · Hugging Face Hub 멀티모달 모델 활용 실습**

파인튜닝 전에 Hub의 사전학습 모델로 이미지 · 음성 · 문서 · 영상 AI 서비스를 빠르게 체험

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?logo=uv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.6-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-5.x-FFD21E)
![Hugging Face Hub](https://img.shields.io/badge/Hugging%20Face-Hub-FFD21E?logo=huggingface&logoColor=black)

</div>

<br>

## 실습 구성

### PyTorch 기초 — `pytorch_basic/`

| 파일 | 내용 |
|---|---|
| `pytorch_basic_practice.ipynb` | Tensor 생성 · shape · dtype · device, 인덱싱, Broadcasting, NumPy 변환, Autograd, 간단한 분류기 학습 · 저장 |

### Transformer 원리 — `transformer_ex/`

| 파일 | 내용 |
|---|---|
| `transformer_core.ipynb` | Query · Key · Value로 Self-Attention 직접 계산, attention weights 시각화, context vector |

### Hugging Face 멀티모달 — `huggingface_ex/`

| 파일 | 태스크 | 내용 |
|---|---|---|
| `01_product_photo_caption_tags` | Image Captioning | 상품 사진 → 설명 · 태그 · 검색 키워드 |
| `02_zero_shot_image_classifier` | Zero-shot 분류 | 학습 없이 새 카테고리로 이미지 분류 |
| `03_visual_question_answering` | VQA | 이미지에 대해 질문하고 답변 |
| `04_document_question_answering` | Document QA | 문서 이미지에서 질의응답 |
| `05_whisper_speech_to_text` | STT | Whisper 음성 → 텍스트 |
| `06_korean_text_to_speech` | TTS | 한국어 텍스트 → 음성 |
| `07_clip_image_text_search` | CLIP | 텍스트로 가장 비슷한 이미지 검색 |
| `08_monocular_depth_estimation` | Depth Estimation | 사진 한 장에서 거리감 추정 |
| `09_object_detection_detr` | Object Detection | DETR로 사진 속 객체 탐지 |
| `10_image_segmentation` | Segmentation | 이미지 영역 분할 |
| `11_video_classification` | Video Classification | 짧은 영상 행동 분류 (VideoMAE) |

샘플 이미지 · 음성 · 영상은 `huggingface_ex/images · audio · videos/`에 포함

<br>

## 시작하기

### 사전 요구사항

- NVIDIA GPU (RTX 4060 8GB 수준) 또는 Colab GPU — CPU로도 일부 실습 가능하나 느림
- [uv](https://docs.astral.sh/uv/)

### 설치

```bash
git clone https://github.com/HwangSoon96/gg1th_hf_pytorch_ex.git
cd gg1th_hf_pytorch_ex

uv sync   # pyproject.toml에 PyTorch cu126 인덱스 지정되어 있음
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run jupyter lab
```

### 주요 패키지

| 용도 | 패키지 |
|---|---|
| 모델 | transformers 5.x, huggingface-hub, accelerate, safetensors, timm |
| 오디오 | librosa, soundfile, scipy |
| 영상 | av, torchcodec |
| 기타 | sentencepiece, protobuf, pillow, datasets |
