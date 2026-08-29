# Audio2Face · Web (INT8 WASM, iOS realtime)

UniTalker `unitalker_v0.4.0_base.onnx` (Audio2Face) 를 **브라우저에서 직접** 구동하는
로컬 퍼스트 데모. 서버 GPU 불필요, 아이폰 Safari에서 실시간 립싱크 검증 완료.

## 검증 결과 (iPhone 16 Pro, iOS 26.6, Safari 26)
- **EP: WASM·INT8** (103MB, ORT Web 1.29 멀티스레드 SIMD, 4 threads)
- 모델 로드: **1.8s** (IndexedDB 캐시 → 재다운로드 없음)
- 청크 추론: **평균 40~105ms / p95 ~130ms** (WIN 0.05s, 아이폰 16 Pro)
- 홉 **100ms** (HOP=0.10) → 추론 속도에 맞춰 실시간 OK ✓
- producer(pre-compute blendshapes) + consumer(audio-clock driven animation) 분리로 오디오 종료 시 입도 정지
- 크래시 없이 100% 완료, 연속 재생 OOM 해결 (세션 1회 재사용)

> WebGPU EP는 iOS Safari에서 INT8 초기화 실패/지연 → WASM 우선 폴백.

## 구조
```
public/
  index.html      # 1번: TTS 스트리밍 플레이어 (립싱크 + 레이턴시 HUD)
  avatar.html      # 2번: Babylon.js 아바타 rig (51 blendshape → morph target)
  test_voice.wav   # 기본 TTS 샘플
  tts_hello.wav    # 영어 샘플
  tts_korean.wav   # 한국어 샘플
  models/
    unitalker_v0.4.0_base.int8.onnx   # 103MB (브라우저용)
    unitalker_v0.4.0_base.fp32.onnx   # 389MB (fallback)
  serve.py         # HTTPS + COOP/COEP + IndexedDB-friendly 정적 서버
```

## 실행 (로컬)
```bash
cd public
python3 serve.py 8898
```
아이폰에서 접속 (Tailscale HTTPS 도메인 권장 — secure context 필요):
```
https://<magicdns>.ts.net:8898/index.html
```

### 왜 HTTPS인가
WASM 멀티스레드(SIMD + SharedArrayBuffer)는 **cross-origin isolation**(COOP/COEP)이
필요하고, 이는 **secure context(HTTPS)** 에서만 효력이 있다. 평문 HTTP IP는
`crossOriginIsolated=false` 가 되어 단일스레드로 느려진다.

## 모델 준비
`public/models/` 에 두 파일이 필요 (용량 커서 Git에 미포함):
- `unitalker_v0.4.0_base.fp32.onnx` (389MB) — 다운로드:
  ```bash
  bash download_models.sh
  ```
- `unitalker_v0.4.0_base.int8.onnx` (103MB) — FP32를 INT8 동적 양자화:
  ```python
  from onnxruntime.quantization import quantize_dynamic, QuantType
  quantize_dynamic(
      "public/models/unitalker_v0.4.0_base.fp32.onnx",
      "public/models/unitalker_v0.4.0_base.int8.onnx",
      op_types_to_quantize=["MatMul","Gemm","Conv"],
      quant_type=QuantType.QInt8,
  )
  ```
  > INT8은 ORT Web **WASM** EP에서만 동작 (WebGPU는 iOS에서 ConvInteger 미지원).

## blendshape 매핑
출력은 51차원 blendshape 벡터. 현재 `avatar.html` 은 **채널 32(입)** 만
프로시저럴 헤드의 `mouthOpen` morph에 매핑.
실제 DLP3D 아바타 GLB 적용 시: `loadGLB()` 로 GLB 로드 후 51채널 각각을
해당 morph target influence 에 연결 (TODO).

## 출처
- 모델: `unitalker_v0.4.0_base.onnx` (DLP3D Audio2Face, UniTalker ECCV 2024 기반)
- 프로젝트: https://github.com/acidsound/dlp3d.ai
