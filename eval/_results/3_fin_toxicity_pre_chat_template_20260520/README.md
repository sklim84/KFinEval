# Pre-Chat-Template Archive (2026-05-20)

이 디렉토리는 `3_1_gen_toxicity_openlm.py`에 **chat template 적용 패치 전** 생성된 vLLM 백엔드 답변 결과를 보존합니다.

## 패치 내용 (2026-05-20)
- `model.generate(raw_text)` → `model.chat(messages)`로 변경
- manifest `think_mode=yes` 컬럼을 읽어 `chat_template_kwargs={"enable_thinking": True}` 전달
- raw_response에 `chat_template_applied: true` 마커 추가

## 패치 이유
- reasoning 모델이 chat template 없이 raw text를 받아 think 모드가 안 켜지거나 부분 작동
- 검증된 문제 케이스:
  - K_EXAONE_236B_A23B: 62% truncated (내장 reasoning, max_tok=2048 부족)
  - Qwen3-4B-Thinking-2507: 44% truncated (max_tok=8192 부족)
  - Phi-4-reasoning, Phi-4-mini-reasoning: think tag 비대칭

## 같이 적용된 manifest 정책
- 모든 think_mode=yes 모델 → max_output_tokens ≥ 16384
- Qwen3-4B-Thinking-2507: 8192 → 32768 (44% trunc 대응)
- K_EXAONE_236B_A23B: think_mode no → yes, max_tok 2048 → 16384

## 보존 대상
- 16개 vLLM 답변 CSV (15 정상 + Solar-Open-100B 1-row dry-run)
- DeepSeek-R1 staged-rerun 백업 2개 (.16k.bak, .32k.bak)

## 동결
이 디렉토리는 diff/audit 용도로만 보존합니다. 새로운 결과는 `../3_fin_toxicity/`에 저장됩니다.
