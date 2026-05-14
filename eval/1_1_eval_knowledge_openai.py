"""
LLM 모델 평가 스크립트 (OpenAI API 사용, 답변 생성 전용)

- OpenAI Structured Outputs (JSON Schema 기반, API 레벨 보장)
- GPT-5 / GPT-5-mini / GPT-5.2 대응 (Responses API 전용)
- 모델은 반드시 A~E 중 하나만 출력
- system 프롬프트 미사용
- 문자열 파싱 없음 (SDK output_parsed 사용)
- openai>=2.14.x 완전 호환

정답 비교 및 통계 계산은 1_2_stats_knowledge.py에서 수행
"""

import os
import json
import time
import random
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from typing import Optional, Tuple
from dotenv import load_dotenv
from openai import InternalServerError

# =================================
# OpenAI API 키 설정 (.env 파일에서 로드)
# =================================
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =================================
# Structured Output JSON Schema
# =================================
ANSWER_SCHEMA = {
    "type": "json_schema",
    "name": "mcq_answer",
    "schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": ["A", "B", "C", "D", "E"]
            }
        },
        "required": ["answer"],
        "additionalProperties": False
    }
}

# =================================
# 프롬프트 생성 함수
# =================================
def create_prompt(question: str, A: str, B: str, C: str, D: str, E: str) -> str:
    return f"""
다음 객관식 질문에 답하세요.

질문:
{question}

선택지:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
""".strip()

# =================================
# 모델별 reasoning 설정
# - gpt-5.2 모델은 none 이외 설정시 해당 토큰에서 nonetype 반환 발생
# - gpt-5.2 이외 모델은 minimal 이상 설정 시 nonetype 반환 발생
# - default effot : medium
# =================================
def get_reasoning_config(model_name: str):
    if model_name.startswith("gpt-5.2"):
        return {"effort": "none"} 
    else:
        return {"effort": "minimal"}

# =================================
# Structured output 추출 (모델 공통)
# =================================
def extract_structured_answer(response):
    # SDK convenience
    if hasattr(response, "output_parsed") and response.output_parsed is not None:
        return response.output_parsed

    # Responses API 객체 기반 접근 (gpt-5 / gpt-5.2 공통)
    try:
        for message in response.output:
            # reasoning item 등 content=None 인 경우 스킵
            if not hasattr(message, "content") or message.content is None:
                continue

            for content in message.content:
                if getattr(content, "type", None) == "output_text":
                    return json.loads(content.text)
    except Exception as e:
        print(f"extract_structured_answer 실패: {e}")

    return None

# =================================
# 답변 생성 함수 (retry 포함)
# =================================
def generate_answer_single(
    model_name: str,
    prompt: str,
    max_output_tokens: int = 16,
    max_retries: int = 5
) -> Tuple[Optional[str], Optional[str], Optional[str]]:

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model_name,
                input=[{"role": "user", "content": prompt}],
                reasoning=get_reasoning_config(model_name),
                text={
                    "format": ANSWER_SCHEMA,
                    "verbosity": "low"
                },
                max_output_tokens=max_output_tokens,
            )

            parsed = extract_structured_answer(response)
            answer = parsed["answer"]
            print(f'answer: {answer}')

            return (
                answer,
                json.dumps(parsed, ensure_ascii=False),
                json.dumps(response.model_dump(), ensure_ascii=False)
            )

        except InternalServerError as e:
            if attempt == max_retries - 1:
                print(f"500 오류 재시도 실패 (최종)")
                raise
            sleep_time = 2 ** attempt + random.uniform(0, 0.5)
            print(f"  500 서버 오류, {sleep_time:.2f}s 후 재시도 ({attempt+1}/{max_retries})")
            time.sleep(sleep_time)

        except Exception as e:
            print(f"  생성 오류 (structured): {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

# =================================
# CSV 처리 함수
# =================================
def process_csv(
    model_name: str,
    input_csv_path: str,
    output_csv_path: str,
    max_output_tokens: int = 16
) -> None:

    data = pd.read_csv(input_csv_path)

    results = []
    total_count = len(data)
    success_count = 0
    error_ids = []

    print(f"\n평가 모델: {model_name}")
    print(f"총 {total_count}개 질문 처리 시작...\n")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    for index, row in tqdm(
        data.iterrows(),
        total=total_count,
        desc=f"처리 중 ({model_name})"
    ):
        prompt = create_prompt(
            row["question"],
            row["A"], row["B"], row["C"], row["D"], row["E"]
        )

        answer, answer_structured, raw_response = generate_answer_single(
            model_name=model_name,
            prompt=prompt,
            max_output_tokens=max_output_tokens
        )

        results.append({
            **row.to_dict(),
            "answer": answer,
            "answer_structured": answer_structured,
            "raw_response": raw_response
        })

        if answer is not None:
            success_count += 1
        else:
            error_ids.append(index)

    print(f"\n처리 완료: {success_count}/{total_count} 성공")
    if error_ids:
        print(f"실패한 행 인덱스: {error_ids}")

    pd.DataFrame(results).to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\n결과 저장 완료: {output_csv_path}")


# =================================
# 메인 실행 로직
# =================================
if __name__ == "__main__":

    TARGET_MODELS = [
         "gpt-5.2",
        # "gpt-5",
        # "gpt-5-mini",
        # "gpt-5-nano",
    ]

    MAX_OUTPUT_TOKENS = 32

    DATASET_DIR = "/workspace/Fin-Ben/_datasets/0_integration"
    BENCHMARKS = [
        "1_fin_knowledge.csv"
    ]

    print("=" * 60)
    print("OpenAI Responses API Structured Evaluation Mode")
    print("=" * 60)

    for model_name in TARGET_MODELS:
        print(f"\n{'=' * 60}")
        print(f"모델: {model_name}")
        print(f"{'=' * 60}")

        for benchmark in BENCHMARKS:
            input_path = os.path.join(DATASET_DIR, benchmark)

            model_safe = model_name.replace("/", "_")
            results_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "_results/1_fin_knowledge"
            )
            os.makedirs(results_dir, exist_ok=True)

            output_path = os.path.join(
                results_dir,
                f"{benchmark.replace('.csv', '')}_{model_safe}_response.csv"
            )

            process_csv(
                model_name=model_name,
                input_csv_path=input_path,
                output_csv_path=output_path,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )

    print("\n모든 모델 평가 완료!")
