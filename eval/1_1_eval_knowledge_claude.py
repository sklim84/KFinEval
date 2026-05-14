"""
LLM 모델 평가 스크립트 (Anthropic Claude API 사용, 답변 생성 전용)

- Claude Messages API Structured Outputs 사용 (beta)
- 모델은 반드시 A~E 중 하나만 출력
- JSON Schema 기반, API 레벨 보장
- anthropic>=0.34.x 완전 호환
- beta header: structured-outputs-2025-11-13

정답 비교 및 통계 계산은 1_2_stats_knowledge.py에서 수행
"""

import os
import json
import time
import random
import re
import pandas as pd
import anthropic
from tqdm import tqdm
from typing import Optional, Tuple
from dotenv import load_dotenv

# =================================
# Anthropic API 키 설정 (.env 파일에서 로드)
# =================================
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# =================================
# Structured Output JSON Schema
# =================================
ANSWER_SCHEMA = {
    "type": "json_schema",
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
    """
    객관식 질문 프롬프트 생성.
    JSON 형식 요청은 output_format 파라미터로 처리되므로 프롬프트에 포함하지 않음.
    """
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
# Structured output 추출
# =================================
def extract_structured_answer(response) -> Optional[dict]:
    """
    Claude API Structured Output 응답에서 JSON 형식의 답변을 추출합니다.
    
    Structured Output 사용 시:
    - response.content[0].text에 유효한 JSON 문자열이 반환됨
    - response.parsed_output이 있는 경우 (parse() 메서드 사용 시) 직접 사용 가능
    """
    try:
        # parse() 메서드를 사용한 경우 parsed_output 속성 확인
        if hasattr(response, "parsed_output") and response.parsed_output is not None:
            parsed = response.parsed_output
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed
        
        # create() 메서드를 사용한 경우 content[0].text에서 JSON 파싱
        if hasattr(response, "content") and response.content:
            # 첫 번째 텍스트 블록에서 JSON 추출
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    if hasattr(block, "text") and block.text:
                        try:
                            # Structured output은 항상 유효한 JSON을 반환
                            parsed = json.loads(block.text.strip())
                            if isinstance(parsed, dict) and "answer" in parsed:
                                return parsed
                        except json.JSONDecodeError as e:
                            print(f"  JSON 파싱 실패: {e}")
                            print(f"  텍스트 내용: {block.text[:200]}...")
                            return None
        
        return None
    except Exception as e:
        print(f"extract_structured_answer 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

# =================================
# 답변 생성 함수 (retry 포함)
# =================================
def generate_answer_single(
    model_name: str,
    prompt: str,
    max_tokens: int = 100,
    max_retries: int = 5
) -> Tuple[Optional[str], Optional[str], Optional[str]]:

    for attempt in range(max_retries):
        try:
            # Structured Outputs 사용 (beta)
            response = client.beta.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                betas=["structured-outputs-2025-11-13"],
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                output_format=ANSWER_SCHEMA
            )

            parsed = extract_structured_answer(response)
            
            if parsed is None or "answer" not in parsed:
                print(f"  구조화된 답변 추출 실패, 재시도 ({attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None, None, None
            
            answer = parsed["answer"]
            
            # 답변이 유효한지 확인 (A-E 중 하나)
            if answer not in ["A", "B", "C", "D", "E"]:
                print(f"  유효하지 않은 답변: {answer}, 재시도 ({attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None, None, None
            
            print(f'answer: {answer}')

            # 응답 객체를 JSON으로 직렬화
            try:
                content_list = []
                for block in response.content:
                    block_dict = {}
                    if hasattr(block, "type"):
                        block_dict["type"] = block.type
                    if hasattr(block, "text"):
                        block_dict["text"] = block.text
                    content_list.append(block_dict)
                
                raw_response_dict = {
                    "id": getattr(response, "id", None),
                    "model": getattr(response, "model", None),
                    "role": getattr(response, "role", None),
                    "content": content_list,
                    "stop_reason": getattr(response, "stop_reason", None),
                    "stop_sequence": getattr(response, "stop_sequence", None),
                }
                
                if hasattr(response, "usage") and response.usage:
                    raw_response_dict["usage"] = {
                        "input_tokens": getattr(response.usage, "input_tokens", None),
                        "output_tokens": getattr(response.usage, "output_tokens", None)
                    }
                
                raw_response_json = json.dumps(raw_response_dict, ensure_ascii=False)
            except Exception as e:
                print(f"  raw_response 직렬화 실패: {e}")
                raw_response_json = json.dumps({"error": str(e)}, ensure_ascii=False)

            return (
                answer,
                json.dumps(parsed, ensure_ascii=False),
                raw_response_json
            )

        except anthropic.APIError as e:
            if e.status_code == 500 or e.status_code == 503:
                if attempt == max_retries - 1:
                    print(f"  {e.status_code} 오류 재시도 실패 (최종)")
                    return None, None, None
                sleep_time = 2 ** attempt + random.uniform(0, 0.5)
                print(f"  {e.status_code} 서버 오류, {sleep_time:.2f}s 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                print(f"  생성 오류 (API): {e}")
                import traceback
                traceback.print_exc()
                return None, None, None

        except Exception as e:
            print(f"  생성 오류: {e}")
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
    max_tokens: int = 100
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
            max_tokens=max_tokens
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
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-5",
    ]

    MAX_TOKENS = 32

    DATASET_DIR = "/workspace/Fin-Ben/_datasets/0_integration"
    BENCHMARKS = [
        "1_fin_knowledge.csv"
    ]

    print("=" * 60)
    print("Anthropic Claude Structured Outputs Evaluation Mode")
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
                max_tokens=MAX_TOKENS
            )

    print("\n모든 모델 평가 완료!")
