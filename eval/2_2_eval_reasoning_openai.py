"""
추론 응답 평가 스크립트 (OpenAI API 사용)

이 스크립트는 2_1_gen_reasoning_openlm.py에서 생성된 추론 응답을 평가합니다.
OpenAI API를 사용하여 각 모델의 추론 응답을 전문가 검수 정답과 비교하여 평가합니다.
"""

import pandas as pd
import json
import os
from typing import Optional
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# =================================
# OpenAI API 키 설정 (.env 파일에서 로드)
# =================================
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =================================
# Structured Output JSON Schema
# =================================
EVAL_SCHEMA = {
    "type": "json_schema",
    "name": "reasoning_eval",
    "schema": {
        "type": "object",
        "properties": {
            "coherence": {
                "type": "integer", "minimum": 1, "maximum": 10
            },
            "consistency": {
                "type": "integer", "minimum": 1, "maximum": 10
            },
            "accuracy": {
                "type": "integer", "minimum": 1, "maximum": 10
            },
            "completeness": {
                "type": "integer", "minimum": 1, "maximum": 10
            },
            "reasoning": {
                "type": "integer", "minimum": 1, "maximum": 10
            },
            "overall_quality": {
                "type": "integer", "minimum": 1, "maximum": 10
            },
            "explanation": {
                "type": "string",
                "maxLength": 500
            }
        },
        "required": [
            "coherence",
            "consistency",
            "accuracy",
            "completeness",
            "reasoning",
            "overall_quality",
            "explanation"
        ],
        "additionalProperties": False
    }
}

# =================================
# Structured output 추출
# =================================
def extract_structured_eval(response):
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
# 평가 함수
# =================================
def generate_evaluation(
    model: str,
    context: str,
    question: str,
    gold: str,
    answer: str
) -> Optional[dict]:
    """
    질문에 대한 모델의 답변을 평가하는 함수
    
    Args:
        model: OpenAI API 모델명
        context: 참조 문서
        question: 질문
        gold: 정답 (전문가 검수)
        answer: 평가할 답변 (모델이 생성한 추론 응답)
    
    Returns:
        평가 결과 (JSON 문자열) 또는 None (오류 시)
    """
    user_prompt = f"""
      LLM 응답 품질 평가 전문가입니다. 답변을 전문가 검수 정답과 비교하여 평가합니다.

      **평가 대상:**
      - 질문: {question}
      - 평가할 답변: {answer}
      - 정답 (전문가 검수): {gold}

      **평가 방식:**
      1. 정답이 사용한 문서 정보를 파악
      2. 답변이 동일한 정보와 논리를 사용했는지 비교
      3. 결론이 정답과 다르면 감점

      **중요: 아래 참조 문서에서 정답이 어떤 정보를 사용했는지 반드시 확인하세요.**

      **참조 문서:**
      {context}

      ---

      위 문서를 기반으로 다음 평가 기준에 따라 점수를 매기세요.

      **평가 기준 (각 1-10점):**

      1. **정합성**: 정답이 사용한 관련 정보를 답변도 사용했는가?
        - 1-3: 정답과 다른 무관한 정보 다수 사용, 핵심 문맥 왜곡
        - 4-6: 일부 정답과 같은 정보 사용하나 무관한 내용도 포함
        - 7-8: 대부분 정답과 동일한 정보 사용, 소수 부차적 정보 혼입
        - 9-10: 정답과 동일한 정보만 선별적 활용, 불필요한 문맥 배제

      2. **일관성**: 정답처럼 주제 초점이 흔들리지 않는가?
        - 1-3: 주제가 빈번히 전환됨, 정답과 다른 방향
        - 4-6: 전반적 일관성 있으나 비관련 내용 간헐적 혼입
        - 7-8: 정답과 유사하게 톤과 논점 유지, 약간의 주변 정보
        - 9-10: 정답처럼 처음부터 끝까지 주제 집중, 논리적 일관성

      3. **정확성**: 정답의 사실과 일치하는가?
        - 1-3: 정답과 다른 사실 제시, 무관한 정보를 사실로 제시
        - 4-6: 주요 사실은 정답과 맞으나 일부 세부 오류
        - 7-8: 정답과 전반적으로 일치, 약간의 모호함
        - 9-10: 정답의 사실과 완전히 일치

      4. **완전성**: 정답이 다룬 핵심 쟁점을 빠짐없이 다루었는가?
        - 1-3: 정답의 핵심 요소 대부분 누락
        - 4-6: 정답의 주요 쟁점은 언급했으나 하위 맥락 부족
        - 7-8: 정답의 대부분 측면을 다루나 부차적 요소 약함
        - 9-10: 정답처럼 모든 측면을 빠짐없이 포괄적으로 다룸

      5. **추론성**: 정답과 유사한 논리적 추론 과정을 보이는가?
        - 1-3: 정답과 다른 추론, 과정 없거나 비논리적
        - 4-6: 일부 정답과 유사하나 논리 비약 발생
        - 7-8: 정답과 대부분 유사한 자연스러운 추론
        - 9-10: 정답과 동일한 단계별 명확한 논리 흐름

      6. **전체 품질**: 정답과의 전반적 일치도
        - 1-3: 정답과 큰 차이, 전반적 품질 낮음
        - 4-6: 일정 수준 일치하나 논리 흐름 불완전
        - 7-8: 정답과 유사하게 잘 구성, 소수 불일치
        - 9-10: 정답과 높은 일치도, 완성도 높음

      **출력 형식 (JSON):**
      {{
        "coherence": 점수,
        "consistency": 점수,
        "accuracy": 점수,
        "completeness": 점수,
        "reasoning": 점수,
        "overall_quality": 점수,
        "explanation": "정답과의 주요 차이점과 일치 정도 요약 (2-3문장)"
      }}
      """

    try:
        response = client.responses.create(
            model=model,
            input=[
                    {"role": "system", "content": "You are a financial domain expert. Your task is to evaluate the clarity and difficulty "
                                "of the given LLM response based on your expertise in finance."},
                    {"role": "user", "content": user_prompt}
                ],
            reasoning={"effort": "none"},
            text={
                "format": EVAL_SCHEMA,
                "verbosity": "low"
            },
            max_output_tokens=256
        )

        # 구조화된 결과 추출
        eval_result = extract_structured_eval(response)

        return eval_result
    except Exception as e:
        print(f"  평가 생성 오류: {e}")
        return None


# =================================
# CSV 파일 처리 함수
# =================================
def process_csv(input_csv_path: str, output_csv_path: str, eval_model: str):
    """
    CSV 파일을 읽어서 모델의 추론 응답을 평가
    
    Args:
        input_csv_path: 입력 CSV 파일 경로 (2_1_gen_reasoning_openlm.py의 출력)
        output_csv_path: 출력 CSV 파일 경로 (평가 결과)
        eval_model: 평가에 사용할 OpenAI API 모델명
    """
    # 입력 CSV 파일 읽기
    try:
        data = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return
    
    print(f"CSV 파일 로드 완료: {data.shape[0]}개 행, {data.shape[1]}개 컬럼")
    print(f"컬럼: {list(data.columns)}")
    
    # 결과 저장용 리스트
    results = []
    error_ids = []
    total_count = len(data)
    
    print(f"\n평가 시작: 총 {total_count}개 질문 처리...\n")
    
    # tqdm 진행 표시줄 설정
    pbar = tqdm(data.iterrows(), total=total_count, desc="평가 중")
    for index, row in pbar:
        try:
            _id = row['id']
            question = row['question']
            context = row['context']  # 참조 문서
            gold = row['gold']  # 2_1_gen_reasoning_openlm.py에서는 'gold' 컬럼 사용
            answer = row['answer']  # 평가할 모델의 추론 응답
            
            # 모델 답변이 None이거나 비어있으면 스킵
            if pd.isna(answer) or not str(answer).strip():
                tqdm.write(f"  [{index + 1}/{total_count}] ID: {_id} - 모델 답변이 비어있음, 스킵")
                error_ids.append(index)
                continue
            
            # 평가 생성
            eval_result = generate_evaluation(
                model=eval_model,
                context=context,
                question=question,
                gold=gold,
                answer=str(answer)
            )
            print(eval_result)

            # 평가 결과 파싱
            result_row = {
                'id': _id,
                'source': row.get('source', ''),
                'category': row.get('category', ''),
                'question': question,
                'gold': gold,
                'context': context,
                'answer': answer,
                'eval_structured': json.dumps(eval_result, ensure_ascii=False)
            }

            if eval_result:
                result_row.update({
                    'coherence': eval_result['coherence'],
                    'consistency': eval_result['consistency'],
                    'accuracy': eval_result['accuracy'],
                    'completeness': eval_result['completeness'],
                    'reasoning': eval_result['reasoning'],
                    'overall_quality': eval_result['overall_quality'],
                    'explanation': eval_result['explanation']
                })
            else:
                result_row.update({
                    'coherence': None,
                    'consistency': None,
                    'accuracy': None,
                    'completeness': None,
                    'reasoning': None,
                    'overall_quality': None,
                    'explanation': ''
                })

            results.append(result_row)
            
            # 진행 상황 출력 (간헐적으로)
            if (index + 1) % 10 == 0 or (index + 1) == total_count:
                if eval_result:
                    avg_score = sum([
                        eval_result.get('coherence', 0),
                        eval_result.get('consistency', 0),
                        eval_result.get('accuracy', 0),
                        eval_result.get('completeness', 0),
                        eval_result.get('reasoning', 0),
                        eval_result.get('overall_quality', 0)
                    ]) / 6
                    tqdm.write(f"  [{index + 1}/{total_count}] ID: {_id} - 평균 점수: {avg_score:.2f}")
                else:
                    tqdm.write(f"  [{index + 1}/{total_count}] ID: {_id} - 평가 실패")
            
        except KeyError as e:
            tqdm.write(f"Error: CSV 파일에 필요한 컬럼이 없습니다: {e}")
            error_ids.append(index)
            continue
        except Exception as e:
            tqdm.write(f"Error processing row {index}: {e}")
            error_ids.append(index)
            continue
    
    print(f"\n처리 완료: 총 {total_count}개 중 {len(results)}개 성공, {len(error_ids)}개 실패")
    if error_ids:
        print(f"실패한 행 인덱스: {error_ids}")
    
    # 결과를 DataFrame으로 변환 후 CSV 파일로 저장
    if results:
        results_df = pd.DataFrame(results)
        # CSV 파일 저장 (UTF-8 BOM 추가하여 Excel에서 한글 깨짐 방지)
        os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)
        results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n결과 저장 완료: {output_csv_path}")
    else:
        print("저장할 결과가 없습니다.")


# =================================
# 메인 실행 로직
# =================================
if __name__ == "__main__":
    try:
        # ==========================================
        # 1단계: 평가 모델 설정 (OpenAI API 모델)
        # ==========================================
        EVAL_MODEL = "gpt-5.2"
        
        # ==========================================
        # 2단계: 평가할 모델 리스트 설정 (2_1_gen_reasoning_openlm.py와 동일)
        # ==========================================
        TARGET_MODELS = [
            # "mistralai/Mistral-Small-3.2-24B-Instruct-2506",  
            # "mistralai/Ministral-3-14B-Instruct-2512",    
            # "mistralai/Ministral-3-8B-Instruct-2512", 
            # "mistralai/Ministral-3-3B-Instruct-2512", 
            # "Qwen/Qwen3-30B-A3B-Instruct-2507", 
            # "Qwen/Qwen3-30B-A3B-Thinking-2507", 
            # "Qwen/Qwen3-4B-Instruct-2507",
            # "Qwen/Qwen3-4B-Thinking-2507", 
            # "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            # "kakaocorp/kanana-2-30b-a3b-instruct",
            # "kakaocorp/kanana-1.5-15.7b-a3b-instruct",
            # "kakaocorp/kanana-1.5-8b-instruct-2505",
            # "kakaocorp/kanana-1.5-2.1b-instruct-2505",
            # "google/gemma-3-27b-it",
            # "google/gemma-3-12b-it",
            # "google/gemma-3-4b-it",
            # "google/gemma-3-1b-it",
            # "google/gemma-3-270m-it",
            # "microsoft/Phi-4-reasoning",
            # "microsoft/Phi-4-mini-instruct",
            # "microsoft/Phi-4-mini-reasoning",
            # "openai/gpt-oss-120b",
            # "openai/gpt-oss-20b",
            # "LGAI-EXAONE/EXAONE-4.0-32B",
            # "LGAI-EXAONE/EXAONE-4.0-1.2B",
            # "gpt-5-mini",
            # "gpt-5-nano",
            # "gpt-5",
            # "gpt-5.2",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-opus-4-5",
        ]
        
        # ==========================================
        # 3단계: 결과 디렉토리 설정
        # ==========================================
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results/2_fin_reasoning")
        os.makedirs(results_dir, exist_ok=True)
        
        # ==========================================
        # 4단계: 설정 정보 출력
        # ==========================================
        print("=" * 60)
        print("추론 응답 평가 모드 (OpenAI API)")
        print("=" * 60)
        print(f"평가 모델 (OpenAI API): {EVAL_MODEL}")
        print(f"평가할 모델 리스트: {len(TARGET_MODELS)}개")
        for hf_model in TARGET_MODELS:
            print(f"  ✓ {hf_model}")
        print(f"결과 디렉토리: {results_dir}")
        print("=" * 60)
        
        # ==========================================
        # 5단계: 각 모델에 대해 평가 실행
        # ==========================================
        total_models = len(TARGET_MODELS)
        
        for model_idx, hf_model in enumerate(TARGET_MODELS, 1):
            # 모델명은 HuggingFace 경로의 마지막 부분 사용
            model_name = hf_model.split("/")[-1]
            model_name_safe = model_name.replace("/", "_").replace(":", "_")
            
            print(f"\n{'='*60}")
            print(f"[모델 {model_idx}/{total_models}] {hf_model}")
            print(f"{'='*60}")
            
            # 입력 파일 경로 생성 (2_1_gen_reasoning_openlm.py와 동일한 형식)
            input_file_name = f"2_fin_reasoning_{model_name_safe}_answer.csv"
            input_csv_path = os.path.join(results_dir, input_file_name)
            
            # 출력 파일명 생성 (response.csv -> eval.csv로 변경)
            output_file_name = f"2_fin_reasoning_{model_name_safe}_eval.csv"
            output_csv_path = os.path.join(results_dir, output_file_name)
            
            # 입력 파일이 존재하는지 확인
            if not os.path.exists(input_csv_path):
                print(f"경고: 입력 파일을 찾을 수 없습니다. 스킵합니다: {input_csv_path}")
                continue
            
            # 이미 평가된 파일이면 스킵
            if os.path.exists(output_csv_path):
                print(f"이미 평가된 파일입니다. 스킵합니다: {output_csv_path}")
                continue
            
            # CSV 처리 및 평가 실행
            process_csv(input_csv_path, output_csv_path, eval_model=EVAL_MODEL)
            
            print(f"✓ {hf_model} 평가 완료")
        
        print("\n" + "=" * 60)
        print("모든 모델 평가 완료!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

