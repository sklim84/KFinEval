"""
추론 응답 생성 스크립트 (Anthropic Claude API 사용)

이 스크립트는 Anthropic Claude API를 사용하여 주어진 문맥과 질문에 대한 단계별 추론을 생성합니다.
각 질문에 대해 JSON 형식의 단계별 추론 과정을 생성하고 결과를 CSV 파일로 저장합니다.

- Claude Messages API 사용
- 프롬프트 기반 JSON 형식 응답 요청
- 응답 파싱으로 구조화된 답변 추출
- anthropic>=0.34.x 완전 호환
"""

import os
import json
import time
import random
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

def create_reasoning_prompt(context: str, question: str) -> str:
    """
    추론 생성용 프롬프트 생성
    
    Args:
        context: 문맥
        question: 질문
    
    Returns:
        프롬프트 문자열
    """
    return f"""주어진 문맥을 기반으로 질문에 대한 답을 작성해주세요. 답은 반드시 결론을 포함한 추론과정을 단계별로 작성해야합니다.

문맥: {context}
질문: {question}

- 문체는 반드시 문어체로 작성해주세요.
- 추론과정에 거짓된 정보는 없어야 합니다.
- 반드시 한글로 작성되어야 합니다.

출력 형식은 JSON 으로 작성하며, 반드시 아래와 같은 예시를 따르세요:
{{
"step 1": "추론 내용 1",
"step 2": "추론 내용 2",
"step 3": "추론 내용 3",
...
}}
"""

# =================================
# 추론 텍스트 추출
# =================================
def extract_reasoning_text(response) -> Optional[str]:
    """
    Claude API 응답에서 추론 텍스트를 추출합니다.
    
    Claude API 응답 구조:
    - response.content: TextBlock 객체들의 리스트
    - 각 TextBlock은 text 속성과 type 속성을 가짐
    - type은 보통 'text' 문자열
    """
    try:
        # Messages API 응답 구조: response.content는 TextBlock 리스트
        if hasattr(response, "content") and response.content:
            # 모든 텍스트 블록의 텍스트를 합침
            full_text = ""
            for block in response.content:
                # TextBlock 객체 확인 (type='text'인 경우)
                if hasattr(block, "type") and block.type == "text":
                    if hasattr(block, "text") and block.text:
                        full_text += block.text + "\n"
                # type 속성이 없어도 text 속성이 있으면 처리
                elif hasattr(block, "text") and block.text:
                    full_text += block.text + "\n"
            
            if full_text.strip():
                return full_text.strip()
        
        return None
    except Exception as e:
        print(f"extract_reasoning_text 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

# =================================
# 추론 생성 함수 (retry 포함)
# =================================
def generate_reasoning(
    model_name: str,
    context: str,
    question: str,
    max_tokens: int = 2048,
    max_retries: int = 5
) -> Tuple[Optional[str], Optional[str]]:
    """
    주어진 문맥과 질문에 대한 추론 생성
    
    Args:
        model_name: Claude 모델 이름
        context: 문맥
        question: 질문
        max_tokens: 최대 출력 토큰 수
        max_retries: 최대 재시도 횟수
    
    Returns:
        (생성된 추론 텍스트(JSON 문자열), raw_response JSON 문자열) 튜플
    """
    prompt = create_reasoning_prompt(context, question)
    
    for attempt in range(max_retries):
        try:
            # Claude Messages API 사용
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # 일반 텍스트 추출
            reasoning_text = extract_reasoning_text(response)
            
            if not reasoning_text:
                print(f"  추론 텍스트 추출 실패, 재시도 ({attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None, None
            
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
            
            print(f'reasoning_text: {reasoning_text}')
            return reasoning_text, raw_response_json

        except anthropic.APIError as e:
            if e.status_code == 500 or e.status_code == 503:
                if attempt == max_retries - 1:
                    print(f"  ⚠️ {e.status_code} 오류 재시도 실패 (최종)")
                    return None, None
                sleep_time = 2 ** attempt + random.uniform(0, 0.5)
                print(f"  ⚠️ {e.status_code} 서버 오류, {sleep_time:.2f}s 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                print(f"  생성 오류 (API): {e}")
                import traceback
                traceback.print_exc()
                return None, None

        except Exception as e:
            print(f"  생성 오류: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    return None, None

# =================================
# CSV 파일 처리 함수
# =================================
def process_csv(
    model_name: str,
    input_csv_path: str,
    output_csv_path: str,
    max_tokens: int = 2048
) -> None:
    """
    CSV 파일을 읽어서 모델로 추론 생성 수행
    
    Args:
        model_name: 모델 이름
        input_csv_path: 입력 CSV 파일 경로
        output_csv_path: 출력 CSV 파일 경로
        max_tokens: 최대 출력 토큰 수
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
    success_count = 0
    
    print(f"\n평가 모델: {model_name}")
    print(f"총 {total_count}개 질문 처리 시작...\n")
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    for index, row in tqdm(data.iterrows(), total=len(data), desc=f"처리 중 ({model_name})"):
        try:
            _id = row['id']
            source = row['source']
            category = row['category']
            question = row['question']
            context = row['context']
            gold = row['gold']
            
            # 추론 생성 (context와 question 사용)
            answer, raw_response = generate_reasoning(
                model_name=model_name,
                context=context,
                question=question,
                max_tokens=max_tokens
            )
            
            # 결과 저장 (원본 데이터 + 생성된 답변)
            result_row = {
                'id': _id,
                'source': source,
                'category': category,
                'question': question,
                'context': context,
                'gold': gold,
                'answer': answer,
            }
            
            # raw_response가 있으면 추가
            if raw_response:
                result_row['raw_response'] = raw_response
            
            results.append(result_row)
            
            if answer is not None:
                success_count += 1
            else:
                error_ids.append(index)
            
            # 진행 상황 출력 (간헐적으로)
            if (index + 1) % 10 == 0 or (index + 1) == total_count:
                print(f"  [{index + 1}/{total_count}] ID: {_id} - 처리 완료")
            
        except KeyError as e:
            print(f"Error: CSV 파일에 필요한 컬럼이 없습니다: {e}")
            error_ids.append(index)
            continue
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            error_ids.append(index)
            continue

    print(f"\n처리 완료: 총 {total_count}개 중 {success_count}개 성공, {len(error_ids)}개 실패")
    if error_ids:
        print(f"실패한 행 인덱스: {error_ids}")

    # 결과를 DataFrame으로 변환 후 CSV 파일로 저장
    if results:
        results_df = pd.DataFrame(results)
        # CSV 파일 저장 (UTF-8 BOM 추가하여 Excel에서 한글 깨짐 방지)
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
        # 1단계: 평가할 모델 설정
        # ==========================================
        TARGET_MODELS = [
            # "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-opus-4-5",
        ]
        
        # 최대 출력 토큰 수 설정 (추론 생성이므로 충분히 길게 설정)
        MAX_TOKENS = 2048
        
        # ==========================================
        # 2단계: 입력 파일 경로 설정
        # ==========================================
        input_csv_path = "/workspace/Fin-Ben/_datasets/0_integration/2_fin_reasoning.csv"
        
        # ==========================================
        # 3단계: 설정 정보 출력
        # ==========================================
        print("=" * 60)
        print("Anthropic Claude API 추론 생성 모드")
        print("=" * 60)
        print("평가 모델 설정:")
        for model_name in TARGET_MODELS:
            print(f"  ✓ {model_name}")
        print("=" * 60)
        print(f"입력 파일: {input_csv_path}")
        print(f"최대 출력 토큰 수: {MAX_TOKENS}")
        print("=" * 60)
        
        # ==========================================
        # 4단계: 모델별 순차 추론 생성 실행
        # ==========================================
        total_models = len(TARGET_MODELS)
        
        for model_idx, model_name in enumerate(TARGET_MODELS, 1):
            model_name_safe = model_name.replace("/", "_").replace(":", "_")
            
            print(f"\n{'='*60}")
            print(f"[모델 {model_idx}/{total_models}] {model_name}")
            print(f"{'='*60}")
            
            # 출력 디렉토리: eval/_results/2_fin_reasoning/
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results/2_fin_reasoning")
            os.makedirs(results_dir, exist_ok=True)
            output_csv_path = os.path.join(
                results_dir,
                f"2_fin_reasoning_{model_name_safe}_answer.csv"
            )
            
            # 추론 생성 실행
            print(f"\n추론 생성 시작...")
            process_csv(
                model_name=model_name,
                input_csv_path=input_csv_path,
                output_csv_path=output_csv_path,
                max_tokens=MAX_TOKENS
            )
            
            print(f"✓ {model_name} 처리 완료\n")
        
        print("=" * 60)
        print("모든 모델 처리 완료!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
