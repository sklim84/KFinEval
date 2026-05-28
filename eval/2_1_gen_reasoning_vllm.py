"""
추론 응답 생성 스크립트 (vLLM 사용)

이 스크립트는 vLLM을 직접 사용하여 주어진 문맥과 질문에 대한 단계별 추론을 생성합니다.
각 질문에 대해 JSON 형식의 단계별 추론 과정을 생성하고 결과를 CSV 파일로 저장합니다.
"""

import pandas as pd
import json
import os
import torch
import shutil
from tqdm import tqdm
from typing import Optional

# HuggingFace 토큰 설정 (gated repository 접근용)
# 환경 변수에 토큰이 없으면 기본 토큰 사용
if "HF_TOKEN" not in os.environ and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
    default_token = "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"
    os.environ["HF_TOKEN"] = default_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = default_token
    print("✓ HuggingFace 토큰 설정 완료 (기본 토큰 사용)")
else:
    print("✓ HuggingFace 토큰 확인됨 (환경 변수에서 로드)")

# HuggingFace 캐시를 workspace로 설정 (모듈 로드 전에 설정해야 함)
workspace_cache_dir = "/workspace/.cache/huggingface"
os.makedirs(workspace_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = workspace_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(workspace_cache_dir, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(workspace_cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(workspace_cache_dir, "datasets")

# vLLM 캐시도 workspace로 설정
workspace_vllm_cache = "/workspace/.cache/vllm"
os.makedirs(workspace_vllm_cache, exist_ok=True)

# 심볼릭 링크 생성 (기존 캐시는 삭제 예정이므로 이동하지 않음)
def setup_cache_symlinks():
    """workspace 캐시 디렉토리로 심볼릭 링크 생성"""
    home_cache = os.path.expanduser("~/.cache")
    
    # HuggingFace 캐시 심볼릭 링크
    hf_cache_old = os.path.join(home_cache, "huggingface")
    if os.path.exists(hf_cache_old) and not os.path.islink(hf_cache_old):
        print(f"경고: {hf_cache_old} 디렉토리가 존재합니다. 수동으로 삭제해주세요.")
    elif not os.path.exists(hf_cache_old):
        try:
            os.makedirs(os.path.dirname(hf_cache_old), exist_ok=True)
            os.symlink(workspace_cache_dir, hf_cache_old)
            print(f"✓ HuggingFace 캐시 심볼릭 링크 생성: {hf_cache_old} -> {workspace_cache_dir}")
        except Exception as e:
            print(f"HuggingFace 캐시 심볼릭 링크 생성 중 오류 (무시): {e}")
    
    # vLLM 캐시 심볼릭 링크
    vllm_cache_old = os.path.join(home_cache, "vllm")
    if os.path.exists(vllm_cache_old) and not os.path.islink(vllm_cache_old):
        print(f"경고: {vllm_cache_old} 디렉토리가 존재합니다. 수동으로 삭제해주세요.")
    elif not os.path.exists(vllm_cache_old):
        try:
            os.makedirs(os.path.dirname(vllm_cache_old), exist_ok=True)
            os.symlink(workspace_vllm_cache, vllm_cache_old)
            print(f"✓ vLLM 캐시 심볼릭 링크 생성: {vllm_cache_old} -> {workspace_vllm_cache}")
        except Exception as e:
            print(f"vLLM 캐시 심볼릭 링크 생성 중 오류 (무시): {e}")

# 심볼릭 링크 설정 실행
setup_cache_symlinks()

from vllm import LLM, SamplingParams

# =================================
# 모델 관리 함수
# =================================
def delete_model_cache(hf_model: str) -> bool:
    """
    HuggingFace 모델 캐시 디렉토리 삭제
    
    Args:
        hf_model: HuggingFace 모델 경로 (예: "Qwen/Qwen3-4B-Instruct-2507")
    
    Returns:
        삭제 성공 여부
    """
    try:
        # HuggingFace 캐시 디렉토리 경로
        workspace_cache_dir = "/workspace/.cache/huggingface"
        hub_cache_dir = os.path.join(workspace_cache_dir, "hub")
        
        # 모델 디렉토리명 변환: "Qwen/Qwen3-4B-Instruct-2507" -> "models--Qwen--Qwen3-4B-Instruct-2507"
        model_dir_name = f"models--{hf_model.replace('/', '--')}"
        model_cache_path = os.path.join(hub_cache_dir, model_dir_name)
        
        if os.path.exists(model_cache_path):
            print(f"\n모델 캐시 삭제 중: {model_cache_path}")
            shutil.rmtree(model_cache_path)
            print(f"✓ 모델 캐시 삭제 완료: {hf_model}")
            return True
        else:
            print(f"모델 캐시를 찾을 수 없습니다: {model_cache_path}")
            return False
    except Exception as e:
        print(f"모델 캐시 삭제 중 오류 발생: {e}")
        return False

def load_model(
    hf_model: str, 
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None
) -> LLM:
    """
    vLLM 모델 로드
    
    Args:
        hf_model: HuggingFace 모델 경로
        gpu_memory_utilization: GPU 메모리 사용률 (0.0 ~ 1.0)
        max_model_len: 최대 시퀀스 길이 (None이면 모델 기본값 사용)
    
    Returns:
        LLM 객체
    """
    print(f"\n모델 로딩 중: {hf_model}")
    print(f"GPU 메모리 사용률: {gpu_memory_utilization}")
    if max_model_len:
        print(f"최대 시퀀스 길이: {max_model_len}")
    
    try:
        llm_params = {
            "model": hf_model,
            "tensor_parallel_size": torch.cuda.device_count(),
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "dtype": "bfloat16",
        }
        
        if max_model_len is not None:
            llm_params["max_model_len"] = max_model_len
        
        # 멀티모달 모델의 경우 Flash Attention 호환성 문제로 인해 비활성화
        if "omni" in hf_model.lower() or "multimodal" in hf_model.lower():
            print("경고: 멀티모달 모델 감지. Flash Attention 호환성 문제를 피하기 위해 enforce_eager 모드 사용")
            llm_params["enforce_eager"] = True
        
        llm = LLM(**llm_params)
        print(f"✓ 모델 로딩 완료: {hf_model}")
        return llm
    except Exception as e:
        print(f"✗ 모델 로딩 실패: {e}")
        raise


# =================================
# 프롬프트 생성 함수
# =================================
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
# 추론 생성 함수
# =================================
def generate_reasoning(model: LLM, context: str, question: str, sampling_params: SamplingParams) -> Optional[str]:
    """
    주어진 문맥과 질문에 대한 추론 생성
    
    Args:
        model: LLM 객체
        context: 문맥
        question: 질문
        sampling_params: 샘플링 파라미터
    
    Returns:
        생성된 추론 텍스트 또는 None (오류 시)
    """
    prompt = create_reasoning_prompt(context, question)
    
    try:
        outputs = model.generate([prompt], sampling_params)
        reasoning_text = outputs[0].outputs[0].text.strip()
        # print(reasoning_text)
        return reasoning_text
    except Exception as e:
        print(f"  생성 오류: {e}")
        return None
    
# =================================
# CSV 파일 처리 함수
# =================================
def process_csv(model: LLM, model_name: str, input_csv_path: str, output_csv_path: str):
    """
    CSV 파일을 읽어서 모델로 추론 생성 수행
    
    Args:
        model: LLM 객체
        model_name: 모델 이름 (출력 파일명에 사용)
        input_csv_path: 입력 CSV 파일 경로
        output_csv_path: 출력 CSV 파일 경로
    """
    # 입력 CSV 파일 읽기
    try:
        data = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return
    
    print(f"CSV 파일 로드 완료: {data.shape[0]}개 행, {data.shape[1]}개 컬럼")
    print(f"컬럼: {list(data.columns)}")

    # 샘플링 파라미터 설정
    sampling_params = SamplingParams(
        temperature=0.7,  # 추론 생성이므로 약간의 다양성 허용
        max_tokens=2048,  # 추론 과정이 길 수 있으므로 충분한 토큰 수 설정
        stop=["\n\n\n"],  # 연속된 줄바꿈에서 중단
    )

    # 결과 저장용 리스트
    results = []
    error_ids = []
    total_count = len(data)
    
    print(f"\n평가 모델: {model_name}")
    print(f"총 {total_count}개 질문 처리 시작...\n")
    
    for index, row in tqdm(data.iterrows(), total=len(data)):
        try:
            _id = row['id']
            source = row['source']
            category = row['category']
            question = row['question']
            context = row['context']
            gold = row['gold']
            
            # 추론 생성 (context와 question 사용)
            answer = generate_reasoning(model, context, question, sampling_params)
            
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
            
            results.append(result_row)
            
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

    print(f"\n처리 완료: 총 {total_count}개 중 {len(results)}개 성공, {len(error_ids)}개 실패")
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
        # 1단계: 평가할 모델 설정 (1_1_eval_knowledge.py와 동일한 모델 리스트)
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
            "LGAI-EXAONE/EXAONE-4.0-32B",
            "LGAI-EXAONE/EXAONE-4.0-1.2B",
        ]
        
        # GPU 메모리 사용률 설정 (필요시 모델별로 다르게 설정 가능)
        GPU_MEMORY_UTILIZATION = 0.9  # 메모리 부족 시 이 값을 높임 (0.7~0.9 권장)
        
        # 최대 시퀀스 길이 설정 (추론 생성이므로 충분히 길게 설정)
        MAX_MODEL_LEN = 32768  # 필요시 조정
        
        # ==========================================
        # 2단계: 입력 파일 경로 설정
        # ==========================================
        input_csv_path = "/workspace/Fin-Ben/_datasets/0_integration/2_fin_reasoning.csv"
        
        # ==========================================
        # 3단계: 설정 정보 출력
        # ==========================================
        print("=" * 60)
        print("vLLM 직접 사용 추론 생성 모드")
        print("=" * 60)
        print("평가 모델 설정:")
        for hf_model in TARGET_MODELS:
            print(f"  ✓ {hf_model}")
        print("=" * 60)
        print(f"입력 파일: {input_csv_path}")
        print(f"처리 모드: 단일 처리 (1건씩)")
        print(f"GPU 메모리 사용률: {GPU_MEMORY_UTILIZATION}")
        if MAX_MODEL_LEN:
            print(f"최대 시퀀스 길이: {MAX_MODEL_LEN}")
        print("=" * 60)
        
        # ==========================================
        # 4단계: 모델별 순차 추론 생성 실행
        # ==========================================
        total_models = len(TARGET_MODELS)
        
        for model_idx, hf_model in enumerate(TARGET_MODELS, 1):
            # 모델명은 HuggingFace 경로의 마지막 부분 사용
            model_name = hf_model.split("/")[-1]
            model_name_safe = model_name.replace("/", "_").replace(":", "_")
            
            print(f"\n{'='*60}")
            print(f"[모델 {model_idx}/{total_models}] {hf_model}")
            print(f"{'='*60}")
            
            # 출력 디렉토리: eval/output/
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results/2_fin_reasoning")
            os.makedirs(results_dir, exist_ok=True)
            output_csv_path = os.path.join(
                results_dir,
                f"2_fin_reasoning_{model_name_safe}_answer.csv"
            )
            
            # 모델 로드
            print(f"\n[1/3] 모델 로딩 중...")
            try:
                model = load_model(
                    hf_model,
                    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                    max_model_len=MAX_MODEL_LEN
                )
            except Exception as e:
                print(f"✗ 모델 '{hf_model}' 로딩 실패. 다음 모델로 진행합니다.")
                print(f"  오류: {e}")
                continue
            
            print(f"✓ 모델 준비 완료")
            
            # 추론 생성 실행
            print(f"\n[2/3] 추론 생성 시작...")
            process_csv(model, model_name, input_csv_path, output_csv_path)
            
            # 모델 메모리 해제
            print(f"\n[3/3] 모델 메모리 해제 중...")
            del model
            torch.cuda.empty_cache()  # GPU 메모리 정리
            print(f"✓ {hf_model} 처리 완료")
            
            # 모델 파일 삭제 (디스크 관리)
            delete_model_cache(hf_model)
            
            print()
        
        print("=" * 60)
        print("모든 모델 처리 완료!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        print("\nGPU 메모리 정리 완료")