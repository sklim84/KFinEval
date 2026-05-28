"""
LLM 모델 평가 스크립트 (답변 생성 전용)

이 스크립트는 vLLM을 직접 사용하여 여러 LLM 모델을 순차적으로 평가합니다.
각 모델에 대해 벤치마크 데이터셋의 질문에 답변을 생성하고 결과를 CSV 파일로 저장합니다.
정답 비교 및 통계 계산은 1_2_stats_knowledge.py에서 수행합니다.
"""

import argparse
import pandas as pd
import os
import re
import torch
import shutil
import json
from typing import Optional, Tuple
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from tqdm import tqdm


def parse_mcq_answer_freeform(content: str):
    """추론 모델 자유 출력에서 A~E 추출 (--think 모드용).
    - `</think>` 뒤 텍스트만 사용
    - 그 다음 첫 A~E 글자
    (구 1_1_eval_plus_run_plain_eval.parse_knowledge_answer 와 동일 동작)
    """
    if not content:
        return None
    if "</think>" in content:
        content = content.split("</think>")[-1]
    s = content.strip().upper()
    if not s:
        return None
    if s[0] in "ABCDE":
        return s[0]
    for ch in "ABCDE":
        if ch in s:
            return ch
    return None

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
        # 기존 디렉토리가 있으면 삭제 (사용자가 수동으로 삭제할 예정이므로 경고만)
        print(f"경고: {hf_cache_old} 디렉토리가 존재합니다. 수동으로 삭제해주세요.")
    elif not os.path.exists(hf_cache_old):
        # 심볼릭 링크 생성
        try:
            os.makedirs(os.path.dirname(hf_cache_old), exist_ok=True)
            os.symlink(workspace_cache_dir, hf_cache_old)
            print(f"✓ HuggingFace 캐시 심볼릭 링크 생성: {hf_cache_old} -> {workspace_cache_dir}")
        except Exception as e:
            print(f"HuggingFace 캐시 심볼릭 링크 생성 중 오류 (무시): {e}")
    
    # vLLM 캐시 심볼릭 링크
    vllm_cache_old = os.path.join(home_cache, "vllm")
    if os.path.exists(vllm_cache_old) and not os.path.islink(vllm_cache_old):
        # 기존 디렉토리가 있으면 삭제 (사용자가 수동으로 삭제할 예정이므로 경고만)
        print(f"경고: {vllm_cache_old} 디렉토리가 존재합니다. 수동으로 삭제해주세요.")
    elif not os.path.exists(vllm_cache_old):
        # 심볼릭 링크 생성
        try:
            os.makedirs(os.path.dirname(vllm_cache_old), exist_ok=True)
            os.symlink(workspace_vllm_cache, vllm_cache_old)
            print(f"✓ vLLM 캐시 심볼릭 링크 생성: {vllm_cache_old} -> {workspace_vllm_cache}")
        except Exception as e:
            print(f"vLLM 캐시 심볼릭 링크 생성 중 오류 (무시): {e}")

# 심볼릭 링크 설정 실행
setup_cache_symlinks()


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
                      긴 컨텍스트 모델의 경우 메모리 부족 시 줄여야 함
    
    Returns:
        LLM 객체
    """
    print(f"\n모델 로딩 중: {hf_model}")
    print(f"GPU 메모리 사용률: {gpu_memory_utilization}")
    if max_model_len:
        print(f"최대 시퀀스 길이: {max_model_len}")
    
    # 환경 변수 확인 (이미 모듈 로드 전에 설정됨)
    print(f"HuggingFace 캐시: {os.getenv('HUGGINGFACE_HUB_CACHE', '기본값 사용')}")
    
    try:
        llm_params = {
            "model": hf_model,
            "tensor_parallel_size": torch.cuda.device_count(),
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "dtype": "auto",
        }
        
        if max_model_len is not None:
            llm_params["max_model_len"] = max_model_len

        llm = LLM(**llm_params)
        print(f"✓ 모델 로딩 완료: {hf_model}")
        return llm
    except Exception as e:
        print(f"✗ 모델 로딩 실패: {e}")
        raise


# =================================
# 프롬프트 생성 함수
# =================================
def create_prompt(question: str, A: str, B: str, C: str, D: str, E: str) -> str:
    """
    객관식 질문 프롬프트 생성
    
    Args:
        question: 질문
        A, B, C, D, E: 선택지
    
    Returns:
        프롬프트 문자열
    """
    return f"""다음 객관식 질문에 답하세요.

질문:
{question}

선택지:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
"""

# =================================
# 답변 생성 함수 (단일 처리)
# =================================
def generate_answer_single(
    model: LLM,
    prompt: str,
    sampling_params: SamplingParams
) -> Tuple[Optional[str], Optional[str]]:
    """
    단일 질문에 대한 답변 생성
    
    Args:
        model: LLM 객체
        prompt: 프롬프트 문자열
        sampling_params: 샘플링 파라미터 (StructuredOutputsParams 포함)
    
    Returns:
        (파싱된 답변, 원본 답변 텍스트, 전체 output plain text) 튜플
        - StructuredOutputsParams로 인해 항상 A~E 중 하나가 반환됨
    """
    try:
        outputs = model.generate([prompt], sampling_params)
        answer = outputs[0].outputs[0].text.strip()
        outputs_text = str(outputs)
        print(f'answer: {answer}')

        return (answer, outputs_text)

    except Exception as e:
        print(f"  생성 오류: {e}")
        import traceback
        traceback.print_exc()
        return (None, None)

# =================================
# CSV 파일 처리 함수
# =================================
def process_csv(
    model: LLM,
    model_name: str,
    input_csv_path: str,
    output_csv_path: str,
    sampling_params: SamplingParams,
    think: bool = False,
    limit: Optional[int] = None,
) -> None:
    """
    CSV 파일을 읽어서 모델로 답변 생성 수행 (단일 처리)
    정답 비교 및 통계 계산은 포함하지 않습니다.
    
    Args:
        model: LLM 객체
        model_name: 모델 이름
        input_csv_path: 입력 CSV 파일 경로
        output_csv_path: 출력 CSV 파일 경로
    """
    # 입력 CSV 파일 읽기
    data = pd.read_csv(input_csv_path)
    if limit:
        data = data.head(limit)
        print(f"  [limit] 앞 {limit} 행만 처리")

    # 결과 저장용 리스트
    results = []
    total_count = len(data)
    success_count = 0
    error_ids = []
    
    print(f"\n평가 모델: {model_name}")
    print(f"총 {total_count}개 질문 처리 시작 (단일 처리 모드)...\n")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)
    
    # 단일 처리: 각 질문을 하나씩 처리
    for index, row in tqdm(data.iterrows(), total=total_count, desc=f"처리 중 ({model_name})"):
        try:
            _id = row['id']
            category = row['category']
            sub_category = row['sub_category']
            level = row['level']
            has_table = row['has_table']
            has_fomula = row['has_fomula']
            question = row['question']
            A = row['A']
            B = row['B']
            C = row['C']
            D = row['D']
            E = row['E']
            gold = row['gold']
            
            prompt = create_prompt(question, A, B, C, D, E)

            # 단일 질문에 대한 답변 생성
            answer, outputs_text = generate_answer_single(
                model, prompt, sampling_params
            )

            # --think 모드: 자유 생성된 raw 출력에서 A~E 추출
            if think and answer is not None:
                answer = parse_mcq_answer_freeform(answer)

            # 결과 저장
            result_row = {
                'id': _id,
                'category': category,
                'sub_category': sub_category,
                'level': level,
                'has_table': has_table,
                'has_fomula': has_fomula,
                'question': question,
                'A': A,
                'B': B,
                'C': C,
                'D': D,
                'E': E,
                'gold': gold,
                "answer": answer,
                # 다른 backend (openrouter / hf_direct / vaetki) 와 컬럼 contract 일치:
                # answer_structured (think 모드 미사용) + raw_response (전체 vLLM output).
                # 이렇게 두면 1_2_stats_eval_knowledge.py --llm-judge 가 모든 backend에서 동작.
                "answer_structured": None,
                "raw_response": outputs_text,
            }
            
            results.append(result_row)
            
            # 진행 상황 출력
            current_idx = index + 1
            print(f"  [{current_idx}/{total_count}] ID: {_id} - {model_name}: {answer} (정답: {gold})")
            success_count += 1
            
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
    # ==========================================
    # CLI 인자
    #   --think : 추론 모델 모드 (StructuredOutputsParams 미사용, max_tokens 크게)
    #   --max-tokens N : 명시적 max_tokens 지정 (기본: think=8192, structured=5)
    # ==========================================
    _cli = argparse.ArgumentParser(add_help=True)
    _cli.add_argument("--think", action="store_true",
                      help="추론 모델 모드: structured 미사용, max_tokens 크게, "
                           "</think>+regex 로 A~E 추출. 정밀 판정은 1_2_stats_eval_knowledge.py --llm-judge 사용.")
    _cli.add_argument("--max-tokens", type=int, default=None,
                      help="max_tokens override (기본: --think 시 8192, structured 시 5)")
    _cli.add_argument("--limit", type=int, default=None,
                      help="앞 N개만 처리 (디버그/스모크 테스트)")
    _args, _ = _cli.parse_known_args()
    THINK = _args.think
    MAX_TOKENS = _args.max_tokens if _args.max_tokens is not None else (8192 if THINK else 5)
    LIMIT = _args.limit

    try:
        # ==========================================
        # 1단계: 평가할 모델 설정 (추가 모델: huggingface 모델경로)
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
            # "LGAI-EXAONE/EXAONE-4.0-32B",
            # "LGAI-EXAONE/EXAONE-4.0-1.2B",
        ]
        
        # GPU 메모리 사용률 설정 (필요시 모델별로 다르게 설정 가능)
        GPU_MEMORY_UTILIZATION = 0.9  # 메모리 부족 시 이 값을 높임 (0.7~0.9 권장)
        
        # 최대 시퀀스 길이 설정 (긴 컨텍스트 모델의 경우 메모리 부족 시 줄여야 함)
        # None이면 모델 기본값 사용, 객관식 평가에는 보통 8192~32768이면 충분
        MAX_MODEL_LEN = 32768  # 262144 → 32768로 줄여서 메모리 사용량 감소
        
        # ==========================================
        # 2단계: 벤치마크 데이터셋 설정
        # ==========================================
        # 데이터셋 경로 (스크립트 위치 기반 동적; 옛 환경 /workspace/Fin-Ben 경로 의존 제거)
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "_datasets", "0_integration",
        )
        benchmark_list = [
            "1_fin_knowledge.csv"
        ]
        
        if THINK:
            # 추론 모델 모드: 자유 생성, structured 미사용
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=MAX_TOKENS,  # 기본 8192
                seed=2025,
            )
        else:
            # 비추론 모델: structured 로 A~E 강제
            structured_params = StructuredOutputsParams(
                choice=["A", "B", "C", "D", "E"]
            )
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=MAX_TOKENS,  # 기본 5
                seed=2025,
                structured_outputs=structured_params,
            )
        # ==========================================
        # 3단계: 설정 정보 출력
        # ==========================================
        print("=" * 60)
        print("vLLM 직접 사용 평가 모드")
        print("=" * 60)
        print("평가 모델 설정:")
        for hf_model in TARGET_MODELS:
            print(f"  ✓ {hf_model}")
        print("=" * 60)
        print(f"벤치마크 데이터셋: {len(benchmark_list)}개")
        for benchmark in benchmark_list:
            print(f"  - {benchmark}")
        print(f"처리 모드: 단일 처리 (1건씩)")
        print(f"GPU 메모리 사용률: {GPU_MEMORY_UTILIZATION}")
        if MAX_MODEL_LEN:
            print(f"최대 시퀀스 길이: {MAX_MODEL_LEN}")
        print("=" * 60)
        
        # ==========================================
        # 4단계: 모델별 순차 평가 실행
        # ==========================================
        total_models = len(TARGET_MODELS)
        total_benchmarks = len(benchmark_list)
        
        for model_idx, hf_model in enumerate(TARGET_MODELS, 1):
            # 모델명은 HuggingFace 경로의 마지막 부분 사용
            model_name = hf_model.split("/")[-1]
            
            print(f"\n{'='*60}")
            print(f"[모델 {model_idx}/{total_models}] {hf_model}")
            print(f"{'='*60}")
            
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
            
            # 벤치마크 평가 실행
            print(f"\n[2/3] 벤치마크 평가 시작...")
            for bench_idx, benchmark_name in enumerate(benchmark_list, 1):
                print(f"\n  [벤치마크 {bench_idx}/{total_benchmarks}] {benchmark_name}")
                
                input_csv_path = os.path.join(file_path, benchmark_name)
                # 출력 파일명에 모델명 포함
                model_name_safe = model_name.replace("/", "_").replace(":", "_")
                # 출력 디렉토리: eval/_results/
                results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results/1_fin_knowledge")
                os.makedirs(results_dir, exist_ok=True)
                output_csv_path = os.path.join(
                    results_dir,
                    f"{benchmark_name.replace('.csv', '')}_{model_name_safe}_response.csv"
                )
                
                # CSV 처리 및 평가 실행
                process_csv(model, model_name, input_csv_path, output_csv_path, sampling_params, think=THINK, limit=LIMIT)
                
                print(f"  ✓ {benchmark_name} 평가 완료")
            
            # 모델 메모리 해제
            print(f"\n[3/3] 모델 메모리 해제 중...")
            del model
            torch.cuda.empty_cache()  # GPU 메모리 정리
            print(f"✓ {hf_model} 평가 완료")
            
            # 모델 파일 삭제 (디스크 관리)
            delete_model_cache(hf_model)
            
            print()
        
        print("=" * 60)
        print("모든 모델 평가 완료!")
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


