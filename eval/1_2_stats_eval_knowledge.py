"""
LLM 모델 평가 결과 통계 계산 스크립트

`1_1_eval_knowledge_{vllm,openrouter}.py` 가 생성한 `_response.csv` 를 읽어
정답 비교, `is_correct` 추가, `_response_stats.json` 생성.

옵션:
  기본       : 단순 문자열 매치 (`check_answer(answer, gold)`)
  --llm-judge: LLM judge (기본 OpenRouter `openai/gpt-5.2`) 로 정답 여부 판별.
               --think 모드 자유 출력의 정확 판정을 위해 사용.
               (구 1_2_eval_plus_judge_knowledge_response.py 의 기능 흡수)
"""

import argparse
import pandas as pd
import os
import json
import glob
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, List

from tqdm import tqdm


# =================================
# 카테고리 번역 딕셔너리 (vis_category_detail.py 참고)
# =================================
category_translation = {
    # 회계 관련
    '중급회계': 'Intermediate Accounting',
    '세법': 'Tax Law',
    
    # 경제학 관련
    '미시경제학': 'Microeconomics',
    '거시경제학': 'Macroeconomics',
    '국제경제학': 'International Economics',
    '계량경제': 'Econometrics',
    
    # 재무관리 관련
    '재무관리': 'Financial Management',
    
    # 금융시장 관련
    '금융상품': 'Financial Products',
    '금융의 기초': 'Financial Fundamentals',
    '금융기관': 'Financial Institutions',
    '화폐금융': 'Monetary Finance',
    
    # 시장 관련
    '증권시장': 'Securities Market',
    '채권시장': 'Bond Market',
    '부동산시장': 'Real Estate Market',
    '유통시장': 'Distribution Market',
    
    # 파생상품 관련
    '파생상품': 'Derivatives',
    
    # 디지털 금융 관련
    '디지털 금융': 'Digital Finance',
    
    # 기타
    '생산운영관리': 'Production & Operations Management',
    '보험상품': 'Insurance Products',
    '국제금융정책': 'International Financial Policy',
    
    # 빈 값 처리
    '': 'Uncategorized',
}


# =================================
# 정답 비교 함수
# =================================
def check_answer(predicted: Optional[str], gold: str) -> bool:
    """
    모델 답변과 정답 비교
    
    Args:
        predicted: 모델이 예측한 답변 (A~E 또는 None/NaN)
        gold: 정답 (A~E)
    
    Returns:
        정답 여부 (True/False)
    """
    # None 또는 NaN 체크 (pandas에서 읽은 NaN은 float 타입)
    if predicted is None or pd.isna(predicted):
        return False
    # 문자열로 변환 후 대소문자 구분 없이 비교
    return str(predicted).upper().strip() == str(gold).upper().strip()


# =================================
# 통계 계산 함수
# =================================
def calculate_statistics(results_df: pd.DataFrame, model_name: str) -> Dict:
    """
    평가 결과 통계 계산
    
    Args:
        results_df: 결과 DataFrame
        model_name: 모델 이름
    
    Returns:
        통계 딕셔너리
    """
    answer_col = "answer"
    is_correct_col = "is_correct"
    
    # 전체 통계
    total = len(results_df)
    correct = results_df[is_correct_col].sum()
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    stats = {
        "model": model_name,
        "total_questions": int(total),
        "correct_answers": int(correct),
        "wrong_answers": int(total - correct),
        "accuracy": round(accuracy, 2),
        "accuracy_percentage": f"{accuracy:.2f}%"
    }
    
    # 카테고리별 통계
    if 'category' in results_df.columns:
        category_stats = []
        for category in results_df['category'].unique():
            cat_df = results_df[results_df['category'] == category]
            cat_total = len(cat_df)
            cat_correct = cat_df[is_correct_col].sum()
            cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0.0
            category_stats.append({
                "category": category,
                "total": int(cat_total),
                "correct": int(cat_correct),
                "wrong": int(cat_total - cat_correct),
                "accuracy": round(cat_accuracy, 2),
                "accuracy_percentage": f"{cat_accuracy:.2f}%"
            })
        stats["by_category"] = category_stats
    
    # 레벨별 통계
    if 'level' in results_df.columns:
        level_stats = []
        for level in sorted(results_df['level'].unique()):
            level_df = results_df[results_df['level'] == level]
            level_total = len(level_df)
            level_correct = level_df[is_correct_col].sum()
            level_accuracy = (level_correct / level_total * 100) if level_total > 0 else 0.0
            level_stats.append({
                "level": level,
                "total": int(level_total),
                "correct": int(level_correct),
                "wrong": int(level_total - level_correct),
                "accuracy": round(level_accuracy, 2),
                "accuracy_percentage": f"{level_accuracy:.2f}%"
            })
        stats["by_level"] = level_stats
    
    # 서브카테고리별 통계
    if 'sub_category' in results_df.columns:
        subcat_stats = []
        for subcat in results_df['sub_category'].unique():
            subcat_df = results_df[results_df['sub_category'] == subcat]
            subcat_total = len(subcat_df)
            subcat_correct = subcat_df[is_correct_col].sum()
            subcat_accuracy = (subcat_correct / subcat_total * 100) if subcat_total > 0 else 0.0
            subcat_stats.append({
                "sub_category": subcat,
                "total": int(subcat_total),
                "correct": int(subcat_correct),
                "wrong": int(subcat_total - subcat_correct),
                "accuracy": round(subcat_accuracy, 2),
                "accuracy_percentage": f"{subcat_accuracy:.2f}%"
            })
        # 정확도 순으로 정렬
        subcat_stats.sort(key=lambda x: x['accuracy'], reverse=True)
        stats["by_sub_category"] = subcat_stats
    
    # 테이블/수식 포함 여부별 통계
    if 'has_table' in results_df.columns:
        table_stats = []
        for has_table in sorted(results_df['has_table'].unique()):
            table_df = results_df[results_df['has_table'] == has_table]
            table_total = len(table_df)
            table_correct = table_df[is_correct_col].sum()
            table_accuracy = (table_correct / table_total * 100) if table_total > 0 else 0.0
            table_stats.append({
                "has_table": has_table,
                "total": int(table_total),
                "correct": int(table_correct),
                "wrong": int(table_total - table_correct),
                "accuracy": round(table_accuracy, 2),
                "accuracy_percentage": f"{table_accuracy:.2f}%"
            })
        stats["by_has_table"] = table_stats
    
    if 'has_fomula' in results_df.columns:
        formula_stats = []
        for has_formula in sorted(results_df['has_fomula'].unique()):
            formula_df = results_df[results_df['has_fomula'] == has_formula]
            formula_total = len(formula_df)
            formula_correct = formula_df[is_correct_col].sum()
            formula_accuracy = (formula_correct / formula_total * 100) if formula_total > 0 else 0.0
            formula_stats.append({
                "has_formula": has_formula,
                "total": int(formula_total),
                "correct": int(formula_correct),
                "wrong": int(formula_total - formula_correct),
                "accuracy": round(formula_accuracy, 2),
                "accuracy_percentage": f"{formula_accuracy:.2f}%"
            })
        stats["by_has_formula"] = formula_stats
    
    return stats


def print_statistics(stats: Dict):
    """
    통계를 콘솔에 출력
    
    Args:
        stats: 통계 딕셔너리
    """
    print("\n" + "=" * 60)
    print(f"평가 결과 통계: {stats['model']}")
    print("=" * 60)
    print(f"전체 정확도: {stats['accuracy_percentage']} ({stats['correct_answers']}/{stats['total_questions']})")
    print(f"정답: {stats['correct_answers']}개")
    print(f"오답: {stats['wrong_answers']}개")
    
    if 'by_category' in stats:
        print("\n[카테고리별 정확도]")
        for cat in stats['by_category']:
            print(f"  {cat['category']}: {cat['accuracy_percentage']} ({cat['correct']}/{cat['total']})")
    
    if 'by_level' in stats:
        print("\n[레벨별 정확도]")
        for level in stats['by_level']:
            print(f"  {level['level']}: {level['accuracy_percentage']} ({level['correct']}/{level['total']})")
    
    if 'by_sub_category' in stats:
        print("\n[서브카테고리별 정확도 (상위 10개)]")
        for subcat in stats['by_sub_category'][:10]:
            print(f"  {subcat['sub_category']}: {subcat['accuracy_percentage']} ({subcat['correct']}/{subcat['total']})")
    
    if 'by_has_table' in stats:
        print("\n[테이블 포함 여부별 정확도]")
        for table in stats['by_has_table']:
            print(f"  테이블 {'포함' if table['has_table'] == 'Y' else '미포함'}: {table['accuracy_percentage']} ({table['correct']}/{table['total']})")
    
    if 'by_has_formula' in stats:
        print("\n[수식 포함 여부별 정확도]")
        for formula in stats['by_has_formula']:
            print(f"  수식 {'포함' if formula['has_formula'] == 'Y' else '미포함'}: {formula['accuracy_percentage']} ({formula['correct']}/{formula['total']})")
    
    print("=" * 60)


# =================================
# 모델 순서 정의 (사용자 지정 순서)
# =================================
MODEL_ORDER = [
    'Mistral-Small-3.2-24B-Instruct-2506',
    'Ministral-3-14B-Instruct-2512',
    'Ministral-3-8B-Instruct-2512',
    'Ministral-3-3B-Instruct-2512',
    'Qwen3-30B-A3B-Instruct-2507',
    'Qwen3-30B-A3B-Thinking-2507',
    'Qwen3-4B-Instruct-2507',
    'Qwen3-4B-Thinking-2507',
    'DeepSeek-R1-0528-Qwen3-8B',
    'kanana-2-30b-a3b-instruct',
    'kanana-1.5-15.7b-a3b-instruct',
    'kanana-1.5-8b-instruct-2505',
    'kanana-1.5-2.1b-instruct-2505',
    'gemma-3-27b-it',
    'gemma-3-12b-it',
    'gemma-3-4b-it',
    'gemma-3-1b-it',
    'gemma-3-270m-it',
    'Phi-4-reasoning',
    'Phi-4-mini-instruct',
    'Phi-4-mini-reasoning',
    'gpt-oss-120b',
    'gpt-oss-20b',
    'EXAONE-4.0-32B',
    'EXAONE-4.0-1.2B',
    'gpt-5.2',
    'gpt-5.2_reasoning',
    'gpt-5',
    'gpt-5-mini',
    'gpt-5-mini_reasoning',
    'gpt-5-nano',
    'gpt-5-nano_reasoning',
    'gpt-4.1',
    'claude-sonnet-4-5-20250929',
    'claude-haiku-4-5-20251001',
    'claude-opus-4-5-20251101',
    'gemini-3-pro-preview',
    'gemini-3-flash-preview',
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'mistral-medium-2508',
    # 'magistral-medium-2509'
    "grok-4-1-fast-reasoning",
    "grok-4-fast-reasoning",
]


# =================================
# 모델별 카테고리 점수 DataFrame 생성 함수
# =================================
def create_model_category_dataframe(all_stats: List[Dict]) -> pd.DataFrame:
    """
    모든 모델의 통계를 수집하여 모델별(row) 카테고리별(column) 점수 DataFrame 생성
    
    Args:
        all_stats: 모든 모델의 통계 딕셔너리 리스트
    
    Returns:
        모델별 카테고리별 정확도 DataFrame (Model 컬럼 + 각 카테고리별 정확도 컬럼)
    """
    # 모든 카테고리 수집
    all_categories = set()
    for stats in all_stats:
        if 'by_category' in stats:
            for cat_stat in stats['by_category']:
                all_categories.add(cat_stat['category'])
    
    # 카테고리 번역 및 정렬
    category_list = sorted(all_categories)
    
    # 모델별 카테고리별 정확도 딕셔너리 생성
    model_category_data = []
    
    for stats in all_stats:
        model_name = stats.get('model', 'Unknown')
        row_data = {'Model': model_name}
        
        # 각 카테고리에 대한 정확도 추가
        category_accuracies = {}
        if 'by_category' in stats:
            for cat_stat in stats['by_category']:
                category_accuracies[cat_stat['category']] = cat_stat['accuracy']
        
        # 모든 카테고리에 대해 정확도 설정 (없으면 NaN)
        for category in category_list:
            # 영문 컬럼명 사용
            eng_category = category_translation.get(category, category)
            row_data[eng_category] = category_accuracies.get(category, None)
        
        model_category_data.append(row_data)
    
    # DataFrame 생성
    df = pd.DataFrame(model_category_data)
    
    # Model 컬럼을 첫 번째로 이동
    cols = ['Model'] + [col for col in df.columns if col != 'Model']
    df = df[cols]
    
    # 지정된 모델 순서로 정렬
    # 모델 순서 딕셔너리 생성 (순서 인덱스)
    model_order_dict = {}
    for idx, model in enumerate(MODEL_ORDER):
        # 원본 모델명과 소문자 버전 모두 저장
        model_order_dict[model] = idx
        model_order_dict[model.lower()] = idx
    
    # 정렬을 위한 임시 컬럼 추가 (지정된 순서가 없으면 큰 값으로 처리)
    def get_sort_order(model_name):
        # 정확히 일치하는 경우
        if model_name in model_order_dict:
            return model_order_dict[model_name]
        
        # 소문자로 변환해서 매칭 시도
        model_lower = model_name.lower()
        if model_lower in model_order_dict:
            return model_order_dict[model_lower]
        
        # 부분 매칭 시도 (실제 모델명이 순서 리스트의 모델명의 시작 부분과 일치하는 경우)
        # 예: "claude-opus-4-5"는 "claude-opus-4-5-20251101"의 시작 부분과 매칭
        for idx, ordered_model in enumerate(MODEL_ORDER):
            ordered_model_lower = ordered_model.lower()
            # 순서 리스트의 모델명이 실제 모델명으로 시작하거나, 그 반대인 경우
            if model_lower.startswith(ordered_model_lower) or ordered_model_lower.startswith(model_lower):
                return idx
        
        # 특수 문자 정규화해서 매칭 시도 (언더스코어, 점을 하이픈으로 변환)
        def normalize_model_name(name):
            return name.lower().replace('_', '-')
        
        model_normalized = normalize_model_name(model_name)
        for idx, ordered_model in enumerate(MODEL_ORDER):
            ordered_model_normalized = normalize_model_name(ordered_model)
            if model_normalized.startswith(ordered_model_normalized) or ordered_model_normalized.startswith(model_normalized):
                return idx
        
        # 매칭되지 않으면 큰 값으로 처리 (맨 뒤로)
        return 999999
    
    df['_sort_order'] = df['Model'].map(get_sort_order)
    
    # 정렬 (지정된 순서 우선, 그 다음 모델명 알파벳 순)
    df = df.sort_values(['_sort_order', 'Model'])
    df = df.drop('_sort_order', axis=1)
    
    return df


# =================================
# CSV 파일 처리 함수
# =================================
# =================================
# LLM-as-Judge (옵션, --llm-judge)
# =================================
_judge_client = None  # 지연 초기화
_judge_lock = threading.Lock()


def _get_judge_client():
    """OpenRouter judge 클라이언트 (지연 초기화). `--llm-judge` 사용 시에만 필요."""
    global _judge_client
    if _judge_client is None:
        from dotenv import load_dotenv  # 옵션 import (--llm-judge 미사용 시 의존성 불필요)
        from openai import OpenAI
        repo_root = Path(__file__).resolve().parent.parent
        load_dotenv(repo_root / ".env")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"OPENROUTER_API_KEY not set. Add it to {repo_root}/.env or export it."
            )
        _judge_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/sklim84/KFinEval",
                "X-Title": "KFinEval Knowledge LLM Judge",
            },
        )
    return _judge_client


def _strip_think(raw: str) -> str:
    """`</think>` 뒤 텍스트만 반환 (구 plus_judge.extract_answer 동일)."""
    if not raw:
        return ""
    if "</think>" in raw:
        return raw.split("</think>")[-1].strip()
    return raw.strip()


def _judge_prompt(gold: str, stripped: str) -> str:
    """plus_judge 와 동일 문구."""
    return (
        f"""다음 객관식 문제의 정답과 모델 응답을 비교하여 정답 여부를 판별하세요.

정답: {gold}
모델 응답: {stripped}

모델 응답이 정답과 일치하면 "CORRECT", 일치하지 않으면 "INCORRECT"를 출력하세요.
모델 응답에서 정답 알파벳(A, B, C, D, E)이 명시적으로 언급되어 있는지 확인하세요.

판별 결과:"""
    )


def _judge_one(gold: str, raw_response: str, judge_model: str, max_retries: int = 3) -> str:
    """단일 행 판정 → "CORRECT" / "INCORRECT" / "ERROR"

    Reasoning 모델(예: openai/gpt-5.2)에 max_tokens=10 만 주면 reasoning 토큰만
    소비하고 content=None 으로 끝나는 함정이 있다. `2_2_eval_reasoning_openrouter.py`
    의 build_request_kwargs 와 같이 `extra_body.reasoning.effort=none` 을 명시해서
    reasoning 토큰을 끔 (judge 는 단순 "CORRECT"/"INCORRECT" 분류라 reasoning 불필요).
    """
    stripped = _strip_think(str(raw_response or ""))
    if not stripped:
        return "INCORRECT"
    client = _get_judge_client()
    last_exc = None
    for _ in range(max_retries):
        try:
            # max_tokens=32: OpenRouter→Azure(gpt-5.2) 는 max_output_tokens >= 16
            # 을 강제(아래는 400 BadRequest)하므로 안전 마진 포함. "CORRECT"/"INCORRECT"
            # 한 단어만 필요한 분류라 32 면 충분.
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": _judge_prompt(str(gold), stripped)}],
                max_tokens=32,
                temperature=0.0,
                extra_body={"reasoning": {"effort": "none"}},
            )
            result = (resp.choices[0].message.content or "").strip().upper()
            if "CORRECT" in result and "INCORRECT" not in result:
                return "CORRECT"
            if "INCORRECT" in result:
                return "INCORRECT"
            return result or "ERROR"
        except Exception as e:
            last_exc = e
    print(f"  [judge] 실패: {last_exc}")
    return "ERROR"


def llm_judge_dataframe(
    data: pd.DataFrame,
    model_name: str,
    judge_model: str,
    workers: int,
) -> pd.DataFrame:
    """`llm_judge` 컬럼 추가 → CORRECT/INCORRECT/ERROR. raw_response 필수."""
    if "raw_response" not in data.columns:
        raise RuntimeError(
            "--llm-judge: 'raw_response' 컬럼이 없습니다. --think 생성 결과여야 합니다."
        )
    records = data.to_dict(orient="records")

    def _w(r):
        return _judge_one(r["gold"], r.get("raw_response", ""), judge_model)

    results: List[str] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_w, r): i for i, r in enumerate(records)}
        pbar = tqdm(total=len(futs), desc=f"LLM judge ({model_name})")
        for fut in futs:
            try:
                results[futs[fut]] = fut.result()
            except Exception as e:
                results[futs[fut]] = "ERROR"
                print(f"  [judge] worker 실패: {e}")
            pbar.update(1)
        pbar.close()
    data = data.copy()
    data["llm_judge"] = results
    return data


def process_csv(
    input_csv_path: str,
    output_csv_path: str,
    model_name: str,
    llm_judge: bool = False,
    judge_model: str = "openai/gpt-5.2",
    judge_workers: int = 4,
) -> Optional[Dict]:
    """
    CSV 파일을 읽어서 정답 비교 및 통계 계산 수행

    Args:
        input_csv_path: 입력 CSV 파일 경로 (1_1_eval_knowledge_*.py 가 생성)
        output_csv_path: 출력 CSV 파일 경로 (정답 여부 컬럼이 추가된 파일)
        model_name: 모델 이름
        llm_judge: True 시 LLM judge 로 is_correct 판정 (raw_response 필요)
        judge_model: judge 모델 (OpenRouter id, 기본 openai/gpt-5.2)
        judge_workers: judge 호출 동시성
    """
    # 입력 CSV 파일 읽기
    try:
        data = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return None

    answer_col = f"answer"

    # 모델 답변 컬럼이 있는지 확인
    if answer_col not in data.columns:
        print(f"오류: '{answer_col}' 컬럼을 찾을 수 없습니다.")
        return None

    # 정답 여부 컬럼
    is_correct_col = f"is_correct"

    if llm_judge:
        # LLM judge 모드: raw_response 기반 판정
        data = llm_judge_dataframe(data, model_name, judge_model, judge_workers)
        data[is_correct_col] = data["llm_judge"] == "CORRECT"
    else:
        # 기본: 단순 문자열 매치
        data[is_correct_col] = data.apply(
            lambda row: check_answer(row[answer_col], row['gold']),
            axis=1
        )
    
    # 통계 계산
    stats = calculate_statistics(data, model_name)
    
    # 통계 출력
    print_statistics(stats)
    
    # 통계를 JSON 파일로 저장
    stats_output_path = output_csv_path.replace('.csv', '_stats.json')
    os.makedirs(os.path.dirname(stats_output_path) if os.path.dirname(stats_output_path) else ".", exist_ok=True)
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n통계 저장 완료: {stats_output_path}")
    
    # CSV 파일 저장 (UTF-8 BOM 추가하여 Excel에서 한글 깨짐 방지)
    os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)
    data.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"결과 저장 완료: {output_csv_path}")
    
    # 통계 딕셔너리 반환 (모델별 카테고리 점수 CSV 생성을 위해)
    return stats


# =================================
# 메인 실행 로직
# =================================
if __name__ == "__main__":
    # ==========================================
    # CLI 인자
    # ==========================================
    _cli = argparse.ArgumentParser(description="Knowledge stats / scoring")
    _cli.add_argument("--llm-judge", action="store_true",
                      help="raw_response 기반 LLM judge 로 is_correct 판정 "
                           "(--think 생성물에 사용 권장). 미사용 시 단순 문자열 매치.")
    _cli.add_argument("--judge-model", default="openai/gpt-5.2",
                      help="LLM judge 모델 (OpenRouter id, 기본 openai/gpt-5.2)")
    _cli.add_argument("--judge-workers", type=int, default=4,
                      help="LLM judge 동시성 (기본 4)")
    _cli.add_argument("--pattern", default=None,
                      help="입력 파일 글롭 패턴 override (기본 *_response.csv). 특정 모델만 처리할 때 사용")
    _args = _cli.parse_args()

    try:
        # ==========================================
        # 1단계: 처리할 결과 파일 설정
        #
        # 주의사항: 통계 계산 전에 결과 파일을 확인하세요.
        # - answer 컬럼이 비어있거나 형식이 잘못된 경우, answer_text 컬럼을 확인하여
        #   수작업으로 올바른 답변(A~E)을 answer 컬럼에 입력해주세요.
        # ==========================================
        # 결과 디렉토리
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results/1_fin_knowledge")

        # 처리할 파일 패턴
        input_pattern = _args.pattern or os.path.join(results_dir, "1_fin_knowledge_*_response.csv")
        input_files = glob.glob(input_pattern)

        if not input_files:
            print(f"처리할 파일을 찾을 수 없습니다: {input_pattern}")
            exit(1)

        if _args.llm_judge:
            print(f"[LLM-judge ON] judge={_args.judge_model} workers={_args.judge_workers}")
        
        print("=" * 60)
        print("평가 결과 통계 계산 모드")
        print("=" * 60)
        print(f"처리할 파일: {len(input_files)}개")
        for f in input_files:
            print(f"  - {os.path.basename(f)}")
        print("=" * 60)
        
        # ==========================================
        # 2단계: 각 파일 처리 및 통계 수집
        # ==========================================
        all_stats = []  # 모든 모델의 통계를 수집
        
        for input_file in input_files:
            print(f"\n{'='*60}")
            print(f"처리 중: {os.path.basename(input_file)}")
            print(f"{'='*60}")
            
            # 파일명에서 모델명 추출
            # 예: 1_fin_knowledge_gpt-oss-20b_response.csv -> gpt-oss-20b
            filename = os.path.basename(input_file)
            # 파일명 패턴: {benchmark}_{model_name}_response.csv
            parts = filename.replace('_response.csv', '').split('_')
            # benchmark 이름 제거 (예: 1_fin_knowledge)
            if len(parts) >= 3:
                model_name = '_'.join(parts[3:])  # 모델명 부분만 추출
            else:
                # 패턴이 맞지 않으면 파일명에서 직접 추출 시도
                model_name = filename.replace('1_fin_knowledge_', '').replace('_response.csv', '')
            
            # 출력 파일 경로 (같은 디렉토리에 저장)
            output_file = input_file  # 같은 파일에 덮어쓰기 (정답 여부 컬럼 추가)
            
            # CSV 처리 및 통계 계산 실행
            stats = process_csv(
                input_file, output_file, model_name,
                llm_judge=_args.llm_judge,
                judge_model=_args.judge_model,
                judge_workers=_args.judge_workers,
            )
            
            # 통계 수집 (카테고리별 통계가 있는 경우만)
            if stats and 'by_category' in stats:
                all_stats.append(stats)
            
            print(f"✓ {os.path.basename(input_file)} 처리 완료")
        
        print("\n" + "=" * 60)
        print("모든 파일 처리 완료!")
        print("=" * 60)
        
        # ==========================================
        # 3단계: 모델별 카테고리별 점수 CSV 생성
        # ==========================================
        if all_stats:
            print("\n" + "=" * 60)
            print("모델별 카테고리별 점수 CSV 생성")
            print("=" * 60)
            
            # 모델별 카테고리별 정확도 DataFrame 생성
            model_category_df = create_model_category_dataframe(all_stats)
            
            # CSV 저장
            output_csv_path = os.path.join(results_dir, "model_category_scores.csv")
            model_category_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"✓ 모델별 카테고리별 점수 CSV 저장 완료: {output_csv_path}")
            print(f"  - 모델 수: {len(model_category_df)}개")
            print(f"  - 카테고리 수: {len(model_category_df.columns) - 1}개 (Model 컬럼 제외)")
        else:
            print("\n경고: 카테고리별 통계가 없어 모델별 카테고리 점수 CSV를 생성할 수 없습니다.")
            
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        import traceback
        traceback.print_exc()



