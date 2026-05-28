"""
추론 응답 평가 결과 통계 계산 스크립트

이 스크립트는 2_2_eval_reasoning_openrouter.py에서 생성된 CSV 파일을 읽어서
평가 점수 기반 통계를 계산하고 결과를 저장합니다.
"""

import pandas as pd
import os
import json
import glob
from typing import Dict, List, Optional


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
    # 평가 항목 리스트
    eval_metrics = ['coherence', 'consistency', 'accuracy', 'completeness', 'reasoning', 'overall_quality']
    
    # 전체 통계
    total = len(results_df)
    
    # 각 평가 항목별 통계
    metric_stats = {}
    for metric in eval_metrics:
        if metric in results_df.columns:
            metric_values = results_df[metric].dropna()
            if len(metric_values) > 0:
                metric_stats[metric] = {
                    "mean": round(metric_values.mean(), 2),
                    "median": round(metric_values.median(), 2),
                    "std": round(metric_values.std(), 2),
                    "min": int(metric_values.min()),
                    "max": int(metric_values.max()),
                    "count": int(len(metric_values)),
                    "missing": int(total - len(metric_values))
                }
            else:
                metric_stats[metric] = {
                    "mean": None,
                    "median": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "count": 0,
                    "missing": total
                }
    
    # 전체 평균 점수 계산
    all_scores = []
    for metric in eval_metrics:
        if metric in results_df.columns:
            all_scores.extend(results_df[metric].dropna().tolist())
    
    overall_mean = round(sum(all_scores) / len(all_scores), 2) if all_scores else None
    
    stats = {
        "model": model_name,
        "total_questions": int(total),
        "overall_mean_score": overall_mean,
        "metrics": metric_stats
    }
    
    # 카테고리별 통계
    if 'category' in results_df.columns:
        category_stats = []
        for category in results_df['category'].unique():
            cat_df = results_df[results_df['category'] == category]
            cat_total = len(cat_df)
            
            cat_metric_stats = {}
            for metric in eval_metrics:
                if metric in cat_df.columns:
                    metric_values = cat_df[metric].dropna()
                    if len(metric_values) > 0:
                        cat_metric_stats[metric] = {
                            "mean": round(metric_values.mean(), 2),
                            "count": int(len(metric_values))
                        }
                    else:
                        cat_metric_stats[metric] = {
                            "mean": None,
                            "count": 0
                        }
            
            # 카테고리별 전체 평균
            cat_all_scores = []
            for metric in eval_metrics:
                if metric in cat_df.columns:
                    cat_all_scores.extend(cat_df[metric].dropna().tolist())
            cat_mean = round(sum(cat_all_scores) / len(cat_all_scores), 2) if cat_all_scores else None
            
            category_stats.append({
                "category": category,
                "total": int(cat_total),
                "mean_score": cat_mean,
                "metrics": cat_metric_stats
            })
        # 평균 점수 순으로 정렬
        category_stats.sort(key=lambda x: x['mean_score'] if x['mean_score'] is not None else 0, reverse=True)
        stats["by_category"] = category_stats
    
    # 소스별 통계
    if 'source' in results_df.columns:
        source_stats = []
        for source in results_df['source'].unique():
            source_df = results_df[results_df['source'] == source]
            source_total = len(source_df)
            
            source_metric_stats = {}
            for metric in eval_metrics:
                if metric in source_df.columns:
                    metric_values = source_df[metric].dropna()
                    if len(metric_values) > 0:
                        source_metric_stats[metric] = {
                            "mean": round(metric_values.mean(), 2),
                            "count": int(len(metric_values))
                        }
                    else:
                        source_metric_stats[metric] = {
                            "mean": None,
                            "count": 0
                        }
            
            # 소스별 전체 평균
            source_all_scores = []
            for metric in eval_metrics:
                if metric in source_df.columns:
                    source_all_scores.extend(source_df[metric].dropna().tolist())
            source_mean = round(sum(source_all_scores) / len(source_all_scores), 2) if source_all_scores else None
            
            source_stats.append({
                "source": source,
                "total": int(source_total),
                "mean_score": source_mean,
                "metrics": source_metric_stats
            })
        # 평균 점수 순으로 정렬
        source_stats.sort(key=lambda x: x['mean_score'] if x['mean_score'] is not None else 0, reverse=True)
        stats["by_source"] = source_stats
    
    # 점수 분포 (1-10점 구간별)
    score_distribution = {}
    for metric in eval_metrics:
        if metric in results_df.columns:
            metric_values = results_df[metric].dropna()
            if len(metric_values) > 0:
                distribution = {}
                for score_range in [(1, 3), (4, 6), (7, 8), (9, 10)]:
                    count = len(metric_values[(metric_values >= score_range[0]) & (metric_values <= score_range[1])])
                    distribution[f"{score_range[0]}-{score_range[1]}"] = {
                        "count": int(count),
                        "percentage": round(count / len(metric_values) * 100, 2) if len(metric_values) > 0 else 0
                    }
                score_distribution[metric] = distribution
    stats["score_distribution"] = score_distribution
    
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
    print(f"전체 질문 수: {stats['total_questions']}개")
    if stats['overall_mean_score'] is not None:
        print(f"전체 평균 점수: {stats['overall_mean_score']:.2f}/10.00")
    
    print("\n[평가 항목별 통계]")
    if 'metrics' in stats:
        for metric, metric_stat in stats['metrics'].items():
            if metric_stat['mean'] is not None:
                print(f"  {metric}:")
                print(f"    평균: {metric_stat['mean']:.2f}")
                print(f"    중앙값: {metric_stat['median']:.2f}")
                print(f"    표준편차: {metric_stat['std']:.2f}")
                print(f"    범위: {metric_stat['min']} ~ {metric_stat['max']}")
                print(f"    유효 데이터: {metric_stat['count']}개")
                if metric_stat['missing'] > 0:
                    print(f"    누락: {metric_stat['missing']}개")
    
    if 'by_category' in stats:
        print("\n[카테고리별 평균 점수]")
        for cat in stats['by_category']:
            if cat['mean_score'] is not None:
                print(f"  {cat['category']}: {cat['mean_score']:.2f}/10.00 ({cat['total']}개)")
    
    if 'by_source' in stats:
        print("\n[소스별 평균 점수]")
        for source in stats['by_source']:
            if source['mean_score'] is not None:
                print(f"  {source['source']}: {source['mean_score']:.2f}/10.00 ({source['total']}개)")
    
    if 'score_distribution' in stats:
        print("\n[점수 분포]")
        for metric, distribution in stats['score_distribution'].items():
            print(f"  {metric}:")
            for score_range, dist in distribution.items():
                print(f"    {score_range}점: {dist['count']}개 ({dist['percentage']:.2f}%)")
    
    print("=" * 60)


# =================================
# CSV 파일 처리 함수
# =================================
def process_csv(
    input_csv_path: str,
    output_stats_path: str,
    model_name: str
) -> None:
    """
    CSV 파일을 읽어서 통계 계산 수행
    
    Args:
        input_csv_path: 입력 CSV 파일 경로 (2_2_eval_reasoning_openrouter.py에서 생성된 파일)
        output_stats_path: 출력 통계 JSON 파일 경로
        model_name: 모델 이름
    """
    # 입력 CSV 파일 읽기
    try:
        data = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return
    
    print(f"CSV 파일 로드 완료: {data.shape[0]}개 행, {data.shape[1]}개 컬럼")
    print(f"컬럼: {list(data.columns)}")
    
    # 통계 계산
    stats = calculate_statistics(data, model_name)
    
    # 통계 출력
    print_statistics(stats)
    
    # 통계를 JSON 파일로 저장
    os.makedirs(os.path.dirname(output_stats_path) if os.path.dirname(output_stats_path) else ".", exist_ok=True)
    with open(output_stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n통계 저장 완료: {output_stats_path}")


# =================================
# 메인 실행 로직
# =================================
if __name__ == "__main__":
    try:
        # ==========================================
        # 1단계: 처리할 결과 파일 설정
        # ==========================================
        # 결과 디렉토리
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results/2_fin_reasoning")
        
        # 처리할 파일 패턴 (예: 2_fin_reasoning_*_eval.csv)
        input_pattern = os.path.join(results_dir, "2_fin_reasoning_*_eval.csv")
        input_files = glob.glob(input_pattern)
        
        if not input_files:
            print(f"처리할 파일을 찾을 수 없습니다: {input_pattern}")
            exit(1)
        
        print("=" * 60)
        print("추론 평가 결과 통계 계산 모드")
        print("=" * 60)
        print(f"처리할 파일: {len(input_files)}개")
        for f in input_files:
            print(f"  - {os.path.basename(f)}")
        print("=" * 60)
        
        # ==========================================
        # 2단계: 각 파일 처리
        # ==========================================
        for input_file in input_files:
            print(f"\n{'='*60}")
            print(f"처리 중: {os.path.basename(input_file)}")
            print(f"{'='*60}")
            
            # 파일명에서 모델명 추출
            # 예: 2_fin_reasoning_Ministral-3-14B-Instruct-2512_eval.csv -> Ministral-3-14B-Instruct-2512
            filename = os.path.basename(input_file)
            # 파일명 패턴: 2_fin_reasoning_{model_name}_eval.csv
            model_name = filename.replace('2_fin_reasoning_', '').replace('_eval.csv', '')
            
            # 출력 파일 경로 (통계 JSON 파일)
            output_stats_file = input_file.replace('_eval.csv', '_eval_stats.json')
            
            # CSV 처리 및 통계 계산 실행
            process_csv(input_file, output_stats_file, model_name)
            
            print(f"✓ {os.path.basename(input_file)} 처리 완료")
        
        print("\n" + "=" * 60)
        print("모든 파일 처리 완료!")
        print("=" * 60)
            
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

