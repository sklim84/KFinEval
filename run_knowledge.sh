# 타겟 모델별 knowledge answer 생성
# local open-weight (vLLM)
# python3 -u eval/1_1_eval_knowledge_vllm.py
# proprietary / OpenRouter-hosted
# python3 -u eval/1_1_eval_knowledge_openrouter.py --model openai/gpt-5.2

# 정답 비교 + _response_stats.json 생성 (is_correct 컬럼 추가)
python3 -u eval/1_2_stats_eval_knowledge.py
