# 타겟 모델별 toxicity answer 생성
# local open-weight (vLLM)
# python3 -u eval/3_1_gen_toxicity_vllm.py
# proprietary / OpenRouter-hosted
# python3 -u eval/3_1_gen_toxicity_openrouter.py

# 타겟 모델별 toxicity answer 평가 (LLM-as-a-Judge, gpt-5.2 via OpenRouter)
python3 -u eval/3_2_eval_toxicity_openrouter.py --all-done
