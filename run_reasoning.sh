# 타겟 모델별 reasoning answer 생성
# local open-weight (vLLM)
# python3 -u eval/2_1_gen_reasoning_vllm.py
# proprietary / OpenRouter-hosted
# python3 -u eval/2_1_gen_reasoning_openrouter.py --model openai/gpt-5.2

# 타겟 모델별 reasoning answer 평가 (LLM-as-a-Judge, gpt-5.2 via OpenRouter)
# 모델별 --target-model 지정; 또는 --all-done 으로 manifest done 일괄 채점
python3 -u eval/2_2_eval_reasoning_openrouter.py --target-model <model> --judge-model openai/gpt-5.2
