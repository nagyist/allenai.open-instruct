# https://wandb.ai/ai2-llm/open_instruct_internal/runs/96221yio/overview
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_vllm_thread_ray_gtrl.py \
    --exp_name qwen2.5_7b_grpo_zero \
    --beta 0.0 \
    --local_mini_batch_size 32 \
    --number_samples_per_prompt 16 \
    --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning \
    --oe_eval_max_length 8192 \
    --local_rollout_batch_size 4 \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list ai2-adapt-dev/math_ground_truth_zs 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/math_ground_truth_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 4096 \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --stop_strings "</answer>" \
    --add_r1_style_format_reward True \
    --apply_verifiable_reward True \
    --chat_template_name r1_simple_chat_postpend_think \
    --non_stop_penalty False \
    --stop_token eos \
    --penalty_reward_value 0.0 \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --total_episodes 10000000 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --actor_num_gpus_per_node 8 4 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 4 \
    --lr_scheduler_type linear \
    --seed 1 \
    --num_evals 200 \
    --reward_model_multiplier 0.0 \
    --save_freq 40 \
    --try_launch_beaker_eval_jobs_on_weka \
    --gradient_checkpointing \
    --with_tracking