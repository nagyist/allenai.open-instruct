# full example
launch_server_mason.sh  # to launch the server - in massive_ds repo.
python mason.py \
    --cluster ai2/augusta-google-1 --image hamishivi/0704_testing_rl_rag \
    --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --preemptible \
    --num_nodes 2 \
    --max_retries 0 \
    --secret YOUCOM_API_KEY=hamishi_youcom_api_key \
    --secret S2_API_KEY=hamishi_s2_api_key \
    --env MASSIVE_DS_URL='http://ceres-cs-aus-441.reviz.ai2.in:56843/search' \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_vllm_thread_ray_gtrl.py \
    --exp_name 0704_testing_rl_rag_no_query_fixed_synth_datastore \
    --beta 0.0 \
    --local_mini_batch_size 64 \
    --number_samples_per_prompt 16 \
    --output_dir /output \
    --num_epochs 1 \
    --local_rollout_batch_size 16 \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list hamishivi/SimpleQA-RLVR 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/SimpleQA-RLVR 32 \
    --dataset_mixer_eval_list_splits test \
    --max_token_length 8192 \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --model_name_or_path ai2-adapt-dev/tulu_3_long_finetune_qwen_7b_reg \
    --non_stop_penalty False \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --total_episodes 70000 \
    --penalty_reward_value 0.0 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --enable_prefix_caching \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --num_evals 100 \
    --save_freq 2000 \
    --reward_model_multiplier 0.0 \
    --no_try_launch_beaker_eval_jobs \
    --try_launch_beaker_eval_jobs_on_weka False \
    --gradient_checkpointing \
    --with_tracking \
    --mask_snippet_loss true \
    --use_search_actor true \
    --stop_strings "'</query>'" "'</answer>'" \
