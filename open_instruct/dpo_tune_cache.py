# !/usr/bin/env python
# coding=utf-8
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DPO tuning script. Adapted from our finetuning script.
"""

# isort: off
import os

# We need to set NCCL_CUMEM_ENABLE=0 for performance reasons; see:
# https://github.com/vllm-project/vllm/issues/5723#issuecomment-2554389656
os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
try:
    import deepspeed

    # @vwxyzjn: when importing on CPU-only machines, we get the following error:
    # RuntimeError: 0 active drivers ([]). There should only be one.
    # so we need to catch the exception and do nothing
    # https://github.com/deepspeedai/DeepSpeed/issues/7028
except Exception:
    pass
# isort: on
import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from typing import Callable, List, Literal, Optional, Union

import datasets
import torch
import torch.utils
import torch.utils.data
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, get_scheduler
from transformers.training_args import _convert_str_dict

from open_instruct.dataset_transformation import (
    CHOSEN_INPUT_IDS_KEY,
    TOKENIZED_PREFERENCE_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.dpo_utils import (
    DataCollatorForSeq2SeqDPO,
    concatenated_forward,
    dpo_loss,
    separate_forward,
    simpo_loss,
    wpo_loss,
)
from open_instruct.model_utils import push_folder_to_hub, save_with_accelerate
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO
from open_instruct.utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_last_checkpoint_path,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)

logger = get_logger(__name__)


@dataclass
class FlatArguments:
    """
    Full arguments class for all fine-tuning jobs.
    """

    # Sometimes users will pass in a `str` repr of a dict in the CLI
    # We need to track what fields those can be. Each time a new arg
    # has a dict type, it must be added to this list.
    # Important: These should be typed with Optional[Union[dict,str,...]]
    _VALID_DICT_FIELDS = ["additional_model_arguments"]

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""
    add_seed_and_date_to_exp_name: bool = True
    """Append the seed and date to exp_name"""
    do_not_randomize_output_dir: bool = False
    """By default the output directory will be randomized"""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    dpo_use_paged_optimizer: bool = field(
        default=False,
        metadata={
            "help": "Use paged optimizer from bitsandbytes."
            " Not compatible with deepspeed (use deepspeed config instead)."
        },
    )
    dpo_beta: float = field(default=0.1, metadata={"help": "Beta parameter for DPO loss. Default is 0.1."})
    dpo_loss_type: str = field(
        default="dpo", metadata={"help": "Type of DPO loss to use. Options are 'dpo', 'dpo_norm', 'simpo', 'wpo'."}
    )
    dpo_gamma_beta_ratio: float = field(
        default=0.3, metadata={"help": "Gamma to beta ratio for SimPO loss. Default is 0.3. Not used for DPO loss."}
    )
    dpo_label_smoothing: float = field(
        default=0.0, metadata={"help": "Label smoothing for DPO/SimPO loss. Default is 0 (no smoothing)."}
    )
    use_flash_attn: bool = field(
        default=True, metadata={"help": "Whether to use flash attention in the model training"}
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    additional_model_arguments: Optional[Union[dict, str]] = field(
        default_factory=dict, metadata={"help": "A dictionary of additional model args used to construct the model."}
    )
    sync_each_batch: bool = False
    """Optionaly sync grads every batch when using grad accumulation. Can significantly reduce memory costs."""
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_mixer: Optional[dict] = field(
        default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."}
    )
    dataset_mixer_list: List[str] = field(
        default_factory=lambda: ["allenai/tulu-3-wildchat-reused-on-policy-8b", "1.0"]
    )
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["preference_tulu_tokenize_and_truncate_v1", "preference_tulu_filter_v1"]
    )
    """The list of transform functions to apply to the dataset."""
    dataset_target_columns: List[str] = field(default_factory=lambda: TOKENIZED_PREFERENCE_DATASET_KEYS)
    """The columns to use for the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: Optional[str] = None
    """The hash of the dataset configuration."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    dataset_mix_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory to save the mixed dataset to disk."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."},
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW optimizer."})
    logging_steps: Optional[int] = field(
        default=None, metadata={"help": "Log the training loss and learning rate every logging_steps steps."}
    )
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: float = field(default=16, metadata={"help": "The alpha parameter of lora."})
    lora_dropout: float = field(default=0.1, metadata={"help": "The dropout rate of lora modules."})
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        },
    )
    num_train_epochs: int = field(default=2, metadata={"help": "Total number of training epochs to perform."})
    output_dir: str = field(
        default="output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If True, will use LORA (low-rank parameter-efficient training) to train the model."},
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."},
    )
    use_8bit_optimizer: bool = field(
        default=False, metadata={"help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    reduce_loss: str = field(
        default="mean",
        metadata={
            "help": "How to reduce loss over tokens. Options are 'mean' or 'sum'."
            "Using 'sum' can improve chat model performance."
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "If the training should continue from a checkpoint folder."}
    )
    report_to: Union[str, List[str]] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    save_to_hub: Optional[str] = field(
        default=None, metadata={"help": "Save the model to the Hub under this name. E.g allenai/your-model"}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Turn on gradient checkpointing. Saves memory but slows training."}
    )
    use_liger_kernel: bool = field(default=False, metadata={"help": "Whether to use LigerKernel for training."})
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization and dataset shuffling."})
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."  # noqa
        },
    )
    keep_last_n_checkpoints: int = field(
        default=3, metadata={"help": "How many checkpoints to keep in the output directory. -1 for all."}
    )
    fused_optimizer: bool = field(default=True, metadata={"help": "Whether to use fused AdamW or not."})
    load_balancing_loss: bool = field(
        default=False, metadata={"help": "Whether to include a load balancing loss (for OLMoE) or not."}
    )
    load_balancing_weight: float = field(
        default=0.001, metadata={"help": "Weight for load balancing loss if applicable."}
    )
    concatenated_forward: bool = True
    """Whether to concatenate chosen and rejected for DPO training; True is good but you can set to False for saving memory."""

    # Experiment tracking
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    try_launch_beaker_eval_jobs: bool = True
    """Whether to launch beaker evaluation jobs after training"""
    hf_metadata_dataset: Optional[str] = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""

    packing: bool = field(
        default=False,
        metadata={"help": "Whether to use packing/padding-free collation via DataCollatorWithFlatteningDPO"},
    )

    # Ai2 specific settings
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: Optional[str] = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: Optional[List[str]] = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""

    def __post_init__(self):
        if self.reduce_loss not in ["mean", "sum"]:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")
        if self.dataset_name is None and self.dataset_mixer is None and self.dataset_mixer_list is None:
            raise ValueError("Need either a dataset name, dataset mixer, or a training file.")
        if (
            (self.dataset_name is not None and (self.dataset_mixer is not None or self.dataset_mixer_list is not None))
            or (self.dataset_name is not None)
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.try_launch_beaker_eval_jobs and not self.push_to_hub:
            raise ValueError("Cannot launch Beaker evaluation jobs without pushing to the Hub.")

        # Parse in args that could be `dict` sent in from the CLI as a string
        for dict_feld in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, dict_feld)
            # We only want to do this if the str starts with a bracket to indicate a `dict`
            # else its likely a filename if supported
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                # Convert str values to types if applicable
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, dict_feld, loaded_dict)


def get_cache_ref_logprobs(
    model: torch.nn.Module,
    active_dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    average_log_prob: bool,
    last_checkpoint_path: Optional[str],
    resume_step: int,
    epoch_range: range,
    forward_fn: Callable,
):
    epoch_cached_reference_chosen_logps = []
    epoch_cached_reference_rejected_logps = []
    for epoch in epoch_range:
        active_dataloader.set_epoch(epoch)
        if last_checkpoint_path and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(active_dataloader, resume_step)
        cached_reference_chosen_logps = []
        cached_reference_rejected_logps = []
        with torch.no_grad():
            for batch in tqdm(
                active_dataloader,
                disable=not accelerator.is_local_main_process,
                desc=f"Generating reference cache (epoch {epoch})",
            ):
                if args.use_lora:
                    with accelerator.unwrap_model(model).disable_adapter():
                        reference_chosen_logps, reference_rejected_logps, _ = forward_fn(
                            model, batch, average_log_prob=average_log_prob
                        )
                else:
                    reference_chosen_logps, reference_rejected_logps, _ = forward_fn(
                        model, batch, average_log_prob=average_log_prob
                    )
                cached_reference_chosen_logps.append(reference_chosen_logps.cpu())
                cached_reference_rejected_logps.append(reference_rejected_logps.cpu())
        epoch_cached_reference_chosen_logps.append(cached_reference_chosen_logps)
        epoch_cached_reference_rejected_logps.append(cached_reference_rejected_logps)
    return epoch_cached_reference_chosen_logps, epoch_cached_reference_rejected_logps


def main(args: FlatArguments, tc: TokenizerConfig):
    # ------------------------------------------------------------
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps, sync_each_batch=args.sync_each_batch
        ),
    )

    # ------------------------------------------------------------
    # Setup tokenizer
    tc.tokenizer_revision = args.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if tc.tokenizer_revision != args.model_revision and tc.tokenizer_name_or_path != args.model_name_or_path:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{args.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{args.model_name_or_path=}`."""
        logger.warning(warning)
    tokenizer = tc.tokenizer

    # ------------------------------------------------------------
    # Set up runtime variables
    if args.add_seed_and_date_to_exp_name:
        args.exp_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        args.exp_name = args.exp_name
    if not args.do_not_randomize_output_dir:
        args.output_dir = os.path.join(args.output_dir, args.exp_name)
    logger.info("using the output directory: %s", args.output_dir)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
    if args.push_to_hub and accelerator.is_main_process:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.exp_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
        if is_beaker_job():
            beaker_config = maybe_get_beaker_config()

    # ------------------------------------------------------------
    # Initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

        # (Optional) Ai2 internal tracking
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
        if accelerator.is_main_process and is_beaker_job():
            experiment_config.update(vars(beaker_config))
        experiment_config.update(vars(tc))
        accelerator.init_trackers(
            args.wandb_project_name,
            experiment_config,
            init_kwargs={
                "wandb": {
                    "name": args.exp_name,
                    "entity": args.wandb_entity,
                    "tags": [args.exp_name] + get_wandb_tags(),
                }
            },
        )
        wandb_tracker = accelerator.get_tracker("wandb")

    if accelerator.is_main_process:
        pprint([args, tc])

    init_gpu_memory = None
    if torch.cuda.is_available():
        init_gpu_memory = torch.cuda.mem_get_info()[0]

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.dataset_mixer is not None:
        args.dataset_mixer_list = [item for pair in args.dataset_mixer.items() for item in pair]
    with accelerator.main_process_first():
        transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]
        train_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.dataset_target_columns,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )
        train_dataset = train_dataset.shuffle(seed=args.seed)
        train_dataset.set_format(type="pt")
    if accelerator.is_main_process:
        visualize_token(train_dataset[0][CHOSEN_INPUT_IDS_KEY], tokenizer)

    if args.cache_dataset_only:
        return

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
            **args.additional_model_arguments,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
            **args.additional_model_arguments,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    def load_model():
        if args.model_name_or_path:
            if args.use_qlora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                device_index = accelerator.local_process_index
                device_map = {"": device_index}  # force data-parallel training.
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    revision=args.model_revision,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    trust_remote_code=tc.trust_remote_code,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                )
            elif args.use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                logger.info("Attempting to apply liger-kernel.")

                # Supported models: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py#L948
                model = AutoLigerKernelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    revision=args.model_revision,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    trust_remote_code=tc.trust_remote_code,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    use_flash_attention_2=True if args.use_flash_attn else False,
                    # liger-kernel specific args
                    fused_linear_cross_entropy=False,  # don't fuse the linear layer with CE loss, since we want logits
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    revision=args.model_revision,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    trust_remote_code=tc.trust_remote_code,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config)
        return model

    model = load_model()
    print("=============model loaded")
    print_gpu_stats(init_gpu_memory)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # debugging tool for fewer samples
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        logger.info(f"Limiting training samples to {max_train_samples} from {len(train_dataset)}.")
        train_dataset = train_dataset.select(range(max_train_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.packing:
        accelerator.print("Using packing/padding-free collation")
        collate_fn = TensorDataCollatorWithFlatteningDPO(return_position_ids=True, return_flash_attn_kwargs=True)
    else:
        collate_fn = DataCollatorForSeq2SeqDPO(tokenizer=tokenizer, model=model, padding="longest")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.use_qlora or args.dpo_use_paged_optimizer:
        from bitsandbytes.optim import AdamW

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.fused_optimizer)
    print("=============optimizer loaded")
    print_gpu_stats(init_gpu_memory)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the
    # entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of
    # updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    print("=============accelerate prepared")
    print_gpu_stats(init_gpu_memory)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(args)
    resume_step = None
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    print(f"Starting from epoch {starting_epoch} and step {completed_steps}.")

    print("=============before cache logprobs")
    print_gpu_stats(init_gpu_memory)

    # Cache the logprobs
    average_log_prob_loss_types = ["simpo", "dpo_norm"]
    average_log_prob = args.dpo_loss_type in average_log_prob_loss_types
    forward_fn = concatenated_forward if args.concatenated_forward else separate_forward
    if args.packing:
        if not args.concatenated_forward:
            raise NotImplementedError("seperate forward not implemented for packing/padding-free")
        forward_fn = partial(forward_fn, packing=True)
    if args.dpo_loss_type == "dpo" or args.dpo_loss_type == "dpo_norm":
        epoch_cached_reference_chosen_logps, epoch_cached_reference_rejected_logps = get_cache_ref_logprobs(
            model,
            train_dataloader,
            accelerator,
            average_log_prob,
            last_checkpoint_path,
            resume_step,
            range(starting_epoch, args.num_train_epochs),
            forward_fn,
        )
        print("=============after cache logprobs")
        print_gpu_stats(init_gpu_memory)
        torch.cuda.empty_cache()  # clear cache

    print("=============after cache logprobs; clear cache")
    print_gpu_stats(init_gpu_memory)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    local_metrics = torch.zeros((20), device=accelerator.device)
    episode = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        if last_checkpoint_path and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        # we need to average the log probs for simpo loss
        for step, batch in enumerate(active_dataloader):
            episode += len(batch["chosen_input_ids"]) * accelerator.num_processes
            # dpo forward pass & loss
            with accelerator.accumulate(model):
                policy_chosen_logps, policy_rejected_logps, aux_loss = forward_fn(
                    model, batch, average_log_prob=average_log_prob, output_router_logits=args.load_balancing_loss
                )  # `aux_loss` is only used when `args.load_balancing_loss = True`
                if args.dpo_loss_type == "dpo" or args.dpo_loss_type == "dpo_norm":
                    p_device = policy_chosen_logps.device
                    reference_chosen_logps = epoch_cached_reference_chosen_logps[epoch][step].to(p_device)
                    reference_rejected_logps = epoch_cached_reference_rejected_logps[epoch][step].to(p_device)
                    losses, _, _ = dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        reference_chosen_logps,
                        reference_rejected_logps,
                        beta=args.dpo_beta,
                        label_smoothing=args.dpo_label_smoothing,
                    )
                elif args.dpo_loss_type == "simpo":
                    losses, _, _ = simpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        beta=args.dpo_beta,
                        gamma_beta_ratio=args.dpo_gamma_beta_ratio,
                        label_smoothing=args.dpo_label_smoothing,
                    )
                elif args.dpo_loss_type == "wpo":
                    losses, _, _ = wpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        reference_chosen_logps,
                        reference_rejected_logps,
                        beta=args.dpo_beta,
                        label_smoothing=args.dpo_label_smoothing,
                        chosen_loss_mask=batch["chosen_labels"] != -100,
                        rejected_loss_mask=batch["rejected_labels"] != -100,
                    )
                else:
                    raise ValueError(f"Invalid dpo loss type {args.dpo_loss_type}.")
                # TODO: metric logging
                loss = losses.mean()
                if args.load_balancing_loss:
                    weighted_aux_loss = args.load_balancing_weight * aux_loss
                    loss += weighted_aux_loss
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # We keep track of the loss at each logged step
                with torch.no_grad():
                    local_metrics[0] += loss
                    if args.dpo_loss_type == "dpo" or args.dpo_loss_type == "dpo_norm":
                        chosen_rewards = (args.dpo_beta * (policy_chosen_logps - reference_chosen_logps)).mean()
                        rejected_rewards = (args.dpo_beta * (policy_rejected_logps - reference_rejected_logps)).mean()
                        average_rewards = (chosen_rewards + rejected_rewards) / 2
                        accuracy = (chosen_rewards > rejected_rewards).float().mean()
                        margin = (chosen_rewards - rejected_rewards).mean()
                        local_metrics[1] += chosen_rewards
                        local_metrics[2] += rejected_rewards
                        local_metrics[3] += average_rewards
                        local_metrics[4] += accuracy
                        local_metrics[5] += margin
                    local_metrics[6] += policy_chosen_logps.mean()
                    local_metrics[7] += policy_rejected_logps.mean()
                    if args.load_balancing_loss:
                        local_metrics[19] += weighted_aux_loss

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    # single all reduce to save time, avoiding per metric all reduce
                    global_metrics = accelerator.reduce(local_metrics, reduction="mean")
                    global_metrics /= args.gradient_accumulation_steps * args.logging_steps
                    global_metrics = global_metrics.tolist()
                    metrics_to_log = {
                        "training_step": completed_steps,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": episode / len(train_dataset),
                        "train_loss": global_metrics[0],
                        "logps/chosen": global_metrics[6],
                        "logps/rejected": global_metrics[7],
                    }
                    if args.dpo_loss_type == "dpo" or args.dpo_loss_type == "dpo_norm":
                        metrics_to_log.update(
                            {
                                "rewards/chosen": global_metrics[1],
                                "rewards/rejected": global_metrics[2],
                                "rewards/average": global_metrics[3],
                                "rewards/accuracy": global_metrics[4],
                                "rewards/margin": global_metrics[5],
                            }
                        )
                    logger_str = (
                        f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {global_metrics[0]}"
                    )
                    if args.load_balancing_loss:
                        logger_str += f" Aux Loss: {global_metrics[19]}"
                        metrics_to_log["aux_loss"] = global_metrics[19]
                    logger.info(logger_str)
                    if args.with_tracking:
                        accelerator.log(metrics_to_log, step=completed_steps)
                    # Reset the local metrics
                    local_metrics.zero_()

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                        # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
                        with open(
                            os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w"
                        ) as f:
                            f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.
                        if accelerator.is_local_main_process:
                            clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
                        accelerator.wait_for_everyone()

                if completed_steps >= args.max_train_steps:
                    break

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
            with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w") as f:
                f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.
            if accelerator.is_local_main_process:
                clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
            accelerator.wait_for_everyone()

    if args.output_dir is not None:
        save_with_accelerate(
            accelerator, model, tokenizer, args.output_dir, args.use_lora, chat_template_name=tc.chat_template_name
        )

    # remove all checkpoints to save space
    if accelerator.is_local_main_process:
        clean_last_n_checkpoints(args.output_dir, keep_last_n_checkpoints=0)

    if (
        args.try_auto_save_to_beaker
        and accelerator.is_main_process
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if is_beaker_job() and accelerator.is_main_process and args.try_launch_beaker_eval_jobs:
        launch_ai2_evals_on_weka(
            path=args.output_dir,
            leaderboard_name=args.hf_repo_revision,
            oe_eval_max_length=args.oe_eval_max_length,
            wandb_url=wandb_tracker.run.get_url(),
            oe_eval_tasks=args.oe_eval_tasks,
            gs_bucket_path=args.gs_bucket_path,
        )
    if args.push_to_hub:
        push_folder_to_hub(accelerator, args.output_dir, args.hf_repo_id, args.hf_repo_revision)
    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


def print_gpu_stats(init_gpu_memory: Optional[int]):
    if torch.cuda.is_available():
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = init_gpu_memory - free_gpu_memory
        print(f"Peak memory usage: {peak_memory / 1024**3:.2f} GB")
        print(f"Total memory usage: {total_gpu_memory / 1024**3:.2f} GB")
        print(f"Free memory: {free_gpu_memory / 1024**3:.2f} GB")


if __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
