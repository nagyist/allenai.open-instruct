import argparse

from datasets import load_dataset

from scripts.data.sft.utils import convert_sft_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process LIMA dataset and optionally upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload the dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face organization to upload to (if not provided, uploads to user's account)",
    )
    parser.add_argument(
        "--converted_dataset_name",
        type=str,
        default="lima_converted",
        help="Name of the converted dataset on Hugging Face Hub.",
    )
    parser.add_argument(
        "--local_save_dir",
        type=str,
        default=None,
        help="Local directory to save the dataset (if not provided, does not save locally)",
    )
    parser.add_argument(
        "--apply_keyword_filters",
        action="store_true",
        help=(
            "Apply keyword filters to the dataset. "
            "Currently filters out conversations with OpenAI, ChatGPT, BingChat, etc."
        ),
    )
    parser.add_argument(
        "--apply_empty_message_filters",
        action="store_true",
        help="Apply empty message filters to the dataset.",
    )
    args = parser.parse_args()

    def conversion_func(example):
        messages = []
        for i in range(len(example["conversations"])):
            if i % 2 == 0:
                messages.append({"role": "user", "content": example["conversations"][i]})
            else:
                messages.append({"role": "assistant", "content": example["conversations"][i]})
        if len(messages) % 2 != 0:
            print(f"Warning: an example has an odd number of messages: ({len(messages)}). We will cut the last message.")
            messages = messages[:-1]
        return {"messages": messages}

    readme_content = (
        "This is a converted version of the LIMA dataset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/lima.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "Please refer to the [original dataset](https://huggingface.co/datasets/GAIR/lima) "
        "for more information about this dataset and the license."
    )

    ds = load_dataset("GAIR/lima")["train"]
    convert_sft_dataset(
        ds=ds,
        hf_dataset_id=None,
        convert_fn=conversion_func,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name=args.converted_dataset_name,
        local_save_dir=args.local_save_dir,
        readme_content=readme_content,
    )
