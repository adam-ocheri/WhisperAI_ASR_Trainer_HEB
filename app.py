
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
from dotenv import load_dotenv
# from peft.utils.other import prepare_model_for_int8_training
# from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

# def print_gpu_utilization():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")


# def print_summary(result):
#     print(f"Time: {result.metrics['train_runtime']:.2f}")
#     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
#     print_gpu_utilization()

load_dotenv()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=10"
hf_token = os.getenv("HF_TOKEN")

print("Device name: ", torch.cuda.get_device_properties('cuda').name)
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
print(f'torch version: {torch.version}')

# Disable flash attention
torch.backends.cuda.enable_flash_sdp(False)

# Enable an alternative attention mechanism
# torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

print("PP - FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())

def init():
    cache_dir = "./Model/"
    base_model_name = "openai/whisper-small"
    # base_model_name = "openai/whisper-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE: ", device)

    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name, cache_dir=cache_dir + base_model_name + "_model"
    ).to(device)
    # model = prepare_model_for_int8_training(model)
    # config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()

    processor = WhisperProcessor.from_pretrained(
        base_model_name, cache_dir=cache_dir + base_model_name + "_processor"
    )

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        base_model_name, cache_dir=cache_dir + base_model_name + "_featureXtractor"
    )

    tokenizer = WhisperTokenizer.from_pretrained(
        base_model_name, cache_dir=cache_dir + base_model_name + "_tokenizer", language="Hebrew", task="transcribe"
    )
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.config.use_reentrant = False
    print("Model loaded successfully! \n")

    
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    dataset = DatasetDict()

    dataset["train"] = load_dataset("imvladikon/hebrew_speech_kan", "default", split="train", token=False)
    dataset["test"] = load_dataset("imvladikon/hebrew_speech_kan", "default", split="validation", token=False)

    print("Printing Dataset: ")
    print(dataset)

    # dataset = dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

    input_str = dataset["train"][0]["sentence"]
    labels = tokenizer(input_str).input_ids
    decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

    print(f"Input:                 {input_str}")
    print(f"Decoded w/ special:    {decoded_with_special}")
    print(f"Decoded w/out special: {decoded_str}")
    print(f"Are equal:             {input_str == decoded_str}")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print("dataset['train'][0]", dataset["train"][0])

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
    dataset["train"] = dataset["train"].shuffle(
    writer_batch_size=10,
    seed=0,
)
    # train_batch_size = 16
    # gradient_steps = 1
    training_args = Seq2SeqTrainingArguments(
        output_dir="OverloadedOperator/tokomni-whisper-v2",  
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,      # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=4000,
        evaluation_strategy="steps",
        # num_train_epochs=3,
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        gradient_checkpointing=True,
        fp16=True,
        per_device_eval_batch_size=2,
        # predict_with_generate=False,
        predict_with_generate=True,
        # generation_max_length=225,
        generation_max_length=128,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
    )

    training_args = training_args.set_push_to_hub(
        model_id="OverloadedOperator/tokomni-whisper-v2", # Replace with your actual username and repo name
        strategy="end", # Push the model at the end of training
        token=hf_token, # Optional: Specify a token if you want to use a different one than the default
        private_repo=False, # Optional: Set to True if you want the repo to be private
        always_push=True, # Optional: Set to True to always push, even if the previous push is not finished
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("STARTING TRAINER...")
    trainer.train()

    # Save Model - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # processor.feature_extractor.save_pretrained("OverloadedOperator/tokomni-whisper-v2", push_to_hub=True)
    kwargs = {
        "dataset_tags": "imvladikon/hebrew_speech_kan",
        "dataset": "KAN Hebrew Speech",  # a 'pretty' name for the training dataset
        "dataset_args": "config: default, split: test",
        "language": "he",
        "model_name": "TK_Whisper_ASR",  # a 'pretty' name for your model
        "finetuned_from": base_model_name,
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }
    trainer.push_to_hub(**kwargs)

    tokenizer.save_pretrained("OverloadedOperator/tokomni-whisper-v2", push_to_hub=True)
    processor.save_pretrained("OverloadedOperator/tokomni-whisper-v2", push_to_hub=True)
    trainer.push_to_hub(**kwargs)
    
if __name__ == "__main__":
    init()
