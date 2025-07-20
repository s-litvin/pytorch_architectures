from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback

from datasets import load_dataset, DatasetDict
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# Загружаем файлы отдельно, указываем названия сплитов явно
data_files = {
    "train": "crystallography_train.txt",
    "validation": "crystallography_eval.txt"
}
dataset = load_dataset("text", data_files=data_files)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    #evaluation_strategy="step",
    overwrite_output_dir=True,
    num_train_epochs=12,
    per_device_train_batch_size=2,
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    logging_steps=5,
    #load_best_model_at_end=True,
    #metric_for_best_model="loss",
    #greater_is_better=False,
    no_cuda=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

