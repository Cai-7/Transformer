"""
Qwen2.5-0.5B LoRA 微调 —— OPUS-100 英译法
=================================================
- 数据集：OPUS-100 en-fr 子集（~100 万句对）
- 模型：Qwen2.5-0.5B（decoder-only）
- 微调方式：LoRA 低秩适配 + 仅目标语言部分计算 loss
"""

import os
# 必须在 import datasets/transformers 之前设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import os

set_seed(42)

# ============================================================
# 全局配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./qwen_opus_enfr_lora"
MAX_LENGTH = 256
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
LOGGING_STEPS = 20
SAVE_STEPS = 200
MAX_TRAIN_SAMPLES = 2000  # CPU 训练，用 2000 条快速验证

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

TEST_SENTENCES = [
    "Hello, how are you?",
    "The weather is beautiful today.",
    "I love programming.",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ============================================================
    # Step 1: 加载数据集
    # ============================================================
    print("\n" + "=" * 50)
    print("下载 OPUS-100 英法数据集")
    print("=" * 50)

    # OPUS-100 格式: {'translation': {'en': '...', 'fr': '...'}}
    raw_dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train")
    print(f"训练集总大小: {len(raw_dataset)} 条")

    # 只取前 N 条用于快速训练
    train_raw = raw_dataset.select(range(min(MAX_TRAIN_SAMPLES, len(raw_dataset))))
    print(f"使用训练集: {len(train_raw)} 条")

    # 从训练集中分出一部分做验证
    split = train_raw.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    val_data = split["test"]
    print(f"训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")

    # OPUS-100 数据格式检查
    sample = raw_dataset[0]
    print(f"样本格式: {sample}")
    print(f"  EN: {sample['translation']['en']}")
    print(f"  FR: {sample['translation']['fr']}")

    # ============================================================
    # Step 2: 加载 Tokenizer
    # ============================================================
    print(f"\n加载 tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检测 assistant 起始 token
    assistant_marker = "<|im_start|>assistant\n"
    assistant_start_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)

    # ============================================================
    # Step 3: 数据格式化
    # ============================================================
    SYSTEM_PROMPT = "You are a professional English-to-French translator."

    def build_messages(en_text, fr_text=None):
        user_msg = f"Translate to French: {en_text}"
        assistant_msg = fr_text if fr_text else ""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]

    demo = tokenizer.apply_chat_template(
        build_messages("Hello", "Bonjour"), tokenize=False
    )
    print(f"\nChat Template 样例:\n{demo}\n")

    def tokenize_batch(examples):
        """批量分词，labels 仅在 assistant 部分有效"""
        texts = []
        for t in examples["translation"]:
            msgs = build_messages(t["en"], t["fr"])
            txt = tokenizer.apply_chat_template(msgs, tokenize=False)
            texts.append(txt)

        tokenized = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)

        all_labels = []
        for input_ids in tokenized["input_ids"]:
            labels = [-100] * len(input_ids)
            found = False
            for i in range(len(input_ids) - len(assistant_start_ids) + 1):
                if input_ids[i:i + len(assistant_start_ids)] == assistant_start_ids:
                    for j in range(i + len(assistant_start_ids), len(input_ids)):
                        labels[j] = input_ids[j]
                    found = True
                    break
            if not found:
                labels = input_ids.copy()
            all_labels.append(labels)

        tokenized["labels"] = all_labels
        return tokenized

    # 分词
    print("分词训练集...")
    train_dataset = train_data.map(tokenize_batch, batched=True, remove_columns=train_data.column_names)
    print("分词验证集...")
    val_dataset = val_data.map(tokenize_batch, batched=True, remove_columns=val_data.column_names)

    # 验证
    s = train_dataset[0]
    valid_pct = 1 - s["labels"].count(-100) / len(s["labels"])
    valid_tokens = [tid for tid, lbl in zip(s["input_ids"], s["labels"]) if lbl != -100]
    decoded = tokenizer.decode(valid_tokens, skip_special_tokens=True) if valid_tokens else "N/A"
    print(f"\nLabels 验证 → 有效 token 占比: {valid_pct:.1%}")
    print(f"  有效部分解码: {decoded}")

    # ============================================================
    # Step 4: 加载模型 + LoRA
    # ============================================================
    print(f"\n加载模型 {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else {"": device},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ============================================================
    # Step 5: 翻译函数
    # ============================================================
    def translate(en_text):
        model.eval()
        msgs = build_messages(en_text, fr_text=None)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        for marker in ["<|im_end|>", "</s>"]:
            result = result.replace(marker, "").strip()
        # 只取第一行有效内容
        result = result.split("\n")[0].strip()
        return result

    # ============================================================
    # Step 6: 微调前测试
    # ============================================================
    print("\n" + "=" * 50)
    print("微调前翻译测试")
    print("=" * 50)
    for sent in TEST_SENTENCES:
        print(f"  EN: {sent}")
        print(f"  FR: {translate(sent)}")
        print()

    # ============================================================
    # Step 7: 训练
    # ============================================================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=device.type == "cuda",
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 50)
    print("开始 LoRA 微调...")
    print(f"  数据: OPUS-100 en-fr")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  有效 batch: {BATCH_SIZE} × {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  设备: {device}")
    print("=" * 50)

    trainer.train()

    # 保存
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n模型已保存到: {OUTPUT_DIR}")

    # ============================================================
    # Step 8: 微调后测试
    # ============================================================
    print("\n" + "=" * 50)
    print("微调后翻译测试")
    print("=" * 50)
    for sent in TEST_SENTENCES:
        print(f"  EN: {sent}")
        print(f"  FR: {translate(sent)}")
        print()

    print("\n✅ 微调完成！")


if __name__ == "__main__":
    main()
