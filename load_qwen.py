"""
从 HuggingFace 拉取 Qwen2.5-0.5B / Qwen2-0.5B 预训练模型
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Qwen 系列最小的预训练模型选择：
# 1. "Qwen/Qwen2.5-0.5B"        - 0.5B 参数，最新版
# 2. "Qwen/Qwen2-0.5B"          - 0.5B 参数
# 3. "Qwen/Qwen1.5-0.5B"        - 0.5B 参数
# 注：Qwen 没有正好 0.8B 的版本，最接近的有：
#     - 0.5B / 1.5B / 3B / 7B

MODEL_NAME = "Qwen/Qwen2.5-0.5B"   # 约 0.5B 参数，最接近 0.8B

print(f"正在从 HuggingFace 下载: {MODEL_NAME}")
print("-" * 50)

# 加载 tokenizer
print("下载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 加载模型 (使用半精度节省内存)
print("下载模型权重...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True
)

print("-" * 50)

# 模型信息
print(f"模型: {MODEL_NAME}")
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"设备: {model.device}")

# 简单测试
print("\n--- 测试生成 ---")
prompt = "中国的首都是"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"输入: {prompt}")
print(f"输出: {response}")

print("\n✅ 模型下载并测试成功！")

# 查看模型结构
print("\n--- 模型结构 ---")
print(model)
