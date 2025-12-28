import torch
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration

# ----------------------------
# 1. VALID MODEL ID
# ----------------------------
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# ----------------------------
# 2. Load Processor & Model
# ----------------------------
processor = LlavaProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# ----------------------------
# 3. Load MRI Image
# ----------------------------
image = Image.open(rf"E:\Brain tumor Detection\lime_explanation_output12.png").convert("RGB")

# ----------------------------
# 4. Medical-Safe Prompt
# ----------------------------
prompt = """
You are a medical imaging analysis assistant for educational purposes only.

Analyze the provided brain MRI image and generate a structured, non-diagnostic report.

Include:
1. Image overview
2. Observed imaging features
3. Possible associated symptoms (hypothetical)
4. General precautions and next steps

Rules:
- Do not diagnose diseases
- Use cautious language
- This is not medical advice
"""

# ----------------------------
# 5. Prepare Inputs
# ----------------------------
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
).to(model.device)

# ----------------------------
# 6. Generate
# ----------------------------
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True
    )

# ----------------------------
# 7. Decode Output
# ----------------------------
result = processor.decode(outputs[0], skip_special_tokens=True)

print("\n===== MRI ANALYSIS REPORT =====\n")
print(result)
