from transformers import AutoTokenizer
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "/mnt/shared/maas/ai_story/llama3_as_def_en_8b_l3_sfw_1119"
quantized_model_dir = "/mnt/shared/maas/ai_story/llama3_as_def_en_8b_l3_sfw_1119-FP8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="dynamic")

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config=quantize_config
)
model.quantize()
model.save_quantized(quantized_model_dir)
