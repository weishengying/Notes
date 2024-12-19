from vllm import LLM
model = LLM("/mnt/shared/maas/ai_story/llama3_as_def_nemo_german_sft_sfw_1121-W8A8-Dynamic-Per-Token", tensor_parallel_size=2, max_model_len=8192)
output = model.generate("My name is")

for o in output:
    generated_text = o.outputs[0].text
    print(generated_text)