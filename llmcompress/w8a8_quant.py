from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot

# Select model and load it.
MODEL_ID = "/mnt/shared/maas/ai_story/mistral_as_def_12b_french_sfw_500_1126"
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
# ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
# ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

# ds = load_dataset("AgentWaller/german-oasst1-qa-format")
# ds = ds["validation"]

file_path = 'mistral_as_def_12b_french_sfw_500_1126.prompts.json'
ds = load_dataset('json', data_files=file_path)
ds = ds["train"]
# def preprocess(example):
#     messages = example["input"] + example["output"]
#     return {
#         "text": messages
#     }
# def preprocess(example):
#     messages = example["messages"]
#     if messages[0]["role"] != "system":
#         system_message = {'content': '', 'role':'system'}
#         messages.insert(0, system_message)
#     return {
#         "text": tokenizer.apply_chat_template(
#             example["messages"],
#             tokenize=False,
#         )
#     }
# ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure algorithms. In this case, we:
#   * apply SmoothQuant to make the activations easier to quantize
#   * quantize the weights to int8 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = "/mnt/shared/maas/ai_story/mistral_as_def_12b_french_sfw_500_1126" + "-W8A8-Dynamic-Per-Token-2"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)