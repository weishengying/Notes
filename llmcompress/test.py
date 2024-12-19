from datasets import load_dataset

file_path = 'llama3_as_def_nemo_german_sft_sfw_1121.prompts.json'
ds = load_dataset('json', data_files=file_path)
ds = ds["train"]
print(ds)


ds = load_dataset("AgentWaller/german-oasst1-qa-format")
ds = ds["validation"]
def preprocess(example):
    messages = example["input"] + example["output"]
    return {
        "text": messages
    }
ds.map(preprocess)
print(ds)
