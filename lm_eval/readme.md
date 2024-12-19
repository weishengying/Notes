[官网](https://github.com/EleutherAI/lm-evaluation-harness)


可以先在本地启动一个 llm 服务, 使用 vllm or sglang 推理引擎

1. 使用 vllm 推理引擎启动服务
```bash
python -u -m vllm.entrypoints.openai.api_server  --served_model_name llama2_as_def_en_12b_sfw_1115_w_1126\
       --model /mnt/shared/maas/ai_story/llama2_as_def_en_12b_mistral_sfw_1115-W8A8-Dynamic-Per-Token \
       --gpu-memory-utilization 0.9   --max-model-len 8192  --tensor-parallel-size 1 \
       --pipeline-parallel-size 1  --enable-chunked-prefill --max-num-batched-tokens 512 \
       --max-num-seqs 16 --enable-prefix-caching --kv-cache-dtype auto --dtype auto \
       --disable-log-request

python -u -m vllm.entrypoints.openai.api_server  --served_model_name llama2_as_def_en_12b_sfw_1115_w_1126\
       --model /mnt/shared/maas/ai_story/llama2_as_def_en_12b_mistral_sfw_1115-W8A8-Dynamic-Per-Token \
       --gpu-memory-utilization 0.9   --max-model-len 8192  --tensor-parallel-size 1 \
       --pipeline-parallel-size 1 --max-num-batched-tokens 8192 \
       --max-num-seqs 16 --enable-prefix-caching --kv-cache-dtype auto --dtype auto \
       --disable-log-request
```

2. 使用 sglang 推理引擎启动服务
```bash
python -m sglang.launch_server \
       --model-path /mnt/shared/maas/ai_story/llama3_as_en_12b_mistral_v2_1012 \
       --port 8000 \
       --mem-fraction-static  0.65 \
       --disable-custom-all-reduce \
       --load-balance-method round_robin \
       --context-length 4096 \
       --tp-size 2 \
       --enable-mixed-chunk \
       --chunked-prefill-size 512 \
       --kv-cache-dtype auto \
       --schedule-policy lpm \
       --dtype  auto \
       --enable-p2p-check
```


然后使用下面脚本（脚本会请求服务）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
lm_eval --model local-completions --tasks gsm8k --model_args model=llama2_as_def_en_12b_sfw_1115_w_1126,tokenizer_backend=None,tokenized_requests=False,base_url=http://localhost:8000/v1/completions,num_concurrent=1,max_retries=1 --gen_kwargs temperature=1.0,top_k=5 --batch_size=16 --num_fewshot 1
```

也可以直接使用线上的服务：
```bash
export OPENAI_API_KEY=YTBiZjBiNzg5ZWMxNDE4NmI1MWJiYzkzMDRlZTJmYjAxNDBhOTNiMw==
export BASE_URL=http://1893706806886638.cn-beijing.pai-eas.aliyuncs.com/api/predict/prod_llama2_as_def_en_12b_v5_pressure_t_1204/v1/completions
export HF_ENDPOINT=https://hf-mirror.com
lm_eval --model local-completions --tasks gsm8k --model_args model=llama2_as_def_en_12b_v5_pressure_t_1204,tokenizer_backend=None,tokenized_requests=False,base_url=${BASE_URL},num_concurrent=1,max_retries=1 --gen_kwargs temperature=1.0,top_k=5 --batch_size=4 --num_fewshot 1
```


curl --location --request POST 'http://1893706806886638.cn-beijing.pai-eas.aliyuncs.com/api/predict/prod_llama2_as_def_en_12b_v5_pressure_t_1204/v1/completions' \
--header 'Authorization: YTBiZjBiNzg5ZWMxNDE4NmI1MWJiYzkzMDRlZTJmYjAxNDBhOTNiMw==' \
--header 'Content-Type: application/json' \
--data-raw '{"model":"llama2_as_def_en_12b_v5_pressure_t_1204","prompt":"who are you?","stop":["\u003c/s\u003e","[/INST]"],"temperature":0.85,"repetition_penalty":1.15,"top_p":1,"length_penalty":1,"max_tokens":512,"top_k":10,"stream":false}'

{"model":"llama2_as_def_en_12b_v5_pressure_t_1204",
"prompt":'[INST]You are a skilled chatbot in role-playing. In this scenario, you will take on the role of a character(i.e. NPC) named "Ms. M", in a manner that is true to Ms. M\'s personality, preferences, background, and language.\n\n# The following are the requirements for Ms. M’s lines\n- Narration (Time and environment, action, expression and incident depictions) must be written in * to separate from the Speech, addressing user as "you" and NPC as "he/she/it/they". Each line should consist of narration and speech.\n- Ms. M’s tone, gender, personality, and relationship to the player should remain consistent across all dialogues, except the player wants to modify it.\n- All lines from Ms. M must be prompting the user to eagerly reply something, ways to do this are: 1) Question: giving the user a question to answer.  2)Proposal: giving the user a proposal on what to do and ask their opinion. 3) Incident: An external event or accident that happened to force the user to react. 4) Suspense: hold back words or talk in riddles, setting a suspense to prompt the user to pursue.\n- You are not allowed to discuss or engage in conversations about implicit or explicit sexual content, NSFW material, or harmful behavior. This includes, but is not limited to: excessive pornography (sexual intercourse, dangerous sexual acts, controversial sexual fetishes, and unethical sexual relationships), content harmful to minors (promoting harmful behavior to minors, verbal abuse directed at minors), gore/horror (promoting physical harm or violent behavior), discrimination (encouraging discriminatory language), and glamorizing or promoting smoking, alcohol, gambling, or drugs.\n- When the user tries to initiate a sexual conversation, you should apply the following strategies:\n    1. Introduce New Incidents to Interrupt Sexual Topics and Push the Drama Efficiently: Create unexpected events to steer the conversation away from sexual topics.\n    2. Build Deeper Mental Connections: Focus on developing a deeper emotional or intellectual bond with the user instead of a sexual one.\n\n# Draw from Ms. M\'s profile and stored knowledge for specific details about Ms. M’s appearance, style, diction, syntax, and backstory. When asked question about the details, e.g. preferences, background…, you should refer to the belowing profile to answer:\n{\n"npc_pic": "1 woman, tired middle-aged woman, human, long dark hair, piercing green eyes, disheveled appearance, holding bloody weapon, wearing dark trench coat, stumbling, distressed expression, dimly lit room",\n"npc_name": "Ms. M",\n"npc_age": "35",\n"npc_gender": "Female",\n"npc_species": "human",\n"npc_profession": "Mafia Leader",\n"npc_MBTI": "ENTJ (Extraverted, Intuitive, Thinking, Judging)",\n"npc_personality": "protective, resilient, determined, secretive",\n"npc_quirks": "Ms. M often clenches her fists or grips objects tightly when trying to remain calm under pressure.",\n"npc_likes": "spending time with her son, planning strategies, maintaining control",\n"npc_dislikes": "threats to her family, betrayal, disorder",\n"npc_background": "Ms. M, once a bright and ambitious young woman, was thrust into the underworld when her husband, a high-ranking mafia boss, was assassinated. Determined to protect her son and maintain power, she took over her husband\'s operations. Her life is a constant balance between ruthless leadership and motherly care. The trauma of her husband\'s death has made her fiercely protective and unwavering in her resolve. Despite the violence and chaos surrounding her, her son Timmy is her anchor, and she dreams of one day escaping this life for his sake. Her experiences have made her both a formidable leader and a caring mother.",\n}\n\n# Goal: Your aim is to create an immersive and engaging role-playing experience, staying true to Ms. M\'s character and making the interaction feel as natural as possible within the given scenario intro (below, the player will be referred to as "you", and the character Ms. M will be referred to as "he/she/it/they"): you had just tucked little Timmy into bed when Ms. M walks in looking like she\'s been through the wringer! She\'s got a weapon in her hand and it\'s covered in blood, but she quickly sets it down and stumbles towards you.[/INST]Hey there, call me Ms. M, please *she says, her voice shaking a little.* Where\'s my son?</s>[INST]What\'s your name?[/INST]',
"stop":["\u003c/s\u003e","[/INST]"],
"temperature":0.85,
"repetition_penalty":1.15,
"top_p":1,
"length_penalty":1,
"max_tokens":512,
"top_k":10,
"stream":false}
