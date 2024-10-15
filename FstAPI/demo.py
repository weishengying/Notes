from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import FastAPI, File, Form, Request, UploadFile

from vllm import LLM, SamplingParams
import uvicorn

llm = LLM(model="/mnt/shared/maas/ai_story/llama3_as_en_12b_mistral_v2_0929/", quantization="fp8", max_model_len=8192)

sampling_params = SamplingParams(temperature=0.85, top_p=0.9, repetition_penalty=1.05, top_k=7, max_tokens=256, stop=["<|eot_id|>", "<|end_of_text|>", "</s>"])

app = FastAPI()

@app.get("/health")
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)

@app.post("/chat")
async def openai_v1_chat_completions(raw_request: Request):
    message = await raw_request.json()
    prompt =  message["message"]
    output = llm.generate([prompt], sampling_params)
    print(f"output:{output[0].outputs[0].text}")
    # 输出结果
    return output[0].outputs[0].text

if __name__ == "__main__":
    uvicorn.run(app, 
                host="127.0.0.1", 
                port=8000,
                loop="uvloop", #使用 uvloop 事件循环。uvloop 是一个高性能的 asyncio 事件循环的替代品，通常比标准 asyncio 事件循环更快
                )
# 直接 python demo.py 启动服务
