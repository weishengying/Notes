官方文档： [FastAPI](https://fastapi.tiangolo.com/zh/)

## 示例：
```python
from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

for route in app.routes: # 打印看看有哪些路径和对应的允许的请求方法
    print(route)
```

*通过app的修饰符 @app.get()，定义了路径（route）和请求（method）方法，以及请求参数和请求到来时触发的函数。*

启动服务：
```bash
uvicorn main:app --reload
```
>uvicorn main:app 命令含义如下:
>* main：main.py 文件（一个 Python "模块"）。
>* app：在 main.py 文件中通过 app = FastAPI() 创建的对象。
>* --reload：让服务器在更新代码后重新启动。仅在开发时使用该选项。

可以通过 curl 命令访问这个服务：
```bash
curl http://localhost:8000/ # GET 方法请求根路径，触发 read_root 函数
curl http://localhost:8000/items/1 # GET 方法请求路径 /items/{item_id}, 触发 read_item 函数，q 参数为空
curl http://localhost:8000/items/1?q=hello # GET 方法请求路径 /items/{item_id}?q=hello, 触发 read_item 函数，q 参数为 hello
```

接下来使用 vllm 离线推理引擎，创建一个最简单的 LLM 推理服务：
```python
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import FastAPI, File, Form, Request, UploadFile

from vllm import LLM, SamplingParams
model_path = "..."
llm = LLM(model=model_path)
sampling_params = SamplingParams(temperature=0.85, top_p=0.9)

app = FastAPI()

@app.post("/chat") # POST：用于向服务器提交数据，通常用于创建资源。
async def openai_v1_chat_completions(raw_request: Request):
    message = await raw_request.json()
    prompt =  message["message"]
    output = llm.generate([prompt], sampling_params)
    # 输出结果
    return output[0].outputs[0].text
```
启动服务后使用 curl 命令发送请求数据:
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "who are you?"}'
```
>* -X 选项用于指定 HTTP 请求的方法。在这个例子中，POST 表示发送一个 HTTP POST 请求。POST 请求通常用于向服务器提交数据，例如提交表单数据或上传文件.(get是默认的method，不需要加-X)
>* -H 选项用于设置 HTTP 请求头。在这个例子中，Content-Type: application/json 表示请求体中的数据格式是 JSON。服务器可以根据这个头信息来解析请求体中的数据.
>* -d 选项用于指定请求体中的数据。在这个例子中，{"message": "who are you?"} 表示一个 JSON 对象，其中 message 字段表示用户输入的文本。

## 关于异步
[FastAPI文档中一个有趣的简单解释](https://fastapi.tiangolo.com/zh/async/)