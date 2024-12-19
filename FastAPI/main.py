from typing import Union

from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

for route in app.routes:
    print(route)
    
uvicorn.run(app, 
            host="127.0.0.1", 
            port=8000, 
            loop="uvloop", #使用 uvloop 事件循环。uvloop 是一个高性能的 asyncio 事件循环的替代品，通常比标准 asyncio 事件循环更快
            )
# 直接 python main.py 启动服务