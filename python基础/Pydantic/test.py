from pydantic import BaseModel, Field, ValidationError
from typing import Optional

class User(BaseModel):
    id: int
    name: str = Field(..., min_length=3) # 输入的 str 至少要有三个字符
    email: Optional[str] = None
    age: Optional[int] = Field(None, ge=18)

# 创建实例并验证数据
try:
    user = User(id=1, name="John", email="john@example.com", age=25)
    print(user)
except ValidationError as e:
    print(e)

# 从字典解析数据
user_data = {"id": 2, "name": "Jane", "email": "jane@example.com", "age": 30}
user = User(**user_data)
print(user)

# 序列化为字典
user_dict = user.dict()
print(user_dict)

# 序列化为 JSON
user_json = user.json()
print(user_json)