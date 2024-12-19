# Python Pydantic BaseModel 示例

`pydantic` 是一个用于数据验证和解析的 Python 库，它基于类型注解来提供强大的数据验证功能。`BaseModel` 是 `pydantic` 中的一个核心类，用于定义数据模型。通过继承 `BaseModel`，你可以创建一个具有类型检查和验证功能的类。

## 主要用途

1. **数据验证**：
   - `BaseModel` 会根据你在类中定义的字段类型自动进行数据验证。如果传入的数据不符合定义的类型或约束条件，`pydantic` 会抛出 `ValidationError` 异常。

2. **数据解析**：
   - 你可以将传入的 JSON、字典或其他数据结构自动解析为 `BaseModel` 的实例。`pydantic` 会根据字段的类型注解自动进行类型转换。

3. **默认值和可选字段**：
   - 你可以为字段设置默认值，或者将字段标记为可选（使用 `Optional` 或 `None`）。

4. **字段约束**：
   - 你可以为字段添加各种约束，如最小值、最大值、正则表达式匹配等。

5. **嵌套模型**：
   - `BaseModel` 支持嵌套模型，即一个模型中的字段可以是另一个 `BaseModel` 的实例。

6. **序列化和反序列化**：
   - `BaseModel` 实例可以轻松地转换为字典或 JSON 格式，并且可以从字典或 JSON 格式反序列化为 `BaseModel` 实例。

## 示例代码

```python
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

```

输出：

```bash
id=1 name='John' email='john@example.com' age=25
id=2 name='Jane' email='jane@example.com' age=30
{'id': 2, 'name': 'Jane', 'email': 'jane@example.com', 'age': 30}
{"id": 2, "name": "Jane", "email": "jane@example.com", "age": 30}
```