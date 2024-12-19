
# Python `@dataclass` Usage Example

In Python, `@dataclass` is a decorator that simplifies creating data classes. It helps to automatically generate common methods like `__init__`, `__repr__（用于返回对象的字符串表示形式）`, and `__eq__（比较）` for representing data objects without manually writing these methods.

## 1. Using the @dataclass Decorator

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    city: str = "Unknown"  # Default value provided

# Creating a Person instance
person1 = Person(name="Alice", age=30)
print(person1)  # Output: Person(name='Alice', age=30, city='Unknown')
```

**Output:**
```python
Person(name='Alice', age=30, city='Unknown')
```

## 2. Automatically Generated Methods

`@dataclass` automatically generates the following methods:
- `__init__`: Constructor method
- `__repr__`: String representation for printing the object
- `__eq__`: Comparison for object equality

Example:

```python
person2 = Person(name="Alice", age=30)
print(person1 == person2)  # Output: True
```

**Output:**
```python
True
```

## 3. Adding Type Hints and Default Values

You can add type hints for each field and provide default values. Note that fields with default values must appear after all fields without default values.

```python
@dataclass
class Book:
    title: str
    author: str
    pages: int = 100
```

## 4. `field` Function

You can use the `field` function to further customize fields. For example, `field` can be used to specify whether a field is included in comparisons, initialization, etc.

```python
from dataclasses import field

@dataclass
class Car:
    model: str
    year: int
    color: str = field(default="Blue", repr=False)  # Not displayed in __repr__
```

## 5. Mutable Default Values

When using mutable default values (like lists or dictionaries) in a data class, use `field(default_factory=...)` to avoid sharing the same mutable object.

```python
from typing import List

@dataclass
class Team:
    name: str
    members: List[str] = field(default_factory=list)

team1 = Team(name="Team Alpha")
team1.members.append("Alice")
print(team1)  # Output: Team(name='Team Alpha', members=['Alice'])
```

**Output:**
```python
Team(name='Team Alpha', members=['Alice'])
```

## 6. frozen=True 参数
frozen=True 是 @dataclasses.dataclass 装饰器的一个参数，用于指定生成的数据类是不可变的（immutable）。具体来说：

* 不可变（Immutable）：当 frozen=True 时，数据类的实例一旦创建，就不能再修改其属性。任何尝试修改属性的操作都会引发 FrozenInstanceError 异常。

* 可变（Mutable）：默认情况下，frozen=False（或不指定 frozen 参数），数据类的实例是可变的，即可以随时修改其属性。
## Summary

`@dataclass` is a convenient way to define data classes in a simplified manner, especially suited for classes representing data structures. By using `@dataclass`, Python handles much of the redundant code, making the code more concise and readable.
