async for 是 Python 异步编程中的一个语法结构，用于在异步生成器（async generator）中迭代。它允许你在异步环境中逐个获取生成器的值，而不需要手动处理异步上下文。

## 基本用法
1. 定义异步生成器

首先，你需要定义一个异步生成器函数。异步生成器函数使用 async def 定义，并且使用 yield 来生成值。

```python
async def async_generator():
    for i in range(5):
        await asyncio.sleep(1)  # 模拟异步操作
        yield i
```
在这个例子中，async_generator 是一个异步生成器，它会每隔 1 秒生成一个数字。

2. 使用 async for 迭代

你可以使用 async for 语法来迭代异步生成器生成的值。
```python
async def main():
    async for value in async_generator():
        print(value)

asyncio.run(main())
```
在这个例子中，main 函数使用 async for 来迭代 async_generator 生成的值，并打印每个值。

## 示例代码
下面是一个完整的示例，展示了如何使用 async for 来迭代异步生成器：

```python
import asyncio

# 定义一个异步生成器
async def async_generator():
    for i in range(5):
        await asyncio.sleep(1)  # 模拟异步操作
        yield i

# 使用 async for 进行迭代
async def main():
    async for value in async_generator():
        print(value)

# 运行主函数
asyncio.run(main())
```

输出结果：
```bash
0
1
2
3
4
```
在这个例子中，async_generator 每隔 1 秒生成一个数字，main 函数使用 async for 逐个获取这些数字并打印出来。

总结：
* async for 用于在异步环境中迭代异步生成器生成的值。

* 异步生成器使用 async def 定义，并且使用 yield 来生成值。

* async for 语法可以方便地处理异步生成器的迭代，而不需要手动管理异步上下文。

通过 async for，你可以轻松地在异步编程中处理需要逐个获取值的场景，例如异步读取文件、异步网络请求等。

