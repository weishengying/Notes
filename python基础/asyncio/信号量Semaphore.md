asyncio.Semaphore 是 Python 的 asyncio 库中的一个同步原语，用于限制并发访问某个资源的数量。它类似于线程中的信号量（threading.Semaphore），但适用于异步编程环境。

## 1. 基本用法
1. 创建信号量：

你可以通过指定一个整数来创建一个信号量，这个整数表示允许同时访问资源的数量。
```python
import asyncio

semaphore = asyncio.Semaphore(value=3)

```
在这个例子中，semaphore 允许最多 3 个任务同时访问某个资源。

2. 获取信号量

当你需要访问受保护的资源时，你需要先获取信号量。如果信号量的计数器大于 0，则计数器会减 1，并允许你访问资源；否则，任务会被阻塞，直到有其他任务释放信号量。
```python
async def access_resource():
    async with semaphore:
        # 访问受保护的资源
        print("Accessing resource")
        await asyncio.sleep(1)  # 模拟资源访问时间
```
使用 async with semaphore 语法可以自动处理信号量的获取和释放。

3. 释放信号量

当你完成对资源的访问后，信号量会自动释放，计数器会加 1，允许其他任务获取信号量。
```python
async def main():
    tasks = [access_resource() for _ in range(10)]
    await asyncio.gather(*tasks)

asyncio.run(main())
```
在这个例子中，main 函数创建了 10 个任务，但每次最多只有 3 个任务可以同时访问资源。

## 示例代码
下面是一个完整的示例，展示了如何使用 asyncio.Semaphore 来限制并发访问：

```python
import asyncio

# 创建一个信号量，允许最多 3 个任务同时访问资源
semaphore = asyncio.Semaphore(3)

async def access_resource(task_id):
    async with semaphore:
        print(f"Task {task_id} is accessing the resource")
        await asyncio.sleep(1)  # 模拟资源访问时间
        print(f"Task {task_id} is done")

async def main():
    tasks = [access_resource(i) for i in range(10)]
    await asyncio.gather(*tasks)

asyncio.run(main())
```

输出结果：
```bash
Task 0 is accessing the resource
Task 1 is accessing the resource
Task 2 is accessing the resource
Task 0 is done
Task 3 is accessing the resource
Task 1 is done
Task 4 is accessing the resource
Task 2 is done
Task 5 is accessing the resource
Task 3 is done
Task 6 is accessing the resource
Task 4 is done
Task 7 is accessing the resource
Task 5 is done
Task 8 is accessing the resource
Task 6 is done
Task 9 is accessing the resource
Task 7 is done
Task 8 is done
Task 9 is done
```