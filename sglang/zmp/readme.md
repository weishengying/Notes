# ZeroMQ 简介（ZMQ）
简单了解下 ZMQ，不做深入了解，只需能理解 sglang 中如何使用的就行

sglang 中主要用到了 ZMQ 的 `PUSH` 和 `PULL` 模式，即 `推-拉`模式：

**推-拉模式的工作原理**
>* 推者（Push）：负责将任务推送给拉者。推者会将任务均匀地分发给所有连接的拉者。
>* 拉者（Pull）：负责接收并处理推者推送的任务。每个拉者独立处理接收到的任务。

**推-拉模式的特点**
>*负载均衡：推者会将任务均匀地分发给所有拉者，从而实现负载均衡。
>*并行处理：多个拉者可以并行处理任务，提高处理效率。
>*简单易用：推-拉模式的实现相对简单，适合快速搭建任务分发系统。


**sglang 中 TokenizerManager 函数中初始化通信的代码如下：**
```python
# Init inter-process communication
context = zmq.asyncio.Context(2) #创建一个ZeroMQ上下文对象，管理着所有套接字的生命周期。参数2表示使用两个I/O线程来处理消息传递
self.recv_from_detokenizer = context.socket(zmq.PULL) #创建一个PULL类型的套接字，用于从其他进程接收消息。PULL套接字通常用于接收任务或数据。
self.recv_from_detokenizer.bind(f"ipc://{port_args.tokenizer_ipc_name}") # 将PULL套接字绑定到一个特定的IPC（进程间通信）地址。

self.send_to_scheduler = context.socket(zmq.PUSH) #创建一个PUSH类型的套接字，用于向其他进程发送消息。PUSH套接字通常用于分发任务或数据。
self.send_to_scheduler.connect(f"ipc://{port_args.scheduler_input_ipc_name}") #将PUSH套接字连接到另一个IPC地址。
```

**sglang 中 Scheduler 函数中初始化通信的代码如下：**
```python
# Init inter-process communication
context = zmq.Context(2)

if self.tp_rank == 0:
    self.recv_from_tokenizer = context.socket(zmq.PULL)
    self.recv_from_tokenizer.bind(f"ipc://{port_args.scheduler_input_ipc_name}")

    self.send_to_detokenizer = context.socket(zmq.PUSH)
    self.send_to_detokenizer.connect(f"ipc://{port_args.detokenizer_ipc_name}")
else:
    self.recv_from_tokenizer = self.send_to_detokenizer = None
```
注意，Scheduler 只在rank=0 时才需要通信

**sglang 中 DetokenizerManager 函数中初始化通信的代码如下：**
```python
# Init inter-process communication
context = zmq.Context(2)
self.recv_from_scheduler = context.socket(zmq.PULL)
self.recv_from_scheduler.bind(f"ipc://{port_args.detokenizer_ipc_name}")

self.send_to_tokenizer = context.socket(zmq.PUSH)
self.send_to_tokenizer.connect(f"ipc://{port_args.tokenizer_ipc_name}")
```