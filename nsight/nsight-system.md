**使用 nsight-system （nsys） 进行 timeline 分析**

```bash
nsys profile --stats=true -o prof python example.py
```

使用 --stats=true 选项时，nsys 会在分析完成后生成一个统计报告文件（通常是 .qdstrm 或 .qdrep 文件的扩展名）。
此外生成一个 prof.nsys-rep 文件。


**使用 nvtx 进行自定义标记**
