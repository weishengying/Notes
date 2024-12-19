
import torch
import torch.nn.functional as F
def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def compare_f8_mm(size=(16, 16), dtype=torch.float8_e4m3fn) -> None:
    # create test inputs
    # Note: cuBLASLt float8 matmul requires column major
    #        for the second argument
    x = torch.randn (size, dtype=torch.float16, device='cuda')
    w = torch.randn (size, dtype=torch.float16, device='cuda').t()
    # do a scaled cast to float8 on the inputs
    x_f8, x_inv_s = to_float8(x, dtype=dtype)
    w_f8, w_inv_s = to_float8(w)
    # perform the float8 matmul
    y = torch._scaled_mm(x_f8, w_f8, 
                        scale_a=x_inv_s, scale_b=w_inv_s,
                        out_dtype=torch.float16)
    # compare output of float8 matmul to the fp16 baseline
    cos_sim = F.cosine_similarity(torch.mm(x, w).reshape(-1),
                                  y.reshape(-1), dim=0)
    # Cosine similarity between scaled mm and reference
    # should be close to 1.0
    print(f'cos_sim {cos_sim.item():.4f}')


def benchmark(sizes, dtype=torch.float8_e4m3fn) -> None:
    for m, k, n in sizes:
        # 初始化 CUDA 事件
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start_2 = torch.cuda.Event(enable_timing=True)
        end_2 = torch.cuda.Event(enable_timing=True)


        x = torch.randn ((m, k), dtype=torch.float16, device='cuda')
        w = torch.randn ((n, k), dtype=torch.float16, device='cuda').t()

        w_f8, w_inv_s = to_float8(w)

        # fp8_gemm
        ## warm up
        x_f8, x_inv_s = to_float8(x, dtype=dtype)
        for _ in range(10):
            y = torch._scaled_mm(x_f8, w_f8,       
                            scale_a=x_inv_s, scale_b=w_inv_s,
                            out_dtype=torch.float16)

        start.record() # 记录开始时间
        x_f8, x_inv_s = to_float8(x, dtype=dtype)
        y = torch._scaled_mm(x_f8, w_f8, 
                        scale_a=x_inv_s, scale_b=w_inv_s,
                        out_dtype=torch.float16)
        end.record() # 记录结束时间
        torch.cuda.synchronize()
        elasped_time_1 = start.elapsed_time(end)

        # fp16 gemm
        ## warm_up
        for _ in range(10):
            torch.mm(x, w)
        start_2.record() # 记录开始时间
        torch.mm(x, w)
        end_2.record() # 记录结束时间
        torch.cuda.synchronize()
        
        elasped_time_2 = start_2.elapsed_time(end_2)
        print(f'm,k,n: {m},{k},{n}, fp8_gemm: {elasped_time_1:.2f} ms, fp16_gemm: {elasped_time_2:.2f} ms')

if __name__ == "__main__":
    # compare_f8_mm()
    # mlp（以llama3 8B 为例， intermediate_size=14336， hidden_size=4096, m 是batch）
    benchmark([(1, 14336, 4096),  # (m, k, n)
            (1, 4096, 14336),
            (2, 14336, 4096),
            (2, 4096, 14336),
            (4, 14336, 4096),
            (4, 4096, 14336),
            (8, 14336, 4096),
            (8, 4096, 14336),
            (32, 14336, 4096),
            (32, 4096, 14336),
            ])
    # q_k_v proj, m是token的长度
    benchmark([(1, 4096, 3*4096),  # (m, k, n)
            (4, 4096, 3*4096),
            (8, 4096, 3*4096),
            (32, 4096, 3*4096),
            (128, 4096, 3*4096),
            (512, 4096, 3*4096),
            (2048, 4096, 3*4096),
            (8192, 4096, 3*4096),
            ])
