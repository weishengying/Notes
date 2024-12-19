import torch.nn as nn
from typing import Type, Dict

class CustomOp(nn.Module):
    # Dictionary of all custom ops (classes, indexed by registered name).
    op_registry: Dict[str, Type['CustomOp']] = {} # Type['CustomOp'] 这种写法表示引用自身或其子类的类型


    # Decorator to register custom ops.
    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            print(f"Registering op: {name}")
            print(f"Class: {op_cls}")
            assert name not in cls.op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            cls.op_registry[name] = op_cls
            return op_cls
        return decorator

@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    def forward(self):
        print("RMSNorm forward")


op = RMSNorm()
op.forward()
