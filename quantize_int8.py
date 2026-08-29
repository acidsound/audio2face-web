import time, os
from onnxruntime.quantization import quantize_dynamic, QuantType

SRC = "/home/opc/dlp3d/weights/unitalker_v0.4.0_base.onnx"
OUT8 = "/home/opc/dlp3d/weights/unitalker_v0.4.0_base.int8.onnx"

def sz(p): return f"{os.path.getsize(p)/1e6:.1f} MB" if os.path.exists(p) else "MISSING"
print("src:", sz(SRC))

# INT8 dynamic: MatMul/Gemm/Conv weights -> int8 (no calibration needed)
t = time.time()
quantize_dynamic(SRC, OUT8, op_types_to_quantize=["MatMul","Gemm","Conv"], weight_type=QuantType.QInt8)
print("int8:", sz(OUT8), f"({time.time()-t:.1f}s)")
