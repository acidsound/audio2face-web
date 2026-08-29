import time, os
from onnxruntime.quantization import quantize_dynamic, QuantType

SRC = "/home/opc/dlp3d/weights/unitalker_v0.4.0_base.onnx"
OUT16 = "/home/opc/dlp3d/weights/unitalker_v0.4.0_base.fp16.onnx"
OUT8  = "/home/opc/dlp3d/weights/unitalker_v0.4.0_base.int8.onnx"

def sz(p): return f"{os.path.getsize(p)/1e6:.1f} MB" if os.path.exists(p) else "MISSING"
print("src:", sz(SRC))

# --- FP16: weights -> float16 (no calibration, near-lossless) ---
t = time.time()
quantize_dynamic(SRC, OUT16, quant_type=QuantType.QFloat16)
print("fp16:", sz(OUT16), f"({time.time()-t:.1f}s)")

# --- INT8 dynamic: MatMul/Gemm/Conv weights -> int8 (no calibration) ---
t = time.time()
quantize_dynamic(SRC, OUT8, weight_op_types=["MatMul","Gemm","Conv"])
print("int8:", sz(OUT8), f"({time.time()-t:.1f}s)")
