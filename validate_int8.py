import numpy as np, onnxruntime as ort, time, os

M = "/home/opc/dlp3d/weights/unitalker_v0.4.0_base.int8.onnx"
print("file:", f"{os.path.getsize(M)/1e6:.1f} MB")

so = ort.SessionOptions()
so.intra_op_num_threads = 4
# Try WebGPU EP first (to prove browser-side feasibility), fall back to CPU
for ep in (["WebGPU"], ["CPUExecutionProvider"]):
    try:
        sess = ort.InferenceSession(M, so, providers=ep)
        print("loaded on EP:", sess.get_providers())
        break
    except Exception as e:
        print("EP", ep, "failed:", e)

# dummy inputs from model metadata
it = {i.name: i for i in sess.get_inputs()}
print("inputs:", [(n, [d if d else -1 for d in s]) for n,s,_ in [(i.name, i.shape, i.type) for i in sess.get_inputs()]])
# audio ~ 1s @16k mono float; emo_id/time_steps int64 scalars
audio = np.random.randn(1, 16000).astype("float32")
emo = np.array([1], dtype="int32")
ts = np.array([1], dtype="int32")
t = time.time()
out = sess.run(None, {"audio": audio, "emo_id": emo, "time_steps": ts})
print("infer OK, out shape:", [o.shape for o in out], f"({time.time()-t*2:.3f}s)")
