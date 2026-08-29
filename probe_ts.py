import numpy as np, onnxruntime as ort, subprocess, os, wave, tempfile

def make_wav(text, sr=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path=f.name
    subprocess.run(["espeak-ng","-v","en-us","-s","160","-w",path,text],check=True)
    w=wave.open(path,"rb")
    n=w.getnframes(); raw=w.readframes(n)
    data=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
    target_len=int(len(data)*sr/22050)
    idx=np.round(np.linspace(0,len(data)-1,target_len)).astype(int)
    dec=data[idx]
    os.unlink(path)
    return dec.astype(np.float32)

text="Hello, I am your digital assistant, nice to meet you today."
audio=make_wav(text)
print("audio len samples:", len(audio), "≈", round(len(audio)/16000,2), "s")

M_FP32="/home/opc/dlp3d/weights/unitalker_v0.4.0_base.onnx"
M_INT8="/home/opc/dlp3d/weights/unitalker_v0.4.0_base.int8.onnx"

def run(m, audio, ts_val):
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    s=ort.InferenceSession(m, so, providers=["CPUExecutionProvider"])
    a=np.asarray(audio, dtype=np.float32)[None,:]
    emo=np.array([1],dtype=np.int32)
    ts=np.array([ts_val],dtype=np.int32)
    return s.run(None,{"audio":a,"emo_id":emo,"time_steps":ts})[0]

for tv in [1, 10, 50, 100]:
    try:
        o=run(M_FP32, audio[:16000], tv)
        print(f"FP32 ts={tv}: shape={o.shape} min={o.min():.3f} max={o.max():.3f} mean={o.mean():.3f}")
    except Exception as e:
        print(f"FP32 ts={tv}: ERR {e}")
