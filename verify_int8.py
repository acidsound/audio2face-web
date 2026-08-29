import numpy as np, onnxruntime as ort, subprocess, os, wave, tempfile

def make_wav(text, sr=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path=f.name
    subprocess.run(["espeak-ng","-v","en-us","-s","160","-w",path,text],check=True)
    w=wave.open(path,"rb"); n=w.getnframes(); raw=w.readframes(n)
    data=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
    tl=int(len(data)*sr/22050); idx=np.round(np.linspace(0,len(data)-1,tl)).astype(int)
    os.unlink(path); return data[idx].astype(np.float32)

def load(m):
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    return ort.InferenceSession(m, so, providers=["CPUExecutionProvider"])

def run(s, audio, ts):
    a=np.asarray(audio,dtype=np.float32)[None,:]
    return s.run(None,{"audio":a,"emo_id":np.array([1],np.int32),"time_steps":np.array([ts],np.int32)})[0][0]  # (ts,51)

text="Hello, I am your digital assistant, nice to meet you today."
audio=make_wav(text); T=50
s32=load("/home/opc/dlp3d/weights/unitalker_v0.4.0_base.onnx")
s8 =load("/home/opc/dlp3d/weights/unitalker_v0.4.0_base.int8.onnx")
o32=run(s32,audio,T); o8=run(s8,audio,T)
print("shapes FP32",o32.shape,"INT8",o8.shape)

# 1) global cosine similarity
flat32=o32.reshape(-1); flat8=o8.reshape(-1)
cos=np.dot(flat32,flat8)/(np.linalg.norm(flat32)*np.linalg.norm(flat8)+1e-9)
print(f"cosine similarity FP32 vs INT8: {cos:.4f}")
print(f"max abs diff: {np.abs(o32-o8).max():.4f}, mean abs diff: {np.abs(o32-o8).mean():.4f}")

# 2) per-frame cosine
per=np.array([np.dot(o32[i],o8[i])/(np.linalg.norm(o32[i])*np.linalg.norm(o8[i])+1e-9) for i in range(T)])
print(f"per-frame cosine min={per.min():.3f} mean={per.mean():.3f}")

# 3) which channel looks like mouth-open? find channel with highest variance & speech-correlation
energy=np.abs(audio); fr=len(audio)/T; 
speech=np.array([energy[int(i*fr):int((i+1)*fr)].mean() for i in range(T)])
# correlate each channel with speech energy
corr={c:np.corrcoef(o32[:,c],speech)[0,1] for c in range(51)}
top=sorted(corr.items(),key=lambda x:-abs(x[1]))[:5]
print("top channels by |corr| with speech energy (FP32):", [(c,round(v,3)) for c,v in top])
# show mouth-ish channel over frames (use the highest-corr channel)
mc=top[0][0]
print("mouth channel",mc,"FP32 frames:",np.round(o32[:,mc],2))
print("mouth channel",mc,"INT8 frames:",np.round(o8[:,mc],2))
print("speech energy frames:",np.round(speech,3))
