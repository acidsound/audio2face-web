import numpy as np, onnxruntime as ort, subprocess, os, wave, tempfile, time

def make_wav(text, sr=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f: path=f.name
    subprocess.run(["espeak-ng","-v","en-us","-s","160","-w",path,text],check=True)
    w=wave.open(path,"rb"); n=w.getnframes(); raw=w.readframes(n)
    data=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
    tl=int(len(data)*sr/22050); idx=np.round(np.linspace(0,len(data)-1,tl)).astype(int)
    os.unlink(path); return data[idx].astype(np.float32)

text="Hello, I am your digital assistant, nice to meet you today. This is a latency benchmark for the int8 model streaming."
audio=make_wav(text); sr=16000
print(f"audio: {len(audio)} samples = {len(audio)/sr:.2f}s")

so=ort.SessionOptions(); so.intra_op_num_threads=4
s=ort.InferenceSession("/home/opc/dlp3d/weights/unitalker_v0.4.0_base.int8.onnx", so, providers=["CPUExecutionProvider"])

# streaming simulation: 200ms window, 50ms hop (typical realtime chunking)
win=int(0.20*sr); hop=int(0.05*sr)
times=[]; frames=0
t0=time.time()
for start in range(0, len(audio)-win, hop):
    chunk=audio[start:start+win]
    a=chunk[None,:].astype(np.float32)
    ts=np.array([50],dtype=np.int32)  # 50 frames out
    t_in=time.time()
    o=s.run(None,{"audio":a,"emo_id":np.array([1],np.int32),"time_steps":ts})[0]
    times.append((time.time()-t_in)*1000)
    frames+=1
wall=time.time()-t0
times=np.array(times)
print(f"chunks: {frames}, window={win/sr*1000:.0f}ms, hop={hop/sr*1000:.0f}ms")
print(f"per-chunk infer: mean={times.mean():.1f}ms  p95={np.percentile(times,95):.1f}ms  max={times.max():.1f}ms  min={times.min():.1f}ms")
print(f"total wall (incl overhead): {wall*1000:.0f}ms  vs audio {len(audio)/sr*1000:.0f}ms  -> realtime ratio {wall/(len(audio)/sr):.2f}x")
print(f"infer-only total: {times.sum():.0f}ms  vs audio {len(audio)/sr*1000:.0f}ms -> compute ratio {times.sum()/(len(audio)/sr*1000):.2f}x")
# verdict
if times.mean() < hop/sr*1000:
    print("VERDICT: per-chunk infer < hop interval -> realtime-capable on this CPU")
else:
    print("VERDICT: per-chunk infer exceeds hop -> would lag on this CPU (but browser/WebGPU may differ)")
