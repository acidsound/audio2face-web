// infer.worker.js — ORT Web inference off the main thread.
// One worker per playback; terminate() on stop → no dangling promises, no OOM from retries.
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.webgpu.min.mjs");

let session = null;
let emo = null, ts = null;
const SR = 16000, WIN = 0.05, TS_OUT = 50;

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    try {
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
      ort.env.wasm.numThreads = Math.min(4, m.threads || 4);
      const buf = m.buf; // ArrayBuffer transferred from main
      session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      emo = new ort.Tensor("int32", new Int32Array([1]), [1]);
      ts = new ort.Tensor("int32", new Int32Array([TS_OUT]), [1]);
      // warmup
      const wa = new ort.Tensor("float32", new Float32Array(WIN * SR), [1, WIN * SR]);
      for (let k = 0; k < 8; k++) { await session.run({ audio: wa, emo_id: emo, time_steps: ts }); }
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", message: err.message });
    }
    return;
  }
  if (m.type === "infer" && session) {
    try {
      const aT = new ort.Tensor("float32", m.chunk, [1, m.chunk.length]);
      const out = await session.run({ audio: aT, emo_id: emo, time_steps: ts });
      const data = out[Object.keys(out)[0]].data;
      self.postMessage({ type: "bs", bs: Array.from(data.slice(-51)), dt: m.dt }, []);
    } catch (err) {
      self.postMessage({ type: "error", message: err.message });
    }
  }
};
