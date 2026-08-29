#!/usr/bin/env bash
# Download UniTalker Audio2Face onnx models into public/models/
# INT8 (103MB) is the browser target; FP32 (389MB) is fallback.
set -e
cd "$(dirname "$0")/public/models"
base="https://github.com/LazyBusyYang/CatStream/releases/download/a2f_cicd_files"
if [ ! -f unitalker_v0.4.0_base.onnx ]; then
  echo "downloading FP32 (389MB)..."
  curl -L -o unitalker_v0.4.0_base.onnx "$base/unitalker_v0.4.0_base.onnx"
fi
echo "FP32 present. INT8 must be produced by quantizing FP32 (see notes)."
echo "Done. Place int8 onnx at public/models/unitalker_v0.4.0_base.int8.onnx"
