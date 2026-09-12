"""Export SmolLM2-135M to ONNX (CUDA) for v13 USE_ORT=1. Run ONCE.

Prereqs (install when the user greenlights runs -- NOT yet):
    python -m pip install "optimum[onnxruntime]" onnxruntime-gpu
Run (CPU is fine, a few minutes, no GPU needed):
    python ort_export.py
Output: ./ort-smollm2/model.onnx (+ configs). v13 reads ORT_MODEL_DIR.
"""
import sys

sys.path.insert(0, ".")

from optimum.onnxruntime import ORTModelForCausalLM
from bpe_compress import MODEL_DIR

print("exporting", MODEL_DIR, "-> ./ort-smollm2 (use_cache=True for KV)")
model = ORTModelForCausalLM.from_pretrained(MODEL_DIR, export=True)
model.save_pretrained("./ort-smollm2")
print("saved ./ort-smollm2/model.onnx")
