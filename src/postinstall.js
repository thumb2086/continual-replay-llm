#!/usr/bin/env node
"use strict";

/**
 * postinstall — verify Python + deps after npm install.
 * Warns (doesn't fail) if Python/torch missing; user can fix later.
 */

const { spawnSync } = require("child_process");

function findPython() {
  const candidates = [
    process.env.ZLLM_PYTHON,
    "python3", "python", "py",
  ].filter(Boolean);
  for (const cmd of candidates) {
    try {
      const r = spawnSync(cmd, ["--version"], { stdio: "pipe", timeout: 5000, shell: process.platform === "win32" });
      if (r.status === 0 && /Python 3\.\d+/.test(r.stdout?.toString() || "")) return cmd;
    } catch {}
  }
  return null;
}

const py = findPython();
if (!py) {
  console.warn("\n[zllm] Warning: Python 3 not found. Install Python 3.10+ to use zllm.");
  console.warn("       Then: pip install torch transformers numba numpy\n");
  process.exit(0); // don't block npm install
}

try {
  spawnSync(py, ["-c", "import torch, transformers, numba, numpy"], { stdio: "pipe", timeout: 15000 });
  console.log("[zllm] Python deps verified.");
} catch {
  console.warn("\n[zllm] Warning: Missing Python deps. Run:");
  console.warn("       pip install torch transformers numba numpy\n");
}
