#!/usr/bin/env node
"use strict";

/**
 * zllm CLI — Node.js wrapper for Python neural compression engine.
 *
 * Usage:
 *   zllm encode <input> [-o output.zllm] [--preset fast|balanced|ratio|ratio-qwen]
 *   zllm decode <input.zllm> [-o output.txt]
 *   zllm bench [--size 100kb|1mb|10mb] [--preset ...]
 *   zllm info <input.zllm>
 *   zllm --version
 *   zllm --help
 */

const { execFileSync, spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const VERSION = "0.1.0";
const BACKEND = path.resolve(__dirname, "..", "zllm", "cli.py");
const PROJECT_ROOT = path.resolve(__dirname, "..");

// ─── Python detection ─────────────────────────────────────────────
function findPython() {
  const candidates = [
    process.env.ZLLM_PYTHON,
    "python3",
    "python",
    "py",
    process.platform === "win32"
      ? path.join(process.env.LOCALAPPDATA || "", "Programs", "Python", "Python311", "python.exe")
      : null,
    "/usr/bin/python3",
  ].filter(Boolean);

  for (const cmd of candidates) {
    try {
      const r = spawnSync(cmd, ["--version"], {
        timeout: 5000,
        stdio: "pipe",
      });
      if (r.status === 0 && /Python 3\.\d+/.test(r.stdout?.toString() || "")) {
        return cmd;
      }
    } catch {}
  }
  return null;
}

// ─── Dependency check (no shell, avoid arg-quoting issues) ────────
function checkDeps(python) {
  try {
    execFileSync(python, [
      "-c",
      "import torch, transformers, numba, numpy",
    ], {
      stdio: "pipe",
      timeout: 15000,
    });
    return true;
  } catch {
    return false;
  }
}

// ─── Main ─────────────────────────────────────────────────────────
function main() {
  const args = process.argv.slice(2);

  // --version / -v
  if (args.includes("--version") || args.includes("-v")) {
    console.log(`zllm ${VERSION}`);
    console.log(`  Python backend: ${BACKEND}`);
    const py = findPython();
    if (py) {
      try {
        const v = execFileSync(py, ["--version"], { stdio: "pipe", timeout: 5000 });
        console.log(`  Runtime: ${v.stdout.toString().trim()}`);
      } catch {}
    }
    process.exit(0);
  }

  // --help / -h / no args
  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    printHelp();
    process.exit(0);
  }

  // Find Python
  const python = findPython();
  if (!python) {
    console.error("Error: Python 3 not found on PATH.");
    console.error("Install Python 3.10+ or set ZLLM_PYTHON environment variable.");
    process.exit(1);
  }

  // Check deps (no shell — avoids Windows arg-quoting bug)
  if (!checkDeps(python)) {
    console.error("Error: Missing Python dependencies.");
    console.error("Install with:");
    console.error("  pip install torch transformers numba numpy");
    console.error("");
    console.error("Or if using a virtualenv, set ZLLM_PYTHON to point to its python.");
    process.exit(1);
  }

  // Forward to Python backend (no shell — args passed as array)
  const pyArgs = [BACKEND, ...args];
  const result = spawnSync(python, pyArgs, {
    stdio: "inherit",
    timeout: 30 * 60 * 1000,
    cwd: PROJECT_ROOT,
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
  });

  process.exit(result.status ?? 1);
}

function printHelp() {
  console.log(`
zllm ${VERSION} — Neural text compression

Usage:
  zllm encode <input> [-o output.zllm] [--preset fast|balanced|ratio|ratio-qwen]
  zllm decode <input.zllm> [-o output.txt]
  zllm bench [--size 100kb|1mb|10mb] [--preset fast|balanced|ratio|ratio-qwen]
  zllm info <input.zllm>

Presets:
  fast         Speed-optimized (SmolLM2 chunked, ~34.5 KB/s, 0.9276 bpb)
  balanced     Speed-ratio balance (SmolLM2 chunked K2048, ~27 KB/s, 0.9187)
  ratio        Best SmolLM2 ratio (ov4096/K8192, 0.9003 bpb)
  ratio-qwen   Best Qwen ratio (0.8442 bpb, fits 8GB VRAM)

Environment:
  ZLLM_PYTHON  Path to Python 3.10+ with torch/transformers/numba

Requirements:
  - Python 3.10+ on PATH (or ZLLM_PYTHON)
  - pip install torch transformers numba numpy
  - GPU recommended (NVIDIA CUDA), CPU fallback available

Examples:
  zllm encode document.txt -o document.zllm --preset fast
  zllm decode document.zllm -o document.txt
  zllm bench --preset balanced --size 1mb
  zllm info document.zllm

For more info: https://github.com/thumb2086/continual-replay-llm
`);
}

main();
