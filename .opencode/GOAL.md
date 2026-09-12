# Goal
- GoalID: 9f5a2d3b-6c7e-4f8a-9b0c-1d2e3f4a5b6c
- Status: pursuing
- Created: 2026-09-13T08:10:00+08:00
- Updated: 2026-09-13T08:10:00+08:00
## Objective
100KB/s吞吐量：進行100KB到10MB的速度測試，把各尺寸吞吐往100KB/s打（現：100KB 34.5KB/s、1MB ~20KB/s、10MB ~16KB/s）。
## Stopping condition
任一尺寸 bank ≥100KB/s（預算內：速度線贏SOTA 0.9389）即 achieved；若梯子量完＋forward物理拆解證明到不了則 unmet（附ladder＋證據）。（停止條件為推導，聲明如上。）
## Must read first
- ensemble/bpe_ensemble_v13.py：chunk三件套段（CHUNK_PRE/HEAD/SPARSE）、SUBCLOCKS
- tools/scale_plot.py（TIME/BPB梯子現況）、logs/v_chunkAB3.log（2.9s基線）
- data/sota_loop.json 尾（236條）
## Verification
每跑必收 bpb＋Time＋KB/s＋Verified＋PHASES/SUBCLOCKS＋PEAK；ledger append；bank才push。
## Progress log
- [2026-09-13] checkpoint：新goal建立（取代三殺achieved）。還剩：1MB-chunked、10MB-chunked、chunk+overlap解鎖、forward剖析、100KB/s verdict。是否blocked：否。
