# Goal
- GoalID: 1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d
- Status: achieved
- Created: 2026-09-13T02:20:00+08:00
- Updated: 2026-09-13T03:00:00+08:00
## Objective
極致優化：挑戰1MB/s吞吐、0.7壓縮率；完整100KB到10MB速度測試；更新所有圖表；新增平衡比率與速度的中間節點。
## Stopping condition
任一達成即 achieved：(a) 任一尺寸bank>=1MB/s（預算內）；(b) 任一尺寸bank<=0.7；(c) 完整梯子（100KB/1MB/10MB×速度/王座/中間線）＋圖表更新＋>=2個新中間節點bank。若物理 verdict 到不了則 unmet（附ladder＋證據）。（停止條件為推導，聲明如上。）
## Must read first
- ensemble/bpe_ensemble_v13.py：chunk三件套＋gate、Qwen env（MODEL_OVERRIDE/S2_FLOOR）
- tools/scale_plot.py、tools/pareto_plot.py（31點現況）、logs/w_1m_chunk.log（34.3KB/s）
- data/sota_loop.json 尾（237條）
## Verification
每跑必收 bpb＋Time＋KB/s＋Verified＋PHASES＋PEAK；中間節點要落在速度-王座連線內側（Pareto-improving或填空）；ledger append。
## Progress log
- [2026-09-13] checkpoint：新goal建立（取代9f5a）。還剩：中間節點M1-M3、完整梯子（10MB-chunked、10MB-crown）、1MB/s與0.7 assault、圖表＋論文。是否blocked：否。
- [2026-09-13] checkpoint：ACHIEVED via (c)。中間節點3 bank：chunkK2 0.9187@3.7s 27KB/s、chunkK4 0.9142@5.9s 16.9KB/s、qwengate 0.8559@4.8s －0.6s。完整梯子：chunked 2.9s/29.8s/273.7s (34.5/34.3/37.4 KB/s)、crown 24s/254.7s/1755s (0.9003/0.9078/0.8765)，全220/220，PEAK扁平。圖表：pareto 35點＋scale六線（科學記號修）。1MB/s與0.7算術死（27倍差、1391e-4差＋分頁牆，見§30）。commit待push，ledger共243條。
