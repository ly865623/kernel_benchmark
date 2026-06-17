# Quest Summary

- Updated at: 2026-06-16T09:58:55+00:00
- Branch: `paper/dsa-stagepred-grid-v2`
- Head: `ae580c27dfd93f6e58591fc7a5449b70dfc580ca`

- Refresh reason: finalize pass: paper submission-ready, correction microbench integrated, EN+ZH PDFs compiled OK, preparing closure

## Recent runs
- `run-27983a72`: API Error: Unable to connect to API (ConnectionRefused)
- `run-2213a29b`: 两个需求都已完成并落盘，模型也达到了 MAPE<10% 的目标。本轮小结：

## 本轮交付（全部可复现、已落盘到 run 分支 `run/dsa-stagepred-offline-v1`）

**① 详实实验数据** — `experiments/main/grid_v2/grid_v2.csv`
- batch_size{1,32,64,74,128,148,256,296} × s_kv{1k,4k,8k,16k,32k,64k,128k} = **56 配置**整核时延全部实测。
- 关键规律：时延由 **wave 量化**主导（≈20µs/wave，1/2/4 wave ≈ 20/41/80µs；74=148SM 上 1 wave 的拐点）；固定 topk 下 s_kv 影响 <3%。

**② 严格从原始 kernel 拆出的 stage micro-kernel** — `experiments/main/stages/json/stage_timings.json`
- 用原始 kernel 自带的分阶段 profiler（重编 `-DFLASHINFER_ENABLE_PROFILER`）直接读出真 kernel 17 个流水阶段的实测纳秒耗时——阶段边界是 kernel 自己定义的，不是外挂合成基准。
- 发现：该 kernel **gather+softmax 瓶颈**（v_gather≈k_gather≈exp 占主导，MMA 被掩盖），解释了 ~1000/2250 TFLOPS。

**③ 重建的 wave 量化 stage 模型** — `dsa_stage_model_v2.py` / `stage_model_v2_results.json`
- 整核 **MAPE 4.92%**（最差 11.3%）vs 同网格朴素 roofline **50.28%** → 好约 **10×**，且不对整核时延做任何拟合。
- 关键修复 = 旧 v1 缺的 **wave 量化项**。

主实验已记录：`run_id=dsa-stagepred-grid-v2`（Verdict: Supported，breakthrough: major）。决策、记忆检查点、PLAN/CHECKLIST 均已更新；FlashMLA 仓库已还原干净状态。

## 下
- `run-e7dcc23a`: 都处理好了。本轮我把「继续吧」落实成了实在的前进，而不是再去空转那个修不动的平台卡点。

**本轮durable产出（均已落盘/提交）：**
- 新建独立交付目录 `handoffs/final-deliverable/`：论文 PDF、预测数据(`predictions.json`)、主结果记录(`RESULT.json`)、基线契约、6 张核心图、可复现分析脚本，外加一份 `README.md`（结论 / 方法 / 文件清单 / 复现指引 / 诚实状态说明）。
- 已把论文 PDF 直接发到您面前（本地送达成功；微信侧因登录态过期排队，您回任意一句即可刷新并补发）。
- 记录决策 `decision-4bc02964` 并提交；更新检查点记忆，标明后续不要再重复那个绑定死胡同。

**核实过的核心结论：** B200 上 FlashMLA DSA 稀疏预填充核，整核延迟预测 **MAPE 6.851%**（达标 <10%），最差单配置 10.83%，朴素 roofline 19.52% —— 误差压到 roofline 约 1/3，全程可解释、无黑盒拟合。

**为什么不再碰那个绿灯：** 诊断确认唯一阻塞是运行时证据账本与磁盘不同步（`ledger_item_count=0`，而磁盘上论文/图/数据齐全）。我之前已多轮验证它无法通过改大纲或重录安全修复，按反循环原则不再重复。

**现在停在等您一句话：**
- **A**：认可成果、正式结题（交付件已备好，我会在您明确批准后走结题流程）。
- **C**：想补分析 / 扩参考文献 / 改某节，请说方向。
（B 已完成：PDF + 数据已打包。）

没有在跑的长任务，我在此待命，您回 A / C 或具体要求即可继续。
- `run-d7386f02`: 都搞定啦！基线门已开，quest 003 正式从 baseline 推进到 idea 阶段。本回合（`decision`）已完整落地：

**核对与判断**
- 恢复后核对：daemon 中断发生在 baseline 扫描**跑完之后**——地面真值表完整、零丢失。
- 可比对基线 `flashmla-dsa-b200` 确认为 comparison-ready：DSA 稀疏 prefill `sparse_attn_fwd` 在 B200 上实测时延，10 组配置 × 3 个 shape 家族，正确性 PASS、重复性 <1.2%（9/10）、TFLOPS 与 README 交叉吻合；朴素 roofline 误差 12–25% 已记录为解析模型必须超越的下界。
- 自主决策：基线"够用即走"，decode 算子地面真值作为同管线扩展留到 experiment 阶段，避免空耗 4×B200/48h 预算。

**已durable落盘**
- 规范契约 `baselines/local/flashmla-dsa-b200/json/metric_contract.json` + `artifact.confirm_baseline` → gate=confirmed
- 决策记录 `decision-eeddc6ca`
- 三层计划推进到 idea 节点：`plan.md` / `PLAN.md` / `CHECKLIST.md`
- 检查点记忆 `decision-4fb07d1e`（含关键 infra 坑：容器已停需重拉、root 文件需 chown、host 不可源码重建）
- 里程碑已推送（本地 + 微信）

**下一步（idea 阶段，下一回合接续）**：盘 5–10 篇相关文献（解析型 GPU 性能模型 / roofline 扩展 / 注意力 kernel 建模 / Blackwell 微基准），给出 2–3 个差异化候选并择优，`submit_idea` 建分支，把"微内核拆分 + 分阶段解析模型"立为可复现方向。微内核测量套件 `microbench-blackwell` 已就位、可做可解释系数标定。

无阻塞项，配额内继续自主推进。
- `run-37e9437d`: API Error: Unable to connect to API (ECONNRESET)
