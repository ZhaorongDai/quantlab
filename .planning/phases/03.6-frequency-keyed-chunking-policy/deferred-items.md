# Phase 03.6 Deferred Items

范围边界之外、在执行期间被发现但**刻意未处理**的事项。每条只在被显式决定（提交、删除或
归档）之后才可标记 `status: resolved`。

## Deferred Items

- `example/chunk_grid_demo.py` 未被追踪，且不属于任何计划的 diff
  status: open
  **What:** 工作树里有一个未追踪文件 `example/chunk_grid_demo.py`（5,857 字节，创建于
  2026-09-13 11:08:17）。它是一个教学脚本，解释 `XrBackend._append_encoding` 的落盘格网
  规则与 `append_dim_size`——即 plan `03.6-05` 的交付物——但 `03.6-05-SUMMARY.md` 的
  `key-files` 没有列它，`git log -- example/` 里也没有它，仓库内没有任何代码 import 或
  引用它（`grep -rn chunk_grid_demo tests/ quantlab/ *.py` 零命中）。
  **Discovered during:** plan `03.6-06`，Task 3 提交之后的未追踪文件检查。
  **Why deferred:** 它不是本计划任何一个 task 造成的，创建时间早于本计划开始（14:59:16）
  近四小时。本计划的 scope fence 只覆盖 WR-04..WR-09，`files_modified` 逐一列名且不含它；
  把别的计划未提交的脚本塞进本计划的元数据提交会造成归属错误，而删除它会销毁他人的工作。
  **Note:** 本计划 Task 3 的 `git status --porcelain -- example/` 门在运行时只输出
  ` M example/acquisition.md`，未报告这个 `??` 条目；提交之后再查则报告了它。该门的判定
  （「本 task 只动了 acquisition.md」）本身仍然成立且为真，但它当时没有看见这个未追踪文件，
  这一点如实记录在此，不作粉饰。
  **Next step:** 由 `03.6-05` 的所有者或阶段验证决定：提交（若它确为有意交付的示例，应同时
  在 `example/README.md` 登记）、删除（若是临时草稿）、或明确接受其未追踪状态。
