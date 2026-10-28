# DQT Engineering Code Plan

> **Driving Question Tokens — NAVSIM Data Annotation, SimWAM Joint Flow, OneVL Question-First AR**  
> Status: implementation specification / 尚未执行或验证训练；所有 `python -m dqt...` 命令是**本项目拟实现的 CLI**，不是现有上游仓库命令。  
> Companion: [method_desc.md](method_desc.md). 代码风格：Python 3.11+、PyTorch、Hydra/OmegaConf、Pydantic、Parquet/JSONL、pytest、ruff、mypy；具体版本遵循所选 upstream checkpoint 与 CUDA 环境的兼容性测试。

## 1. 目标、首要交付物与不做的事

### 1.1 端到端要实现什么

```text
NAVSIM navtrain
  ├── [t<=0] RGB history / multi-view / ego / navigation
  ├── [t<=0] detection & track / optional map evidence
  │         ↓
  │     deterministic scene evidence builders
  │         ↓
  │     Qwen local VLM answer annotator + rule validation
  │         ↓
  │     DQT annotations (question codes, masks, confidence, evidence)
  │         │
  └── [t>0] GT ego trajectory [8,3] (supervision ONLY)
            │
            ↓
     shared training sample ───┬── SimWAM / action-only
                              │      (a) predicted-Q conditioning
                              │      (b) joint Q slots + trajectory slots
                              └── OneVL / AR
                                     DQT → coarse RVQ → residual RVQ
                                     → decode [8,3]
                                             ↓
                                  NAVSIM v1 PDMS (4 s / 10 Hz)
```

最早要看到的验证链路（所有依赖均用真实数据验证，不需要先训练完整两个模型）：

1. 读取 1 个 NAVSIM sample，导出当前/历史 RGB、ego、导航、det/tracks 与 GT 未来轨迹，明确允许/禁止的字段。
2. 自动生成 26 道题的候选证据，Qwen 对适用题输出严格 JSON；验证 100 个样本的人审可读性和证据定位。
3. 对 1k–10k 样本标注，计算选择题可监督性、分布与一致性；冻结 `question_schema_v1.json` 和 `annotation_manifest`。
4. 使用已有 SimWAM checkpoint 复现 `[8,3]` 输出与 NAVSIM v1 官方评分，微调仅增加 Q conditioning，检查 Q 是否给轨迹带来增益。
5. 用同一份标注数据接入 OneVL，首先 DQT→直接轨迹/原动作头，再 DQT→RVQ 粗到细。
6. 最后再上 SimWAM **联合 QuestionSlot Flow** 与 AR 的时间分辨率扩展；不要把两个新结构同时作为第一版实现。

### 1.2 明确不做

- 不在第一版重写 NAVSIM、SimWAM 或 OneVL 核心训练框架；只做 `adapter`、`dataset`、`head`、`train hooks`。
- 不将 Ground Truth future video / future track / future ego trajectory 交给教师去回答**当前事实题**。
- 不用 GT DQT 直接作为 test-time policy input；仅在 `oracle_diagnostic` 明确可见的设置下使用。
- 不把离散 ABCD label index 直接当作连续轨迹值去添加高斯噪声。
- 不把自己定义的“红灯越线”检查冒充 NAVSIM 官方 PDMS 或真实交通法规全覆盖。
- 不让所有 26 道题都在每个样本硬回答：动态 trigger、`NA`、`UNK`、confidence mask 是数据结构的一部分。

## 2. 依赖与上游接口核验（开工前自动执行）

### 2.1 固定外部仓库版本

```text
third_party/
  navsim/                # git submodule, pinned SHA + branch (e.g. v1-compatible)
  simwam/                # git submodule, pinned SHA
  onevl/                 # git submodule, pinned SHA
  onevl_training/        # if available; if not, implement standalone HuggingFace/TRL adapter
  lightnav0_reference/   # optional reading/reference only; not a runtime dependency
```

通过 `vendor_lock.yaml` 记录每个 commit SHA、checkpoint SHA256、processor/tokenizer 版本、归一化 statistics、依赖环境。这里不提供未经实际运行验证的 SHA，也不假定 OneVL 的训练仓库与推理仓库自动兼容。

### 2.2 需要先校验的已知事实

- NAVSIM agent 以 `AgentInput`（合法传感器/ego/nav）做 `compute_trajectory`，训练 `TargetBuilder` 能读 Scene 的 GT，但推理 `AgentInput` **不能**直接提供未来真值。
- NAVSIM v1 evaluator 的时域是 4 s / 10 Hz；对于官方 SimWAM 预训练配置，模型是 `[8,3]`（8 个 future steps），经官方 sampling/interpolation 输出到评估频率。**第一版按 `[8,3]` 训练。**
- SimWAM 官方 checkpoint action expert 为 flow matching 的规划器，原始 Video Expert 的未来分支通过 attention mask 与动作隔离；本项目增加 QuestionSlots 时必须遵守同样的未来隔离。
- OneVL 现有方案含 latent visual/language token + auxiliary decoder。**“显式选择题→轨迹 RVQ”属于我们新增结构**，而不是给 OneVL 的原 inference prompt 随便插几行文本就可以完事。
- LightNav-0 的 coarse-to-fine 主要是 RVQ 的残差精度多尺度；若增加 `1 Hz→2 Hz→10 Hz`，要实现独立的 time-scale codec / decoder。
- 官方 Qwen3.5 Hub 列表没有确认 `Qwen3.5-72B` 这一官方型号。本地模型按用户实际 `MODEL_PATH` 识别；首个集成测试必须确认 `model_type`、`vision_config` / processor 和 1图、多图、视频输入都可用。可能是其他 Qwen 72B VLM、重命名或私有微调模型。

### 2.3 校验脚本预期输出

```bash
python -m dqt.tools.inspect_runtime \
  --navsim-root third_party/navsim \
  --simwam-root third_party/simwam \
  --onevl-root third_party/onevl \
  --teacher-model /absolute/path/to/local-qwen \
  --output artifacts/runtime_report.json
```

自动检查：GPU、dtype / FlashAttention、model image processor、图像多轮输入、checkpoint `state_dict` keys、SimWAM trajectory length、navsim frame/sensor availability、原始 action normalization、官方 evaluator 是否可运行。任何不符合都返回非零退出码和差异说明；不要悄悄 reshape 轨迹以掩盖兼容错误。

## 3. 推荐仓库布局（将标签、策略、评估彻底解耦）

```text
dqt-driving/
├── README.md
├── pyproject.toml
├── Makefile
├── configs/
│   ├── data/navsim_v1.yaml
│   ├── annotation/qwen_local.yaml
│   ├── annotation/answers_v1.yaml
│   ├── policies/simwam_q_condition.yaml
│   ├── policies/simwam_joint_q_flow.yaml
│   ├── policies/onevl_dqt_rvq.yaml
│   ├── codec/rvq_8x3.yaml
│   └── eval/navsim_v1.yaml
├── schemas/
│   ├── question_schema_v1.json     # all Q01-Q26 choices/triggers/semantics
│   ├── decision_trace.schema.json
│   └── sample.schema.json
├── src/dqt/
│   ├── core/
│   │   ├── types.py                # typed Sample, Evidence, Answer, Trajectory
│   │   ├── questions.py            # QuestionRegistry, codecs, triggers
│   │   ├── tokenization.py         # DQT token ↔ answer; ordering and masks
│   │   └── coordinates.py          # ego/world conversion, timestamps, units
│   ├── data/
│   │   ├── navsim_source.py        # legal data adapter, causal-time filter
│   │   ├── evidence_builder.py     # ego, maps, detections, tracks, ROI
│   │   ├── image_sampler.py        # camera mosaic, timestamp handling
│   │   ├── dataset.py              # train-ready iterable/loader
│   │   ├── split_guard.py          # scene/log leakage checks
│   │   └── collators.py
│   ├── annotation/
│   │   ├── teacher_client.py       # OpenAI-compatible local endpoint OR HF
│   │   ├── prompts.py              # numbered options, no future data
│   │   ├── candidate_selector.py   # dynamic question triggering
│   │   ├── labeler.py              # perception & decisions
│   │   ├── structured_parser.py    # JSON schema / tool-type output validation
│   │   ├── validators.py           # static geometric consistency checks
│   │   ├── auditor.py              # uses future GT only in separate run
│   │   ├── review_queue.py         # needs human review
│   │   └── manifest.py             # resume, sha, source provenance
│   ├── models/
│   │   ├── shared/question_encoder.py
│   │   ├── shared/question_predictor.py
│   │   ├── shared/action_constraints.py
│   │   ├── diffusion/simwam_adapter.py
│   │   ├── diffusion/q_conditioner.py
│   │   ├── diffusion/q_flow_slots.py
│   │   ├── diffusion/masks.py
│   │   ├── ar/onevl_adapter.py
│   │   ├── ar/sequence_builder.py
│   │   ├── ar/constrained_decode.py
│   │   ├── ar/trajectory_rvq.py
│   │   └── ar/multiscale_decoder.py
│   ├── train/
│   │   ├── train_question_predictor.py
│   │   ├── train_simwam.py
│   │   ├── train_rvq.py
│   │   └── train_onevl.py
│   ├── eval/
│   │   ├── question_metrics.py
│   │   ├── planning_metrics.py
│   │   ├── navsim_agent.py
│   │   ├── counterfactual.py
│   │   ├── ablations.py
│   │   └── visualization.py
│   └── tools/
│       ├── inspect_runtime.py
│       ├── export_samples.py
│       ├── annotate.py
│       ├── validate_annotations.py
│       ├── audit_future.py
│       ├── build_gold_set.py
│       └── inspect_sample.py
├── scripts/
│   ├── smoke_test.sh
│   ├── prepare_navsim.sh
│   ├── annotate_navtrain.sh
│   ├── train_simwam_q.sh
│   ├── train_onevl_q.sh
│   └── eval_all.sh
├── tests/
│   ├── unit/                  # parsing, slots, time, codec, coordinates
│   ├── integration/           # teacher ↔ dataset ↔ policy
│   ├── leakage/               # future field access and attention masks
│   └── regression/            # baseline equivalence & checkpoint loading
├── third_party/              # pinned upstream sources
├── data/manifests/           # indices, not raw multimodal blobs
├── runs/                     # configs, metrics, checkpoints
└── artifacts/                # human review HTML, inference examples
```

`third_party` 尽量只读。对上游必须做的小修改用 `patches/` 中可重放 patch；自己加的模块都放 `src/dqt`。禁止长期直接在 vendor 内写“临时代码”。

## 4. 数据样本契约（训练与标注唯一数据格式）

### 4.1 Sample 内存对象

```python
@dataclass(frozen=True)
class DrivingSample:
    sample_id: str
    log_id: str
    timestamp_ns: int
    cameras: dict[str, list[FrameRef]]            # each frame.ts <= timestamp_ns
    ego_history: list[EgoState]                    # each state.ts <= timestamp_ns
    nav_command: str                               # left/straight/right/unknown
    observations: SceneEvidence                    # built without future data
    future_trajectory: Tensor | None               # training target ONLY [8,3]
    dqt: DecisionTrace | None
```

严格区分 `CausalSample` 与 `TrainingSample`：标注服务的入参只接受 `CausalSample`，从类型和序列化层面不可能包含 `future_trajectory`。任何 `t>0` 的 frame/track 字段在 builder 中抛错；不允许靠 prompt 文字实现“防泄漏”。

```python
@dataclass(frozen=True)
class CausalSample:
    sample_id: str
    log_id: str
    timestamp_ns: int
    cameras: dict[str, list[FrameRef]]
    ego_history: list[EgoState]
    nav_command: str
    evidence: SceneEvidence

@dataclass(frozen=True)
class TrainingSample:
    causal: CausalSample
    dqt: DecisionTrace
    target_trajectory: Tensor                # [8,3]
```

### 4.2 最小 Annotation JSON 实例（示意，并非已有模型输出）

```json
{
  "schema_version": "v1",
  "sample_id": "trainlog:timestamp:camera_set",
  "input_cutoff_ns": 1234567890000,
  "teacher": {
    "model_path": "/models/local_vlm",
    "model_revision": "sha256-of-config-or-checkpoint",
    "prompt_sha256": "..."
  },
  "answers": [
    {
      "question_id": "Q01",
      "option_code": "RED",
      "applicability": "applicable",
      "confidence": 0.96,
      "evidence": [{"type": "image_region", "camera": "front", "frame_dt_s": 0, "bbox_xyxy": [120, 35, 165, 95]}],
      "source": "teacher_multimodal"
    },
    {
      "question_id": "Q13",
      "option_code": "DIST_5_15M",
      "applicability": "applicable",
      "confidence": 0.99,
      "evidence": [{"type": "track_measurement", "track_id": "track_07", "distance_m": 12.4, "timestamp_delta_s": 0}],
      "source": "deterministic_geometry"
    },
    {
      "question_id": "Q18",
      "option_code": "NA",
      "applicability": "not_applicable",
      "confidence": 1.0,
      "evidence": [],
      "source": "trigger_engine"
    },
    {
      "question_id": "Q23",
      "option_code": "STOP",
      "applicability": "applicable",
      "confidence": 0.87,
      "evidence_question_ids": ["Q01", "Q12", "Q13"],
      "source": "teacher_decision"
    }
  ],
  "label_status": "validated",
  "audit_status": "not_run"
}
```

示例中 `model_revision` 必须在实际运行时写真实校验值，不能像示例那样保留占位符。对所有 labeled questions 存 `question_id/option_code/source/confidence/evidence`，对 decision 类另存 `evidence_question_ids`，方便后期做因果依赖诊断。

### 4.3 数据索引与缓存

- `sample_index.parquet`：`sample_id/log_id/timestamp/split/scene_flags/camera refs/target refs`。
- `evidence/*.parquet`：可计算的几何和目标事实（非 RGB 本体）。
- `labels/*.jsonl.gz`：可重跑的 DQT 输出，逐条写入，支持增量追加和失败重试。
- `gold/*.jsonl`：人工抽审；记录二次标注与一致性。
- `trajectory/*.npy` 或 dataset-native references：`[8,3]`、`frame`、`sampler`、`normalizer_id`；不要在 JSON 中反复存大数组。
- 训练时读取/解码原始图像，或预缓存视觉 feature（必须保证预缓存 encoder/processor 版本固定）。
- 对象 track ID 仅在 log 范围内唯一，统一加 `log_id` namespace，避免跨日志混淆。

## 5. NAVSIM 读取与防时间泄漏

### 5.1 `NavsimSource` 的必备方法

```python
class NavsimSource(Protocol):
    def enumerate_samples(self, split: str) -> Iterable[SampleKey]: ...
    def load_causal(self, key: SampleKey, history_s: float, cameras: list[str]) -> CausalSample: ...
    def load_future_target(self, key: SampleKey, horizon_s: float, output_steps: int) -> Tensor: ...
    def load_offline_audit(self, key: SampleKey) -> AuditEvidence: ...
```

- 训练/推理输入使用 navsim `AgentInput` 路径；`load_future_target` 只能在 offline training/eval adapter 读取。
- 图像历史范围 `[-H,0]`。如果可用频率是 2 Hz，就按真实 timestamp 采样，不伪造 10 Hz 历史帧。多视角可使用同一时刻的前/左/右/后图像，但要记录时间差与相机外参。
- 轨迹 `[x,y,yaw]` 统一为决策时刻 `t=0` 下的 ego-local 坐标；检查 x 朝前还是 y 朝前、弧度还是角度、是否包含 t=0 点、是否与 SimWAM normalizer 一致。
- `TrajectorySampling(duration=4, interval=0.5)` 的 8 点定义与 evaluator 的 0.1 s 采样严格对齐。若采用其他版本，解析实际配置并写入 metadata，不能将此当作所有数据版本的硬编码事实。
- navtrain 内做验证划分时按 `log_id` 或更严格 scene-group 拆分；公开 navtest 只用于最终模型对照评估，**绝不能从 navtest 标注挑选题目后调模型参数**；开发期评估用 navtrain 的独立 held-out log 划分。
- 训练数据需要避免当前和未来之间重复帧所导致的场景泄漏：严格以 sample timestamp 为 cutoff。

### 5.2 `EvidenceBuilder` 设计

推荐将可解析的客观事实交给程序，而不是全交给 VLM：

| 事实 | 推荐主要来源 | 什么时候调用 VLM |
|---|---|---|
| 当前 ego 速度、加速度、yaw rate | ego state | 仅语义解释 |
| 前车距离与相对速度 | GT/高可信 tracked 3D boxes + ego-local 投影 | track 模糊/对象选取 |
| 同向前车 vs 静止路边车 | 车道归属 + track velocity | 遮挡/困难关系 |
| 道路曲率、车道终止、可合法变道 | 地图/车道结构（如果可用） | 视觉冲突/地图缺失 |
| 信号灯状态、车道箭头与适用性 | 对应路口的视觉 crop、合法地图关联 | 多灯/遮挡/无法关联 |
| 行人进入横道/骑行者切入 | 时序对象轨迹 + 横道 ROI | 动作意图不确定 |
| 遮挡位置、路面异常、路障语义 | RGB/几何 | VLM 识别并保留证据 |
| Q23–26 决策 | 前述问答 + ego/nav + 交通关系 | VLM 受约束决策 |

**异常提醒**：NAVSIM 的可用结构化字段依赖具体版本及用户自己已有的检测标注；`traffic_light`, `stop_line`, `lane_graph`, `object_track` 并不保证每个 sample 都提供真值。首次 inspect 应生成 `feature_availability_report`，缺什么就按视觉模型或 `UNK` 处理，不要造字段。

## 6. 本地 Qwen 教师标注服务

### 6.1 接口与部署策略

优先提供两个互换后端：

- `OpenAICompatibleTeacherClient`：接用户当前已经部署的 vLLM/SGLang 等 OpenAI-compatible service。确认其真的支持目标 model 的多模态图片输入，不能把仅文本 API 当成 VLM。
- `HFTeacherClient`：直接 Transformers load local checkpoint + processor，较易 debug，但多 GPU 并行开销可能更大。

```yaml
teacher:
  backend: openai_compatible   # hf_local or openai_compatible
  model_path: /abs/path/to/local/model
  base_url: http://127.0.0.1:8000/v1
  api_key_env: LOCAL_VLM_API_KEY
  dtype: bfloat16
  temperature: 0.0
  max_output_tokens: 1200
  use_json_schema: true
  max_images_per_sample: 8
  image_resize_long_edge: 1024
  history_offsets_s: [-1.0, -0.5, 0.0]
  camera_ids: [front, left, right, rear]
  retry_limit: 2
  unknown_on_failure: true
  save_prompt: true
  save_model_response: true
```

上述服务端口、camera IDs 和历史偏移是**建议配置示例**；实际 NAVSIM 相机名称和帧可用性需检查后调整。Qwen `max_image_tokens`、视频输入格式取决于实际 processor。使用拼图时要保留每格的 `camera`、`timestamp` 标签和与原始 frame 的可追溯映射。

### 6.2 两轮标注，不能只用一个超长 prompt

**Round A：事实问答**

- 输入：完整的可见证据 `t<=0`、`Q01–Q22` 中触发的问题、allowed options、可验证的 bbox/track/geometry。
- 输出：每个问题一个 option code + confidence + evidence refs + unknown_reason。
- 对有数值 ground truth 的 Q13/Q14/Q03 尽可能 deterministic pre-fill，并由规则验证；避免让 VLM 主观猜米数。
- 对图像看不见的属性允许 `UNK`；禁止根据“常见情况”猜灯色。

**Round B：决策问答**

- 输入：Round A 结果 + 当前 ego 状态 + nav + 原始可见图片/历史。
- 输出：`Q23–Q26`，以及 `evidence_question_ids`；用于捕捉“红灯→减速停车”“前车制动→跟车减速”等依赖。
- 同一场景有互斥决策时允许模型输出 `uncertain` 并给出 `review_flag`，不要用多余的自由文本解释替代正式选项。

**Separate Offline Audit**：接入真实 future ego trajectory 后才判断 `teacher decision vs demonstrated motion` 一致性，例如 Q23=STOP 但未来轨迹继续加速。只做审核/挖掘、**不自动将 Q23 改成 GT 反推的动作标签**。

### 6.3 约束提示词骨架（给实现者）

```text
SYSTEM:
You are labeling a driving scene at time t=0 for a 4-second planning horizon.
Use only current and past evidence, provided ego/nav, and optional causal annotations.
NEVER assume or inspect future ego trajectories, future images, or future objects.
Pick option_code from the supplied schema for every triggered question.
If evidence is insufficient, choose UNK. If not applicable, choose NA.
For visual facts, cite camera/frame/bbox or documented observable evidence.
For decisions Q23-Q26, cite previous Q ids, not ground-truth future motion.
Return JSON only. No free-form CoT or explanatory essay.

USER:
{schema_subset}
{ego_state_t0}
{navigation}
{causal_detection_tracks}
{causal_lane_map}
{multiview_history_images}
```

Prompt 内绝不加入 GT future trajectory、future segmentation 或 PDMS。标注服务另可输出一句短解释供人审，但该解释不进入动作 token 目标。

### 6.4 `QuestionRegistry` 类型、约束与冲突处理

```python
class QuestionSpec(BaseModel):
    id: str                           # Q01...Q26
    text: str
    category: Literal["observational", "decision"]
    options: list[OptionSpec]
    trigger: str                      # a named deterministic predicate
    cardinality: Literal["single", "multiple"]
    evidence_types: list[str]

class AnswerSpec(BaseModel):
    question_id: str
    option_code: str | list[str]
    applicability: Literal["applicable", "not_applicable", "unknown", "conflict"]
    confidence: float
    evidence: list[EvidenceRef]
```

静态校验案例：

```text
Q12=no_front_vehicle  -> Q13=NA, Q14=NA
nav_command != left   -> Q18=NA unless explicit candidate left-turn
no_lane_change_intent -> Q11/Q16 can be NA (but keep if predictive risk requires them)
Q01=RED              -> require signal association to ego target lane; otherwise UNK
Q03=10-20m           -> if stop_line_distance_m exists, require 10<=d<20
Q23=STOP             -> Q26 should normally be a legal stop region (or mark conflict)
```

矛盾时不要随意覆盖教师答案：标记 `label_status=needs_review`，记录冲突原因及两种证据的来源。

### 6.5 批量标注、续跑与质量门槛

```bash
# Commands to implement in this repository, not upstream:
python -m dqt.tools.export_samples \
  --config configs/data/navsim_v1.yaml \
  --split navtrain --limit 1000 \
  --output data/manifests/pilot_samples.parquet

python -m dqt.tools.annotate \
  --samples data/manifests/pilot_samples.parquet \
  --teacher-config configs/annotation/qwen_local.yaml \
  --schema schemas/question_schema_v1.json \
  --output data/labels/pilot_v1.jsonl.gz \
  --resume

python -m dqt.tools.validate_annotations \
  --labels data/labels/pilot_v1.jsonl.gz \
  --report artifacts/annotation_pilot_report.json
```

- worker：以单个 sample 为幂等单元，`sample_id+schema_sha+teacher_sha+prompt_sha` 作为 cache key；服务限流、指数退避、失败记录、断点续跑。
- 质量：JSON 解析成功率、`UNK` 占比、触发题数量、跨帧一致性、约束冲突数、人工审核准确率、标注平均时延/显存。
- 选择 100–300 个分层人工 gold 样本（直路、红灯、排队、行人、变道、转弯、遮挡），对关键题**每题**人工确认；未通过门槛不做全量标签。
- 推荐默认验收目标（工程目标，不是已有结果）：解析成功率≥99%、无未来数据访问、人工 gold 的高置信关键问题准确率≥95% 才进入大规模实验；具体阈值按题目可观测性分组调整。
- 保留 `hard_review_queue`，将 VLM 教师不确定与几何冲突样本审阅后再发布 dataset v1。

## 7. DQT Tokenizer 与统一训练数据

### 7.1 推荐 token 字典

每个具体 QID+option 对应唯一特殊 token，例如：

```text
<DQT_START> <DQT_Q01_RED> <DQT_Q03_DIST_10_20M>
<DQT_Q12_STOPPED> <DQT_Q23_STOP> <DQT_Q24_KEEP_LANE>
<DQT_END>
```

不要把裸字母 `A B C D` 直接作为训练目标：它们在不同题里的意义不同，遇到漏题会错位且不便版本化。若确实想压缩输出序列，可以用 `<Q01><A>` 这样的 two-token schema，但是不能只输出 `A`。

```python
class DQTCodec:
    def encode(self, trace: DecisionTrace, question_mask: Tensor) -> list[int]: ...
    def decode(self, ids: list[int]) -> DecisionTrace: ...
    def validate(self, trace: DecisionTrace) -> list[SchemaIssue]: ...
```

- 按 question_id 排序；可动态省略不适用项，以 `<Q_MASK_...>` 或独立 `applicable_mask` 记录。training collator 给出正确 label mask。
- 添加专用 token 时（OneVL/Qwen 路线）必须 `resize_token_embeddings`，初始化 embedding 后保存 tokenizer 和模型 checkpoint 的绑定 hash；不要认为现有权重天然知道这些 token 的语义。
- SimWAM 不一定使用同一 LM tokenizer：通过 `question_id, option_code -> learned embedding table` 实现同构 DQT，不强行侵入视频 tokenizer 的词表。

### 7.2 训练集张量契约

```text
images/current_history:   [B, V, H, C, 3, h, w]  # or a packed image list
image_time_offsets:      [B, V, H]
ego:                     [B, D_e]
nav:                     [B, D_n]  # original command, not inferred from GT
question_ids:            [B, K]
question_option_targets: [B, K]
question_applicable_mask:[B, K]
question_confidence:     [B, K]
question_evidence:       metadata refs (not fed to deployment unless causally available)
traj_target:             [B, 8, 3]
traj_target_mask:        [B, 8]
```

建议 `K=26` 先固定槽位，以 mask 支持动态适用（Diffusion）；AR 序列可稀疏编码激活题。观测输入的 `V/H` 随模型支持而异：SimWAM baseline 保持官方前视相机输入；ONEVL/teacher 可以多相机，**必须在对照实验中报告输入模态差异**，不要把更多相机导致的增益写成 DQT 的效果。

## 8. SimWAM 适配：先 conditioning，再 joint flow

### 8.1 `SimWAMAdapter`：稳定复现原始输出

```python
class SimWAMAdapter(nn.Module):
    def encode_observation(self, inputs: CausalBatch) -> Tensor: ...
    def denoise_action(self, action_noisy: Tensor, flow_t: Tensor, context: Tensor, **kwargs) -> Tensor: ...
    def sample_trajectory(self, inputs: CausalBatch, steps: int = 10) -> Tensor: ...
```

`SimWAMAdapter` 在默认 `question_mode=none` 时应数值复现官方模型（相同 checkpoint、seed、sampler、normalizer、float tolerance）。复现未通过，不进入 DQT 实验。

### 8.2 A1 — Predicted-Q Conditioning（建议作为首个可训练版本）

```text
observation ──> existing SimWAM observation features ──> action expert (flow)
       └──> QuestionPredictor (MLP/Transformer) ──> predicted Q embeddings
                                                   └─> cross-attn/FiLM ──> action expert
```

```yaml
model:
  base: simwam
  question_mode: predicted_condition
  use_future_video_at_inference: false
  trajectory_shape: [8, 3]
  question:
    K: 26
    hidden_size: 256
    injection: cross_attention
    drop_prob: 0.15
    teacher_answer_prob_start: 0.5      # decay to 0; predicted-answer training is required
    confidence_weighted_ce: true
  train:
    freeze_vision: true
    train_action_lora: true
    train_question_head: true
  loss:
    flow: 1.0
    question_ce: 0.2
```

`0.15/0.5/0.2` 是**起始超参示例**，应按验证集调参，不是论文中已有推荐数值。训练采用逐步从真实 Q 切到预测 Q（teacher_answer_prob →0）；**验证和推理只用 predicted-Q**。联合训练过程中 Q predictor 必须只看当前 observation context，不得从未来 action target 的支路获得梯度捷径。

最简单的对照：`q_head only`（问题分类损失但不把预测答案送 action expert）与 `q_condition`（实际送入）；若两者相同，证明尚未观察到表征注入带来的额外贡献。

### 8.3 A2 — Joint Q Slots & Trajectory Slots（主实验）

对每个 Q 用独立 option embedding `E_Qi`，对 `[8,3]` trajectory 做 `Linear(3,d_model)`：

```text
Noisy question slots:    [B, 26, d_q]
Noisy trajectory slots:  [B,  8, d_a]
Shared DiT hidden space: [B, 34, d_model] (after per-group projection)
Diffusion/flow output:   q_slot_velocity + traj_velocity
Aux:                     logits per question + trajectory decoding head
```

如果基础 SimWAM action expert 假设固定 action token 序列数为 8，需要调整 position embedding、mask 和 action block routing；采用 wrapper + adapter 的方式更稳妥，**不能简单 reshape 官方预训练 positional embeddings**。

**Block attention policy：**

| Query (source) | Current observation tokens | Question slots | Trajectory slots | Future video tokens |
|---|---:|---:|---:|---:|
| Question slots | ✓ | ✓ | × | × |
| Trajectory slots | ✓ | ✓ | ✓ | × |
| Observation tokens | model-defined | optional | × | × |

这里 `Q←X` 禁止，因为 Q 需要仅由观测回答；`X←Q` 允许，保证轨迹使用问题特征。并且每一层都要保证 mask 约束，不能只在第一层设 mask。

训练示意：

```python
q0 = option_embeddings(q_label)                 # [B,K,dq], masked slots handled
x0 = trajectory_projector(traj_target)          # [B,8,da]
eps_q, eps_x = randn_like(q0), randn_like(x0)
t = torch.rand(B, device=device)
q_t = (1 - t) * eps_q + t * q0
x_t = (1 - t) * eps_x + t * x0
v_q_star, v_x_star = q0 - eps_q, x0 - eps_x
pred = joint_expert(q_t, x_t, t, current_context, q_no_x_attention_mask)
loss = w_x * mse(pred.v_x, v_x_star, traj_mask) \
     + w_q * mse(pred.v_q, v_q_star, question_mask) \
     + w_ce * masked_ce(pred.q_logits, q_label, question_mask)
```

这段仅表达 flow 的数学结构，**最终要与原 SimWAM 的时间方向、噪声参数化、loss normalization 一致**，不可把示例直接替代仓库 sampler。question logits 应从 denoising hidden state 或最终预测的 clean slot 读取，先对齐论文/代码训练目标，再设计 CE 的时刻抽样。

推理：`eps_q + eps_x` 全噪声初始化 → 经过 flow sampler 联合生成 → 解码类别及 `[8,3]`。`question_masks` 可以由仅基于当前输入的 trigger predictor 给出；最保守做法始终使用 26 slots + conditional NA masking。**推理不得使用真实标注的 mask（若 mask 本身编码了现场对象存在与否，就属于泄漏）。**

如果 joint slots 一直在 34 tokens 上无法稳定 finetune，可以先冻结原 action expert，只训练 q slot projection + LoRA，在原 8 个 trajectory slots 上插入 Q 的 cross attention，随后逐步解冻。

### 8.4 DQT 对动作有用的证据

- 在相同 backbone、相似参数量下比较 `question_mode=none`, `extra_random_slots`, `predicted_condition`, `joint_flow`。
- DQT dropout / shuffled Q / removed Q 的 PDMS 和行为影响。
- 对红灯、前车减速、路口行人等子集汇报 `Q accuracy → braking decision accuracy → trajectory metrics`。
- 对已观察到的真实 paired counterfactual scenes 做 intervention；**只替换答案 token 而不改图像**只算 sensitivity probe，不算合理场景反事实真值。

## 9. OneVL 适配：显式 DQT-first + RVQ 多尺度

### 9.1 确定模型路径：不要一开始就同时混用 latent CoT 与显式 CoT

建议两个清晰实验配置：

- **AR-Explicit（第一版）**：OneVL 的 Qwen3-VL backbone/预训练权重 → 显式自回归 DQT → RVQ trajectory codes。原有视觉/语言 latent auxiliary loss 暂时关闭或冻结，先验证选择题的价值。
- **AR-Hybrid（消融/扩展）**：保留原 OneVL latent tokens/aux decoders（必须读懂原训练代码和前向）+ 显式 DQT + RVQ；检查是否有互补增益，并报告 extra tokens 和延迟。

使用 OneVL NAVSIM MLP head checkpoint 时，不能假设它自动兼容新增 RVQ 码本：MLP head 是可选**直接轨迹对照**，而 RVQ 分支须训练或加载自己训练的 codec；保存各自 output format。

### 9.2 RVQ 轨迹编码器单独预训练

```python
class TrajectoryRVQ(nn.Module):
    def encode(self, traj: Tensor) -> RVQCodes:       # traj [B,8,3]
        ...
    def decode(self, codes: RVQCodes, levels: int) -> Tensor:
        ...                                               # [B,8,3]
    def decode_prefix(self, codes: RVQCodes, levels: int) -> Tensor:
        ...                                               # partial precision path
```

```yaml
codec:
  trajectory_steps: 8
  trajectory_channels: [x, y, heading]
  representation: ego_local_absolute
  heading_loss: wrapped_angle
  latent_dim: 128
  levels: 3
  codebook_sizes: [256, 256, 256]
  commitment_weight: 0.25
  losses: [reconstruction_l1, heading_sincos, endpoint, velocity_smoothness]
```

codec `L_codec = L1(X, X_hat) + λ_yaw L_angle + λ_fde L_endpoint + λ_v L_velocity + λ_commit L_commit`，评价：各级 `ADE/FDE/heading error/jerk/codebook usage/perplexity`；码本 collapse 或粗级误差过大必须先修好，不要寄希望于 AR 补救。若 `3×256` 不满足厘米级重建要求，则适度增大 latent/codebook、levels、分块量化；这里数值仅为参考起点。

更稳妥的最小实现可以先对全部 `[8,3]` 做单个 latent 向量 + 3 级 RVQ；若单码组表现不足，再分成 `4 个 1 s motion chunks × 3 levels`，按 chunk 顺序解码（代价是更长 AR 输出）。**记录 token 数、延迟，不能声称精度提升必然免费。**

### 9.3 AR sequence 构造与 teacher forcing

```python
sequence = (
    encode_prompt(images, ego, nav)
    + [DQT_START]
    + dqt_codec.encode(filtered_trace)
    + [DQT_END]
    + [TRAJ_COARSE_START]
    + [rvq_codes.level0]
    + [TRAJ_COARSE_END]
    + [TRAJ_REFINE_START]
    + [rvq_codes.level1, rvq_codes.level2]
    + [TRAJ_END]
)
```

- Loss mask：prompt/image 作为上下文不计算 LM label loss；DQT loss、coarse loss、residual loss 分别加权，输出 trajectory codec reconstruction loss 由预训练 codec 或联合微调计算。
- DQT token 的顺序固定；动态题以 `trigger = f(current observations)` 选择，测试时绝不能直接把 GT question mask 送入 sequence builder。
- constrained decoding：限定每个位置合法的 `<DQT_Qi_OPTION>` 和 RVQ codebook tokens，防止 Q ID 错位、非法选项、重复题和缺少 `<DQT_END>`。
- 训练初期真 DQT teacher forcing，后期可对一小部分序列使用自身预测 token 训练或混合 noise corruption；正式评估必须完整自回归 rollout，避免 oracle DQT 过高估计性能。

### 9.4 时间多尺度（V2 独立实现）

- 先输出 `4` 个未来 1 Hz 粗 key poses；使用 spline/kinematic interpolation 得到 8 点参考，再输出每个细采样位置的 RVQ residual。用**带速度约束的插值**防止停车附近出现过冲或负速度，heading 用 angle-aware interpolation。
- 可以进一步输出 40 点 residual，但需要明确 `10Hz GT` 从哪里来：若官方 train log / pose interpolation 只有 2Hz 数据，不应将插值出的 10Hz 当作真实新增监督信号。
- 第一版保留 `[8,3]` 输出、官方 evaluator 插值，保证与 SimWAM 和 OneVL 对照公平。

### 9.5 AR 训练示例配置

```yaml
model:
  base: onevl_qwen3vl
  mode: explicit_dqt_rvq
  use_onevl_latent_aux: false
  dqt_special_tokens: true
  action_codec_path: runs/rvq/best.pt
  freeze_action_codec: true
  output_steps: 8
train:
  tune_backbone: lora
  lora_r: 16
  lora_alpha: 32
  loss_weights:
    dqt: 1.0
    coarse: 1.0
    residual: 1.0
  label_smoothing: 0.0
  train_full_rollout_eval: true
```

实际 LoRA targets 依具体 Qwen/OneVL backbone 确定，不能不检查模块名直接套 `q_proj/k_proj`。

## 10. 训练与评估入口（均为待实现 CLI）

### 10.1 校验数据与训练基础模型

```bash
# (1) 标签质量与样本泄漏审计
python -m dqt.tools.validate_annotations --config configs/annotation/answers_v1.yaml
python -m dqt.data.split_guard --train data/manifests/train.parquet --val data/manifests/val.parquet

# (2) 复现 pretrained SimWAM [8,3] / NAVSIM v1
python -m dqt.eval.planning_metrics --config configs/eval/navsim_v1.yaml model=simwam_original

# (3) 训练离散 DQT 预测 head
python -m dqt.train.train_question_predictor --config configs/policies/simwam_q_condition.yaml

# (4) A1: predicted Q conditioning
python -m dqt.train.train_simwam --config configs/policies/simwam_q_condition.yaml

# (5) A2: fully joint flow
python -m dqt.train.train_simwam --config configs/policies/simwam_joint_q_flow.yaml
```

### 10.2 AR / OneVL

```bash
# (6) RVQ codec
python -m dqt.train.train_rvq --config configs/codec/rvq_8x3.yaml

# (7) OneVL + DQT + RVQ
python -m dqt.train.train_onevl --config configs/policies/onevl_dqt_rvq.yaml

# (8) 使用统一 NAVSIM evaluation pipeline
python -m dqt.eval.planning_metrics --config configs/eval/navsim_v1.yaml model=onevl_dqt_rvq
```

每个训练入口至少输出 `config_resolved.yaml / git_sha / seed / data_manifest_sha / schema_sha / checkpoint / metrics.jsonl / sample_visualizations/`。统一 logging interface，以便进行对照。

## 11. 关键单元测试和验收条件

| 测试 | 具体检查 | 未通过的处理 |
|---|---|---|
| `test_causal_cutoff` | 任意输入含 `t>0` 图像/track/pose 均被拒绝 | 阻止标注启动 |
| `test_target_visibility` | TeacherClient 序列化结果不含 future trajectory 或路径 | 阻止标注启动 |
| `test_question_roundtrip` | 26 题 encode/decode/re-encode 完全一致，UNK/NA 保留 | 禁止发布 schema |
| `test_evidence_refs` | 每条图像 bbox 合法、frame timestamp 非未来、track 可查 | 进入 review |
| `test_schema_version` | checkpoint 与 question schema/tokenizer hash 一致 | 拒绝加载 |
| `test_coordinate_roundtrip` | ego-local ↔ world、yaw wrap、时间插值回环误差 | 禁止评估 |
| `test_navsim_baseline_parity` | 无 DQT 时 SimWAM adapter 在固定 seed/dtype 下复现基线 | 暂停方法开发 |
| `test_attention_no_future_video` | Action/Q slots 无法 attention 读取未来 video token | 阻止训练 |
| `test_q_no_action_leak` | Q slots 任意层不能读 noisy action slots | 阻止联合 flow |
| `test_policy_no_oracle_q` | 推理路径不调用 label lookup | 阻止上线评估 |
| `test_dynamic_trigger_causal` | Question trigger 不读取 GT/未来 trace | 阻止评估 |
| `test_rvq_prefix_decoding` | 每一级 RVQ 可独立解码且 codec 输出正确 shape/单位 | 禁止 AR 训练 |
| `test_ar_grammar` | 合法 Q+RVQ tokens 顺序、EOS、错位处理 | 禁止推理 |
| `test_chunk_continuity` | 分段轨迹连接、速度/yaw continuity | 禁止时间多尺度 |
| `test_train_eval_modality` | 对照实验传感器模态/历史长度一致 | 标记不可直接归因 |
| `test_log_split_isolation` | train/val/test 的 log/scene-group 不交叉 | 阻止指标发布 |

**重要**：NAVSIM 官方 metric/collision/comfort 评估只在**预测轨迹已生成后**使用。它是模型评估或部署安全模块，不作为 DQT 标注问题的前置条件；不要回到“先有鸡还是先有蛋”的错误。

## 12. 评价协议与消融矩阵

### 12.1 固定对照条件

- 统一 NAVSIM v1 split、sensor config、路径与 normalization、4s trajectory horizon、官方 PDMS implementation；所有 SimWAM 变体与其基线采用相同视觉输入。若扩展 multi-view，按 `front-only` 和 `multi-view` 分组对照。
- 同样报告输出 sampler steps、平均/分位推理时延、参数量、峰值显存；AR 需要特别报告 DQT token 个数与 RVQ level 数带来的 token latency。
- 不只报告总 PDMS；保存 metric components、场景 subgroup、ADE/FDE、classification F1、问题—轨迹一致性。
- 至少 3 个 random seeds 做重要 ablation，报告平均与波动，避免偶然 checkpoint 起伏。

### 12.2 首要实验表

| Run ID | 基础模型 | DQT | 轨迹输出 | 要回答的研究问题 |
|---|---|---|---|---|
| S0 | SimWAM official | 无 | 官方 flow `[8,3]` | baseline 能否复现 |
| S1 | SimWAM+same-param | 无 | flow `[8,3]` | 增参是否已足够 |
| S2 | SimWAM+Q-head | CE 监督不注入 | flow `[8,3]` | 多任务监督是否足够 |
| S3 | SimWAM+Q-cond | predicted Q injection | flow `[8,3]` | 语义决策条件是否增益 |
| S4 | SimWAM+JointQFlow | noisy Q slots jointly denoised | flow `[8,3]` | 联合表征是否真正有效 |
| S5 | SimWAM+shuffledQ | 错位/随机 Q | flow `[8,3]` | 排除空槽增益 |
| O0 | Original OneVL | 无新增 Q | 官方输出 | 原 baseline 能否复现 |
| O1 | OneVL+RVQ | 无 Q | 3-level RVQ | 单独 codec 的贡献 |
| O2 | OneVL+DQT | Q→直接动作 | 官方/直接动作 | 先推理是否有收益 |
| O3 | OneVL+DQT+RVQ | Q→3-level RVQ | 粗到细 | 主方法 |
| O4 | OneVL+DQT+time-scale | Q→时间尺度→细化 | 时间多尺度 | V2 扩展 |
| `oracle-Q` | 各模型 | 人审/教师 GT Q | 对应轨迹输出 | **只看上界，不并入正常分数** |

### 12.3 事件挖掘与可解释性 dashboard

从 `DQT gold set` 和 deterministic scene facts 创建 `red_light`, `front_vehicle_braking`, `crosswalk_pedestrian`, `cut_in`, `left_turn_yield`, `occluded_area` 子集。 dashboard 应显示：

```text
sample id / split / camera frame
  ├─ teacher Q / gold Q / model Q (per-question)
  ├─ evidence bbox/track/lane/stop_line (with confidence)
  ├─ reference trajectory / baseline prediction / DQT prediction (BEV)
  ├─ key constraints: signal, lead vehicle, crosswalk
  ├─ collision/PDMS output (only after trajectory is predicted)
  └─ failure label: perception / decision / generation / annotation dispute
```

若 question prediction 正确但轨迹仍违背要求，归类为 **planning failure**；若根本看错红灯，则是 **perception failure**；若标注与图像矛盾则是 **annotation failure**。这才是使用选择题改善解释性的直接收益。

## 13. 研究风险和应急路径

| 风险 | 早期信号 | 处理 |
|---|---|---|
| 标注 VLM 幻觉 | 高置信答案与 bbox/地图明显矛盾 | deterministic prefill、crop 回审、UNK/review |
| Q23–26 成为 GT 轨迹反推捷径 | 不给轨迹就不能判断行动 | 因果教师隔离，GT 只单独审核 |
| 选择题太多且信息稀疏 | 大量 NA、重复答案 | trigger、分层 priority、top-k 关键 Q |
| 仅加 token 就有增益 | shuffled Q 也变好 | same-size/parameter/mask 控制 |
| Diffusion question branch 被忽略 | Q ablation 对轨迹无影响 | 调整 cross attention、联合训练、信息流约束 |
| Joint Flow 不稳定 | q slot CE 波动大、类别 collapse | 退回 S3 predicted Q conditioning，分步解冻 |
| 标签错误在 AR 中累积 | Q 误差使 RVQ 轨迹显著变坏 | DQT corruption、保留直接动作对照 |
| RVQ 精度损失过大 | codec ADE/FDE 显著高于原数值表示 | 优化 codec，改变 codebook/chunks，再训练 AR |
| AR 时延变长 | Q 和 RVQ token 过多 | 动态题/小词表/减少 levels、可选 hybrid latent |
| 未来数据/GT 泄漏 | navtest 性能异常高、特权字段出现在 inputs | type-boundary、静态审计、严格 held-out logs |
| 只有开环优势 | PDMS 上升但事件错误不下降 | 强化风险子集，补 NAVSIM v2 pseudo closed-loop |
| Base code API 演进 | 无法加载 checkpoint 或 args 更改 | vendor lock + adapter regression tests |

## 14. 执行顺序与完成定义（不按固定天数安排）

### Milestone A — 数据可验证

- [ ] `sample inspector` 能画出原始 RGB / tracks / ego / GT trajectory（GT 仅在离线 inspector 中显示）。
- [ ] `CausalSample` 任何情况下不能包含未来 GT。
- [ ] 26 道 QuestionRegistry schema 固定、tokenizer round-trip 通过。
- [ ] 1k 样本成功标注、人工 gold 集可审、错误能够分类。
- [ ] 单次标注可重现，schema 和 teacher hash 写入 manifest。

### Milestone B — Diffusion baseline 和最小收益

- [ ] 原始 SimWAM 在 NAVSIM v1 一次完整评估可复现。
- [ ] Q predictor 达到可用准确率；test 只用 predicted Q。
- [ ] S0/S1/S2/S3 在完全匹配输入和训练预算下比较。
- [ ] 若 S3 优于 S2，再进行 joint Q slots (S4)；若无改进先诊断标签质量与条件注入。

### Milestone C — AR baseline 和多级动作

- [ ] OneVL 原始模型/权重/推理接口跑通 NAVSIM v1。
- [ ] RVQ codec 实测 prefix reconstruction 与正常化精度。
- [ ] O0/O1/O2/O3 进行全量自回归推理对照，不使用 teacher Q。
- [ ] 扩展时间多尺度之前先报告 RVQ AR 的 latency 和分项规划表现。

### Milestone D — 论文/报告可复现

- [ ] 所有实验记录 upstream SHA、代码 commit、data split、seed、normalizer、tag schema、teacher 和 checkpoint hash。
- [ ] 总体/子集结果表、Q 预测 confusion matrix、错误分类图、BEV 案例、counterfactual probes。
- [ ] Oracle Q 和 predicted Q 明确分开报告。
- [ ] 明确开放环 NAVSIM v1 的局限，若声称闭环改进需额外真实闭环环境证明。

## 15. 推荐第一批跑通的真实样本与程序断言

| 场景 | 必答事实题 | 预期可检查决策题 | 关键断言 |
|---|---|---|---|
| 红灯且停止线可见 | Q01/Q03 | Q23=制动停车，Q26=合法停车区 | Q01 证据定位正确；不输入未来轨迹 |
| 绿灯但前车静止 | Q01/Q12/Q13/Q14 | Q23=减速/停车 | 绿灯不导致盲目加速 |
| 前车突然减速 | Q12/Q13/Q14 | Q23=减速 | 历史 tracker 判定减速，不靠未来轨迹 |
| 右转遇行人 | Q04/Q06/Q19 | Q25=让行行人，Q24=右转意图 | 保持导航右转意图与当前停车行为兼容 |
| 目标车道后车快速接近 | Q07/Q11/Q16 | Q24=保持/推迟变道 | 识别后车风险，不靠距当前最近对象 |
| 遮挡人行横道 | Q04/Q21 | Q23=谨慎减速，Q25=等待确认（若需要） | 不把看不见当作没有行人 |

这些例子用于 annotation/policy 的集成测试和人工审阅，不应以严格字面规则把所有真实场景自动判为唯一正确。尤其停车行为仍需依据速度、停止距离和当地规则确定。

---

## 16. 官方参考

- NAVSIM agents (AgentInput, Agent/TargetBuilder, 4s evaluation): <https://github.com/autonomousvision/navsim/blob/main/docs/agents.md>
- NAVSIM split policy: <https://github.com/autonomousvision/navsim/blob/main/docs/splits.md>
- SimWAM repo / training configs / weights: <https://github.com/H-EmbodVis/SimWAM> ; <https://huggingface.co/H-EmbodVis/SimWAM>
- OneVL (original latent reasoning, inference paths): <https://github.com/xiaomi-research/onevl>
- OneVL training repo linked by its README: <https://github.com/GeorgeLuImmortal/OneVL_training> (availability to verify locally)
- LightNav-0 RVQ method: <https://github.com/lightorigins/LightNav-0>
- Qwen3.5 official models: <https://huggingface.co/collections/Qwen/qwen35>

**成功标准不是“回答了多少道题”，而是：标签能够被证据验证；训练/推理都不偷看未来；相较参数匹配基线，真实选择题对具体驾驶场景的轨迹决策产生可复现、可解释的帮助。**
