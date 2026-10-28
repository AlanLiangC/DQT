# DQT: Driving Question Tokens for Decision-Grounded Trajectory Generation

> **Method Description / 研究方法说明（v0.1）**  
> 目标场景：NAVSIM v1，未来 4 s 轨迹规划；两条验证路线：SimWAM（Flow/Diffusion Action Expert）与 OneVL（AR + coarse-to-fine trajectory tokens）。  
> 研究定位：**一套可验证的结构化驾驶判断数据，两个决策—轨迹联合表征方案**，而不是生成轨迹后的事后解释。

## 0. 摘要

现有端到端驾驶模型往往将丰富的多模态输入映射到低维、纯数值的自车轨迹。这样的动作接口容易形成表征不对称：视觉、道路规则和交互证据通过大量特征 token 表达，而输出只有少量连续坐标，缺少与视觉可验证事实对齐的中间决策变量。我们提出 **Driving Question Tokens (DQT)**：将会直接影响短时规划的具体驾驶问题（如“控制本车的灯是否为红色”“前车是否减速”“右转横道是否有人”）离散成有固定语义、可审计的选项 token。通过对现有检测/跟踪、历史 RGB、多相机、地图与自车状态的结构化利用，再由视觉语言模型进行受约束标注，建立 **观测→驾驶证据→决策→轨迹** 的监督链条。在 Flow/Diffusion 路线中，我们让 DQT 槽与连续轨迹槽交互，研究在动作空间中引入语义维度能否提升运动生成；在 AR 路线中，模型先自回归生成紧凑 DQT 决策序列，再通过粗到细的 RVQ/多尺度动作 token 生成连续轨迹。两条路线共享同一标注定义、数据划分与评估流程，能够分别检验结构化驾驶知识对数值规划和自回归规划的贡献。

**English summary:** We introduce Driving Question Tokens (DQT), a set of visually grounded and auditable discrete decisions, to bridge multimodal driving perception and continuous short-horizon trajectory generation. We construct question-answer annotations from causal observations with structured evidence and use the resulting codes in two complementary planners: joint question–trajectory flow generation based on SimWAM, and question-first coarse-to-fine autoregressive planning based on OneVL. Crucially, answers are generated from currently available inputs rather than inferred from future ground-truth trajectories, preventing hindsight leakage. Evaluations distinguish perception correctness, decision correctness, trajectory quality, and counterfactual sensitivity.

## 1. 核心问题和可证伪假设

**P1 — 表征不对称。** 大量视觉/状态 token 被压缩成 `[8,3]` 数值轨迹。我们不预设“token 数多就是好”，而是验证：引入与行动有因果关系的离散决策变量是否比纯扩大 action hidden size 更有效。

**P2 — 缺少视觉可验证的动作语义。** 数值轨迹描述 *how*，并未显式表达 *why*。将“红灯、停车线、前车减速、行人穿越”等具体证据绑定到动作意图，能够支持按问题诊断，并形成可能可干预的行动接口。

**H1（表征）**：在参数/算力受控时，DQT + trajectory 比等维随机/可学习空槽、仅增加轨迹宽度更有效。

**H2（监督）**：由当前观测得到的客观选项具有可验证准确度，能够改进刹停、跟车、让行、变道等事件子集，且改进不仅来自额外参数。

**H3（生成）**：将 DQT 作为被生成的对象（而非推理期注入 GT 答案）可增强模型的决策一致性。

**H4（AR）**：`answer tokens -> coarse action codes -> residual action codes` 相比直接输出轨迹，能在可接受时延下提升规划和可解释性。

> 重要边界：回答选项不等于真正的自由文本 CoT；它是 **structured decision trace / structured rationale**。若论文采用 CoT 一词，需清楚注明“structured CoT”且不暗示模型输出了人类完整推理过程。

## 2. 任务定义与输入/输出

设决策时刻为 `t=0`。合法推理输入为：

\[
C_0 = \big\{I_{[-H,0]}^{1:\,V},\; S_{[-H,0]},\; N_0,\; M_0^{\mathrm{allow}},\; D_{[-H,0]}^{\mathrm{allow}}\big\}.
\]

- `I`：当前及历史多相机 RGB；任何 `t>0` 图像不得作为模型输入或问答标注证据。
- `S`：当前/历史 ego pose、velocity、acceleration 等；`N`：NAVSIM left/straight/right/unknown 导航指令；注意其 left/right 可能兼指弯道、变道和转弯。
- `M_allow`、`D_allow`：若使用地图/检测/跟踪，则必须在训练、验证、部署时定义一致的可用范围。检测 GT 可以作为**训练标注辅助证据**，但不能未经说明在测试推理输入中注入 GT。
- `Y`：未来真实自车轨迹，只用于轨迹训练目标与独立质量审核；**禁止用于生成视觉事实答案**。

统一规划输出使用参考配置 `X_0 ∈ R^{8×3}`（2 Hz 采样、4 s、`[x,y,heading]`），与 SimWAM 官方 NAVSIM 预训练接口对齐。NAVSIM v1 的 PDM 评估在 **4 s / 10 Hz** 上进行，轨迹通过官方 `TrajectorySampling` 与插值接口送评估。可选择研究直接预测 `40×3`，但必须独立报告，且避免与预训练 checkpoint 接口混淆。

决策问题为 `Q={q_i}`，每个有有限选项 `A_i={a_{i1},...}`、触发条件、证据、置信度、缺失原因：

\[
Z_Q=\{(i,a_i,m_i,c_i,e_i)\}_{i=1}^{K},
\]

其中 `m_i∈{applicable, not_applicable, unknown, conflict}`；`c_i` 为标注置信度，`e_i` 为 evidence references。**未知≠不存在，不适用≠否定**。

## 3. 问题体系：具体、可观察、和动作强相关

**原则：一问一个可核验事实；尽可能使用直接对象、属性、距离、状态，而非“是否安全”“交通规则是什么”等抽象结论。** 采用固定 Question ID，不靠 A/B/C 本身推断语义；选项变化必须升级 schema 版本。

### 3.1 核心可观测问题（建议 v1 全题库保留 22 道，样本动态触发）

| ID | 具体问题 | 离散选项（简写，程序中有稳定 code） | 主要证据 | 触发 |
|---|---|---|---|---|
| Q01 | 前方**控制本车预期通行方向**的信号灯是什么状态？ | 红/黄/绿/闪烁/无/未知；箭头按目标方向解释 | 前视图、路口几何 | 有信号路口 |
| Q02 | 前方是否存在本车需遵守的 STOP / YIELD 标志？ | STOP/YIELD/无/看不清 | 前视图、地图 | 有路口/汇入 |
| Q03 | 相关停止线距当前车头多远？ | 0–10/10–20/20–40/>40 m/无或未知 | 地图+自车位姿/视觉 | 信号或停止控制 |
| Q04 | 预期通道上的人行横道是否有行人正在通行/进入？ | 无/正在通过/准备进入/已离开/遮挡 | RGB+检测/跟踪 | 路口/横道 |
| Q05 | 前方路口**出口空间**是否被堵住？ | 未堵/部分/堵死/未知 | RGB+检测/地图 | 将进入路口 |
| Q06 | 转弯路径的冲突区被谁占据？ | 无/对向车/横向车/行人骑行者/多个 | RGB+目标跟踪 | 转弯 |
| Q07 | 当前车道是否允许执行导航方向？ | 允许/需左变道/需右变道/当前无法合法执行/未知 | 导航+地图+车道标识 | 全部 |
| Q08 | 当前车道前方是否结束、合流或分流？ | 延续/结束/左合流/右合流/分流/未知 | 道路结构/地图 | 全部 |
| Q09 | 本车道前方几何形态？ | 直/左缓/左急/右缓/右急/未知 | 车道/地图 | 全部 |
| Q10 | 本车道是否被静态障碍阻断？ | 无/左局部/右局部/中央/全部/未知 | RGB+检测 | 全部 |
| Q11 | 需要变道时目标车道是否有可用空间？ | 左/右/左右均有/均无/未知 | 相邻车道+动态对象 | 变道候选 |
| Q12 | 当前车道正前方是否有车？ | 无/行驶/减速停车/已静止/未知 | 目标检测/跟踪 | 全部 |
| Q13 | 最近相关前车纵向距离？ | <5/5–15/15–30/30–60/>60 m/无或未知 | BEV目标位置 | Q12≠无 |
| Q14 | 前车当前速度趋势？ | 加速/匀速/减速/急减速/静止/未知 | 历史 track + 车灯佐证 | Q12≠无 |
| Q15 | 相邻车是否正在切入或准备切入当前车道？ | 无/左侧/右侧/两侧/未知 | 多帧轨迹、转向灯 | 有相邻对象 |
| Q16 | 目标车道后侧是否有快速接近车辆？ | 无/左后/右后/两侧/未知 | 后视/侧视+跟踪 | 有变道候选 |
| Q17 | 路口是否有横向来车接近冲突区域？ | 无/左/右/双向/遮挡 | 多视角+跟踪 | 有路口 |
| Q18 | 左转/掉头时对向车辆是否接近或占用冲突区？ | 无/接近/占用/已通过/未知 | 前视+跟踪 | 左转/掉头 |
| Q19 | 是否有行人位于或即将进入本车预期通道？ | 无/正前/左前/右前/多处/未知 | RGB+检测 | 有行人/遮挡 |
| Q20 | 是否有骑行者可能进入预期通道？ | 无/同向/准备横穿/正在横穿/可能切入/未知 | RGB+检测/历史 | 有骑行者 |
| Q21 | 是否存在会影响决策的遮挡位置？ | 无/横道/路口来车方向/车道前方/多处 | 多视角、深度几何 | 全部 |
| Q22 | 路面是否有可见的异常附着/通行情况？ | 正常/积水/冰雪/砂石/严重破损/未知 | RGB+其他合法传感器 | 全部 |

**不要过度离散：** 距离、相对速度、停止线坐标、对象位置仍以连续数值保存在 evidence / geometry 字段，Q13 只是可审计语义标签，不能用距离档位代替精准制动规划。对缺少可靠监督的灯色/地图/遮挡等设置 `unknown`，不能强填“无”。

### 3.2 决策题（不是图像事实，应由因果输入推出）

| ID | 问题 | 选项 | 教师标注依据 |
|---|---|---|---|
| Q23 | 未来 4 秒的主要纵向行动？ | 加速/匀速/缓减速/明显减速/停车/保持静止 | Q01–22 + ego + 导航；未来 GT 仅独立审计 |
| Q24 | 未来 4 秒的主要横向行动？ | 车道保持/左变道/右变道/左转/右转/车道内避障 | Q07–11 + 导航 + 环境 |
| Q25 | 当前需要等待什么才可继续？ | 不等待/信号/前车/行人骑行者/其他车辆/视野和道路条件 | Q01–22 + 场景语义 |
| Q26 | 未来 4 秒的目标**区域**？ | 本车道前方/目标车道/转弯后道路/停止线前/前车后方/障碍前安全区 | 几何目标+交互约束；精确坐标由轨迹解码器生成 |

多目标/多约束情形：Q04/Q06/Q15 等题可设计为 multi-label 的**结构化子实例**，每个 `object_id` 一组答案；首次实验为保持小词表，可采用预定义 priority 选项 + `multiple`，同时存储对象级附加 annotation，不要丢掉对象的真实数量。

### 3.3 Tokenization 与有效性规则

单选采用**独立于自然语言的固定原子符号**，如 `<DQT_Q01_RED>`、`<DQT_Q14_DECEL>`、`<DQT_Q23_STOP>`、`<DQT_Q04_NA>`、`<DQT_Q01_UNK>`；多选按稳定顺序输出若干 token 并用 `</DQT_Qi>` 结束。训练和推理共享固定字典。

- `UNK`：适用但看不清/证据不足；`NA`：当前场景不适用；`CONFLICT`：多个来源矛盾需复核。
- `question_mask`：仅对 applicable 且通过质量阈值的答案计入监督；不适用项可生成 `<..._NA>` 或动态省略，但同一实验配置只能选一种。
- `question_order`：规则和路权 → 前车/对象运动 → 横道/遮挡 → 驾驶动作 → 目标区域；根据场景只激活相关题，固定排序，确保可解析。
- `confidence`：标注置信度用于权重或过滤，不作为推理期 GT 特权输入。
- 推理期由模型自行预测答案；**禁止把测试样本的 teacher / GT 选项输入规划器**（Oracle 仅作为单独诊断实验）。

## 4. 可靠标注：当前证据与未来监督严格隔离

### 4.1 两个互不串通的标注通道

**A. Causal Perception & Decision Labeler（正式 DQT 标签）**：输入 `t<=0` 的多相机图像/历史图像、ego state、导航、**允许的**检测/跟踪/道路结构证据，以及上述问题定义。由本地视觉语言教师按约束 JSON 输出。Q01–22 只根据当前和历史观测答题；Q23–26 也仅使用这些输入和前面回答推导驾驶选择。

**B. Hindsight Auditor（离线审核，不作为视觉事实标注来源）**：可访问 `t>0` GT ego trajectory、未来场景/规则等（如果数据存在），检查 A 标注与真实驾驶轨迹的兼容程度、发现问题、打标 `consistent/ambiguous/disagree`。不得把未来 GT 反灌进 A 的事实答案；不应把 GT 轨迹当作唯一正确行为（人类驾驶也可能有次优或违法行为）。

这样可以避免红灯是否存在、前车当前是否减速等答案被未来轨迹反向“猜”出来，也使训练/推理使用相同可见证据。可另做 `oracle_action_from_GT` 研究，但必须独立储存并且仅用于上界分析。

### 4.2 约束式标注协议

1. **自动提取**：轨迹片段索引、对象 track、3D/BEV 位置、前车距离与相对速度、车道结构、ego 当前状态等，生成 `evidence bundle`。没有 GT 红灯/停止线时必须走视觉/地图识别或 `unknown`。
2. **教师多模态判断**：对多帧/多视角输入进行 26 道题的按需选择，模型只能从允许的 `option_codes` 中输出。教师被告知“不能猜测、不能用未来结果反推”。
3. **确定性规则校验**：选项合法性、触发关系、`front_vehicle=none -> front_vehicle_distance=NA`、`turn=none -> oncoming_left_turn=NA`、信号灯与箭头关联、长度单位等。
4. **多证据冲突检验**：运动状态基于 tracks 的数值计算优先，像素可见属性以图像为主；冲突保留 `CONFLICT`，不任意合并。
5. **审核与回流**：低置信度、风险场景、规则冲突、教师多次不一致样本人工复核并建立 gold set，校正 prompt 或确定性算法。
6. **版本冻结**：每个标签保存 schema 版本、模型 hash、prompt hash、输入 checksum、答案、证据、来源、置信度、审核结果；保证训练可追溯。

**教师模型核对**：官方 Qwen3.5 模型列表中未确认存在官方名为 `Qwen3.5-72B` 的发布项；本地可能是自定义目录、微调版或其他 Qwen 系列 72B。工程只要求 `model_path` 可配置，运行前读取 `config.json` 与 processor，确定其真能接收多图/视频；**不能仅凭“72B”假设具备视觉能力**。

## 5. 方法 A：Joint DQT–Trajectory Flow Policy（基于 SimWAM）

### 5.1 为什么不是简单地“把 ABCD 拼到轨迹坐标后面”

轨迹是连续变量，DQT 是离散类别。若直接将类别 ID（例如 A=0、B=1）当作数值一同添加高斯噪声，模型会被迫学习虚假的选项数值顺序；类别 0 到 1 的距离没有实际物理意义。

我们实现两个由简到难的可复现变体：

**A1 — Predicted DQT Conditioning（低风险强基线）**：从当前观测预测 Q，采用可训练 embedding `E(Q)`，用 cross-attention/FiLM/adaptive layer norm 将其注入动作 expert，不改变轨迹的 flow matching 目标。训练前期可用 GT Q，随后逐渐采用模型预测 Q，确保推理时无需 oracle 答案。

\[
\hat Q = g_\phi(C_0),\quad H_Q=\operatorname{Embed}(\hat Q),\quad
\dot X_\theta = f_\theta(X_t,t,C_0,H_Q).
\]

`L = L_flow(X) + λ_q L_CE(Q)`。A1 能直接检验 Q 的语义作用，但不属于严格联合扩散。

**A2 — Joint Continuous Question Slots + Trajectory Slots（核心主张）**：把每道激活问题的类别通过**独立 codebook embedding**映射到连续潜空间 `U_0 = E(Q)`，为问题和轨迹分别准备 token slots：

\[
Z_0 = [\underbrace{U_0}_{K\times d_q};\underbrace{P_X(X_0)}_{T\times d_a}],
\quad Z_t = (1-t)\varepsilon + tZ_0,\quad \varepsilon\sim \mathcal N(0,I).
\]

其中 `P_X` 为动作投影；如果两组维数不同，采用独立投影/噪声调度并在共享 hidden space 交互，不能强行按 `[T, 3+K]` 拼成物理轨迹。训练采用 velocity/flow matching（与 SimWAM 约定的时间方向统一）：

\[
\mathcal L_{\mathrm{joint}}
 = \lambda_x\mathbb E\|\hat v_X-v_X^*\|^2
 +\lambda_u\mathbb E\|\hat v_U-v_U^*\|^2
 +\lambda_{ce}\sum_i m_i\operatorname{CE}(\hat p_i,a_i).
\]

- **关键 attention 约束：** question slots 可以读当前观测与自身 noisy question slots，但**不能读取 noisy trajectory slots**，以免训练时从 GT 轨迹反推选项；trajectory slots 可以读取 question slots 与当前观测。
- 推理时 **Q slots 和 X slots 均从独立噪声开始**，一起迭代去噪，最终解码 `Q_hat` 和 `X_hat`。不能在推理时喂教师 Q。
- 问题类别读取使用分类头 `softmax(W_i h_i)`；continuous codebook 仅作为可学习目标，不把类别字母的顺序当作几何数值。
- 为保证语义 slot 对轨迹确实有影响，加入 question dropout、counterfactual swaps（只在物理一致的情形）与 loss/gradient 监测；评估要检查模型是否忽略 Q。
- 对相同证据的联合去噪不一定天然获得因果可解释性，需做跨数据、反事实和扰动测试。

**起步建议**：先 A1 跑通，再 A2。若 A2 相对 A1 没有稳定增益，可保留 A1 作为更简洁方案，并如实调整论文主张。

### 5.2 与 SimWAM 原始模型的接入

- 保留 SimWAM 已训练的 **action expert、输入编码、归一化与 `[8,3]`**，初始化 action 分支权重；增加 `QuestionPredictor/QuestionSlotEmbedding/QuestionTrajectoryMixer`。
- 训练阶段优先 `action-only` 微调；可选少量 LoRA action blocks。暂不重新训练数十亿参数的视频 expert，便于明确增益来自 DQT；后续才尝试恢复联合 video–action loss。
- 遵循原代码中关于 action 与未来视频 token 隔离的约束。**DQT 不得通过未来视频 latent 泄露**。
- 正则：源模型轨迹输出一致性（可选）、轨迹 flow、分类、轻量 Q–action 对齐；权重通过验证集调整，不能先入为主地加很复杂的损失。
- 比较 `same-width-extra-traj-slots`、`random-semantic-slots`、`shuffled-question-answers`，控制参数与序列长度。

## 6. 方法 B：DQT-first Coarse-to-Fine AR Policy（基于 OneVL）

### 6.1 生成序列

将现有 OneVL VLM 的场景输入作为前缀：

```text
<OBS> multi-view / RGB history / ego / navigation </OBS>
<DQT>
  <DQT_Q01_RED> <DQT_Q03_10_20M> <DQT_Q12_STOPPED> ...
  <DQT_Q23_STOP> <DQT_Q24_KEEP_LANE> <DQT_Q26_BEHIND_FRONT>
</DQT>
<TRAJ_COARSE> <C0_...> <C1_...> ... </TRAJ_COARSE>
<TRAJ_REFINE_1> <R1_...> ... </TRAJ_REFINE_1>
<TRAJ_REFINE_2> <R2_...> ... </TRAJ_REFINE_2>
```

DQT 在轨迹**之前**生成，因此任何被解释为决策证据的 token 都不能看到未来轨迹 token。

### 6.2 多尺度轨迹 tokenizer（严格区分两种“多尺度”）

**V1（最快验证，参考 LightNav-0 的 RVQ 精度多尺度）**：先将原始 `[8,3]` 归一化轨迹编码成 `z=E_τ(X)`，再经逐级残差量化：

\[
q_1=\operatorname{VQ}_1(z),\quad
q_{l}=\operatorname{VQ}_{l}\left(z-\sum_{j<l}e(q_j)\right),\quad
\hat X=D_\tau\left(\sum_l e(q_l)\right).
\]

先输出 coarse code，再输出多个 residual code，每一级都能解出越来越精确的 4 秒路径。初版建议 `L=3`、每层 codebook 256（都是**待验证超参**，并非宣称在驾驶数据上已有效），由重建质量选择码本容量。

**V2（真正的时间分辨率多尺度，作为扩展）**：先输出较稀疏的 1 Hz 关键点/粗轨迹，再生成 2 Hz（或 10 Hz）细节残差。时间尺度可选 `4 waypoints -> 8 waypoints`，10 Hz 由官方插值或另一个专用 40 点 residual decoder 生成。**这是本研究对 LightNav-0 的扩展，不应描述为 LightNav-0 原方法。**

V1 优先保证兼容已有规划器与评估；V2 单独报告效果、长度和延迟。

### 6.3 训练目标

\[
\mathcal L_{AR}=
\lambda_q\mathcal L_{DQT}^{AR}
+\lambda_c\mathcal L_{coarse}^{AR}
+\lambda_r\mathcal L_{residual}^{AR}
+\lambda_{rec}\mathcal L_{traj\,rec}
+\lambda_{aux}\mathcal L_{OneVL\,aux}.
\]

- **先独立训练 trajectory tokenizer**（所有轨迹统一坐标系、单位与 normalizer），测试重建上限，再冻结 codec 训练 AR。
- 使用 teacher forcing 训练 `DQT -> motion codes`；推理时完整 rollout 均使用模型自己预测的 DQT，不再喂 teacher tokens。
- 在训练中加入少量 DQT corruption / predicted DQT 自喂策略，减轻推理误差传播；但不能使错误判断被默许为真相。
- OneVL 原方法偏重 **latent reasoning + visual/language auxiliary decoders**，并非“天然就是显式选择题再输出 RVQ”。建议在其 backbone/training framework 中新增显式 DQT 输出与 trajectory tokenizer，作为**明确的结构性改造**。保留其原 latent aux 可作为独立 ablation；避免混称为原 ONEVL 推理流程。
- Q 的质量由分类准确率与证据定位测量，不要用“生成了选项 token”直接证明模型真的在推理。

### 6.4 决策—动作一致性

定义可执行的一致性检查（用于**训练样本质检、离线评估或辅助损失**；不是在没有轨迹时用题目先检查未来轨迹）：

- `Q01=RED` 且该信号确实控制本车方向时，预期轨迹不越过适用停止线（还需考虑停止能力与已过线等场景）。
- `Q23=STOP` 时，4 s 末速合理接近 0，且前方有足够制动距离；不能仅凭位置点推断速度，必须从轨迹或速度接口计算。
- `Q24=LEFT_LANE_CHANGE` 时，轨迹末端落入合法目标车道（如果 4 s 内动作尚未结束，应标记阶段性执行）。
- `Q25=YIELD_PEDESTRIAN` 时，不占用预测有行人的横道冲突区。

一致性是条件约束，需知道信号归属、停止线位置、路权、物理能力；不能将几条模板当作完整交通法规。

## 7. 训练、推理与“鸡生蛋”问题的解法

**整个系统没有“先生成轨迹，再用轨迹定义问题答案”这个循环**：

```text
训练标注：Causal observation ──> DQT teacher ──> DQT labels
           当前/历史图像 + ego + nav + 合法标注

训练规划：Causal observation + DQT supervision + GT future trajectory
         ──> policy training

推理 Diffusion：Causal observation ──> model-generated DQT slots + trajectory slots
推理 AR       ：Causal observation ──> DQT tokens ──> trajectory codes ──> trajectory

离线评估：预测的 DQT、预测轨迹分别与独立标签、轨迹/模拟评测对比
```

教师标签与未来轨迹只在训练监督/审核阶段存在。部署时只有当前和历史信息。问答输出在轨迹之前或联合生成，**绝不依赖模型先生成一条轨迹才能回答问题**。

## 8. 评价与关键消融

### 8.1 标签本身（必须先做）

- 每题准确率、macro-F1、`UNK/NA` 比例、类别混淆矩阵；特别报告 Q01 灯色、Q04 横道行人、Q12/Q14 前车、Q15 切入、Q21 遮挡。
- 人审标注集：每题精度 + 证据是否足够 + 事实/预测混淆率 + 标注员一致性。
- 事件子集：红灯、前车刹停、横道行人、切入、左转让行、遮挡场景。
- 标注来源分组：RGB-only、RGB+历史、RGB+det/tracks、RGB+地图；测试仅使用可获得的输入。

### 8.2 规划与行为

- NAVSIM v1 官方 PDMS 及其子项（分项名称按固定 evaluator 版本记录）、ADE/FDE、舒适/进度、推理时延、显存、每场景生成 token 数。
- 不宣称 NAVSIM v1 是真实闭环实车验证；如有需要引入 NAVSIM v2 pseudo closed-loop 或其他真闭环环境进行补充，并与 v1 分开报告。
- 子集的行为型错误率：红灯未停、前车停车未减速、行人未让行、错误变道、停止线越界。
- 因果干预：只修改**真实且一致的**信号/障碍状态，观察 Q 与运动是否随之变化；不要在原画面不变时随意替换红/绿 token 就将结果当作物理真实。

### 8.3 对照矩阵

| 组别 | Q 预测/监督 | 轨迹表征 | 目的 |
|---|---|---|---|
| B0 SimWAM 官方模型 | 无 | 官方 `[8,3]` flow | 可复现基线 |
| B1 宽度/参数匹配 | 无 | 扩展 action hidden | 排除参数量解释 |
| D1 SimWAM + Q classifier | 辅助 CE，轨迹不读 Q | 原 flow | 检验纯辅助监督 |
| D2 SimWAM + predicted Q conditioning | 先预测 Q | 原 flow | 验证语义条件 |
| D3 SimWAM + joint question slots | 同时去噪 Q/X | 联合 flow | 核心假设 |
| D4 shuffled/random Q | 错位/随机 | 相同模型 | 证实 Q 的信息作用 |
| A0 OneVL 原始配置 | 原 latent/轨迹接口 | 官方输出 | AR 可复现基线 |
| A1 直接 AR RVQ | 无 | coarse→residual | 检验 tokenizer 作用 |
| A2 DQT→直接数值/量化轨迹 | 显式 Q | 无精度多尺度 | 检验先问答贡献 |
| A3 DQT→RVQ 多级轨迹 | 显式 Q | coarse→residual | 核心方法 |
| A4 DQT→时间多尺度 | 显式 Q | 时间级粗→细 | 扩展方法 |
| Oracle（诊断） | 测试集教师 Q | 各基线 | **只作上界，不作为部署性能** |

## 9. 预期贡献与失败边界

1. **可检查的数据贡献**：固定语义、证据来源、可追踪置信度的结构化驾驶选择题标注协议。
2. **连续生成表征贡献**：决策类 token slots 与数值轨迹 slots 共同建模，不将类别直接当物理数值。
3. **自回归生成贡献**：结构化 DQT 前缀与逐级 residual trajectory tokenizer，前者解释决策，后者恢复数值精度。
4. **可证伪的评估**：在参数匹配、随机题目、oracle/预测答案、风险子集和反事实条件下判断它是否真的有效。

若 DQT 答案本身不准、模型无视 slots、离散选择提前锁死多模态策略、或者 AR 延迟过高，这些都是合理的研究负结果；相应地应从答案质量、token 使用、误差传播和 codec 质量四方面诊断，而非增加更多选择题掩盖问题。

## 10. 基础参考与实现核验

- NAVSIM 官方 agent 接口及 4 s / 10 Hz 评测：<https://github.com/autonomousvision/navsim/blob/main/docs/agents.md>
- NAVSIM 数据划分（训练只用 navtrain，评测 navtest，避免同场景/日志泄漏）：<https://github.com/autonomousvision/navsim/blob/main/docs/splits.md>
- SimWAM 论文与代码：<https://arxiv.org/abs/2608.07468>；<https://github.com/H-EmbodVis/SimWAM>
- SimWAM 发布模型接口 `[8,3]`：<https://huggingface.co/H-EmbodVis/SimWAM>
- OneVL 仓库（latent reasoning + aux decoder，改显式 AR 需额外实现）：<https://github.com/xiaomi-research/onevl>
- OneVL 训练仓库（README 链接，使用前核对可用性）：<https://github.com/GeorgeLuImmortal/OneVL_training>
- LightNav-0（RVQ codebook 粗→细，不等同时间多尺度）：<https://github.com/lightorigins/LightNav-0>
- 官方 Qwen3.5 模型列表：<https://huggingface.co/collections/Qwen/qwen35>
