# Tomato OpenPI 训练仓库

公开版保留基础 π0.5 训练与 T3：双臂逐未来时间位置 z-score 归一化（`timestamp_zscore_arms`）和相关高斯噪声（`paper_beta=0.5`）。训练监督完整动作窗口。T3 仅支持 JAX、30 Hz Sharpa multimodal 全参；双臂 Delta，双手 Absolute / 全局 quantile。当前完成软件与离线验证，真实数据完整训练和实机验证尚未完成。

本仓库来源于 [tomato-openpi/main](https://github.com/LTT-BIT/tomato-openpi/tree/main)，保留共同 Git 历史。公开版已移除 Train RTC 的训练、采样、配置与界面入口；历史提交保留上游原始内容。逐时间统计、相关矩阵及资产校验实现保持不变。

基于 OpenPI 的 π0.5 训练仓库，使用统一 JAX 入口，支持 FR3/Wuji、FR3/Sharpa、单臂 Franka、双 UR/O30 的 LoRA 与全参数微调。数据转换在独立仓库完成，这里负责数据校验、归一化统计和训练。

本页说明下载安装和运行；本地个人配置、公共 baseline 变更和多人协作见 [多人开发 README](多人开发README.md)。

多人在同一服务器训练时，各自在自己的目录独立 clone。日常训练直接使用公共 baseline，或在本地 `template/<负责人>/` 保存个人配置；个人配置已被 Git 忽略，不创建分支、不提交、不上传。只有团队决定新增或修改公共 baseline 时才进入公共开发流程。目录和配置类的组合关系见[开发结构说明](多人开发README.md#目录与配置的组合关系)。

## 500028 服务器：复用公共环境

服务器环境已经准备好，直接使用以下路径和命令。账号需属于 `ai_researchers` 组。

| 内容 | 路径 |
| --- | --- |
| 公共环境 | `/mnt/data/public_tools/training/environments/pi05/20261014` |
| 基础权重 | `/mnt/data/public_ckpt/pi05/pi05_base/params` |
| 仓库启动入口 | `bash scripts/python.sh` |

**下载代码**（已有仓库直接进入自己的仓库目录）：

```bash
git clone --branch main \
  https://github.com/LTT-BIT/tomato-openpi-public.git
cd tomato-openpi-public
```

**启动图形界面**，在自己的仓库根目录执行：

```bash
export PI05_SHARED_ENV=/mnt/data/public_tools/training/environments/pi05/20261014
# 联网记录 W&B 时首次登录；选择 offline 时不需要登录。
bash scripts/python.sh -m wandb login
bash scripts/python.sh -m scripts.training_ui --port 8765
```

Sharpa multimodal 全参训练可在「03 训练参数 → 论文功能」直接勾选「逐时间步归一化」和「相关噪声」，两项同时选择为 T3，无需先运行个人配置生成命令。切换功能自动使用独立统计名称，并固定双臂 delta、双手 absolute 与无放回完整窗口采样；点击启动时保存可复现个人配置。已有个人配置会还原算法选项，当前完成的是配置与软件测试，尚不代表真实数据训练或实机验证完成。同步代码后须重启自己的 UI 进程并刷新页面。

**在自己的电脑打开服务器 UI**：服务监听服务器的 `127.0.0.1:8765`，需要通过 SSH 转发到本地。

- **使用 VS Code Remote SSH**：连接服务器后，在底部“端口”面板点击“转发端口”，输入 `8765`。浏览器打开启动终端打印的完整地址，保留 `#token=...`。如果 VS Code 显示的本地端口不是 8765，把地址里的端口改成本地端口。
- **没有 VS Code**：服务器终端保持 UI 服务运行；另开自己电脑的终端（Linux/macOS Terminal 或 Windows PowerShell），执行：

  ```bash
  ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8765:127.0.0.1:8765 zhiwei500028
  ```

  `zhiwei500028` 是本机 SSH 配置中的服务器别名。如果没有别名，替换为 `用户名@服务器地址`，非默认 SSH 端口加 `-p SSH端口`。保持该终端开启，然后在本机浏览器打开 `http://127.0.0.1:8765/#token=启动时打印的token`。

**多人使用时会有端口冲突**：同一台服务器上的 UI 端口由所有用户共享，即使账号、仓库目录不同，也不能同时使用 `127.0.0.1:8765`。每人启动一个 UI，选不同的服务端口，例如 8765、8766、8767；一个 UI 可以管理多个训练任务，无需为每次训练再开端口。端口分配与 GPU 分配是两回事，仍需协调显卡使用。

例如服务器的 8765 已被别人使用，你可以使用服务器 8766，并转发到自己电脑的 18766：

```bash
# 服务器终端：在自己的仓库目录启动 UI
bash scripts/python.sh -m scripts.training_ui --port 8766
```

```bash
# 自己电脑的终端：保持 SSH 转发运行
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18766:127.0.0.1:8766 zhiwei500028
```

浏览器打开 `http://127.0.0.1:18766/#token=本次UI启动时打印的token`。转发命令中 `18766` 是本机端口，最后的 `8766` 是服务器 UI 端口，必须与 `--port` 一致。VS Code 用户直接转发服务器端口 8766，然后使用端口面板显示的本地地址并附上 token。

本机端口只需在自己的电脑上空闲，不同人的电脑可以使用相同本机端口。UI 启动出现 `Address already in use` 时换服务器端口；SSH 转发提示本地端口被占用时，只换本机端口。不要为了腾端口停止别人的服务。可在服务器 tmux 中运行 UI，断开 SSH 后仍可重新连接页面。

服务器无需重新安装 Python 环境或前端依赖。启动脚本依次选择：显式 `PI05_SHARED_ENV`、服务器公共 `20261014` 环境、本机 `.venv`。服务器公共环境优先于仓库内旧 `.venv`；公共目录存在但不可执行时直接报错。`20261014` 环境为 RTX PRO Blackwell 多卡训练预加载已验证的 NCCL 2.31.2；必须通过 `bash scripts/python.sh` 进入，不能绕过公共 `run-python` 直接调用环境内 Python。

**全部八份 baseline 默认读取公共权重 `/mnt/data/public_ckpt/pi05/pi05_base/params`，无需手动点击切换，也不会因该目录缺失自动回退到云端下载。**旧个人配置保留原参数；选中后检查“基础权重”字段，若仍为 `gs://...`，点击“使用服务器公共权重”或手动填写公共目录。tokenizer 首次使用仍需下载或提前准备仓库缓存。

## 图形界面：配置和启动训练

自己的电脑完成下方环境安装后，同样执行 `bash scripts/python.sh -m scripts.training_ui --port 8765`，打开终端打印的地址。

1. **选择配置**：可选公共 baseline 或本地 `template/<用户名>/*.py` 中已有个人配置，包含手写配置和 UI 创建的配置。点击“刷新配置”发现新文件；选择后读取实际参数、本体维度、频率和英文 prompt。高级模型及 subtask 设置沿用来源配置。
2. **选择数据**：可多次浏览添加目录，或在数据目录框每行输入一个路径。选择父目录后点击“识别目录内的数据集”，核对勾选并点击“应用选中数据集”；默认不把同级 `full` 和 `subtasks` 同时选入。路径属于运行 UI 的训练机器，浏览器不上传本机数据。确认数据版本、repo_id、来源和统计位置；应用选择会填写实际来源并建议独立统计目录。仅搬迁完全相同的数据可校验复用原统计；换数据组合需重新计算统计。
3. **修改参数**：可修改 batch、步数、学习率计划、workers、FSDP、随机种子、保存间隔、基础权重和 W&B 项目。支持 A/B 权重及 episode 分组有放回采样。番茄 Wuji / Sharpa 可选 absolute 或 delta 动作监督；其他本体保持 absolute。本体维度、模型结构和任务视图沿用所选 baseline。
4. **选择 GPU 和 W&B 模式**：界面读取 `nvidia-smi` 的显存和利用率，可多选。全局 batch 必须能被卡数整除；卡数必须能被 FSDP 分片数整除。online 需要预先在终端登录，offline 明确保留本地记录，之后按下文同步。
5. **启动实验**：最终训练参数与来源相同时复用原配置；修改训练参数才保存新的个人 Python 文件。每次启动自动记录独立实验，后台依次校验数据、缺少时计算统计、核对统计绑定并启动训练。可先点击“检查数据与统计”，它只核验所填数据目录的 LeRobot 元信息、频率、维度、相机、英文 prompt 和已有归一化统计，不会训练或生成统计；进度与错误显示在页面下方“运行详情”。“查看最终配置”用于核对实际生效参数。任何阶段失败都会显示错误和日志，不继续下一阶段。

新增个人配置保存在所选任务的 `template/<用户名>/<实验名>.py`，与 `baseline/` 同级；Git 忽略，不用创建分支或上传。仅修改实验名称、数据目录、统计目录/名称、W&B 项目、GPU 或 W&B 模式不会新增配置文件，这些设置写入本次任务的 `run.json`。更改 batch、学习率、步数、采样策略、基础权重、数据身份/来源等参数才新建文件；改回来源值会复用原文件。页面及“查看最终配置”会提示保存方式。

**自动读取采样分组**：选择数据目录，将“采样策略”改为“按分组权重，有放回采样”，点击“从数据读取分组”。单数据集读取 `conversion/manifest.json` 和 `conversion/records/*.json` 恢复来源分组；多个数据集则按每个目录建立分组，并显示全局 episode 编号、条数、帧数和来源。同名组保留已有权重，新组默认权重 1；需要 7:3 时将对应两组权重设为 7、3。编号区间包含两端，比例是长期采样概率，不是每个 batch 的固定配额。

读取成功会替换表单中的分组，失败则保留原值；不会修改数据或立即创建配置。启动时，具体分组编号随最终参数保存到配置（如有参数变化）及本次实验记录。更换数据目录后需重新读取或手动核对编号。历史来源路径只作为标识，不要求原目录仍可访问，也不根据历史 `source_recording_id` 猜组。单数据集缺少转换来源、记录冲突或覆盖不全时请手动填写；带任务/阶段筛选的配置目前也需按有效训练 episode 手动填写。普通逐轮打乱采样不需要设置分组。

**多目录联合训练示例**：本机 `harvest-tomato/0924-250ep` 下有 `A`、`B`，`1010` 下有 `full`、`subtasks`。添加两个父目录并识别后，默认勾选 `0924-250ep/A`、`0924-250ep/B`、`1010/full`；需点击“应用选中数据集”确认。当前这组本机数据共 441 条 episode、481,223 帧。`1010/subtasks` 是另一种任务组织方式，默认不选；可按实际任务改选，但不要无意中同时加入重复样本。

联合训练直接读取各目录，不复制数据或改写原 episode 编号；每个目录独立生成动作窗口、读取自己的英文 task。默认逐轮打乱覆盖所有目录的全部帧，归一化统计也由组合中的全部帧共同计算，采样权重不改变统计计算方式。频率、向量维度/顺序、相机等不兼容时会拒绝启动。当前不支持把带 `TaskView` 筛选规则的配置扩展为多目录，需使用无此筛选的配置。

多目录的采样编号按表单目录顺序、各目录原 episode 编号升序连续分配。上例为 `0-105`、`106-249`、`250-440`；这些是联合采样编号，原文件内编号保持不变。“检查数据与统计”展示映射，实验审计保存每个来源的内容指纹及原编号到全局编号的映射。增删或重排数据目录后重新核对分组并计算统计，使用新实验名；续训会拒绝数据组合或内容改变。

同一配置可反复运行，每次从头训练使用不同实验名，日志与权重独立保存。新配置保留来源源码与所选参数，不覆盖已有个人文件；配置名和实验运行记录各自管理。用户名及新实验名使用英文字母、数字、下划线，不能以数字开头。

关闭网页或退出 UI 服务不会停止后台任务；任务记录在重开界面后仍可读取。**停止任务：直接点击右侧任务记录中对应任务的“停止任务”，或先选中任务，再点击启动区的“停止选中任务”；页面底部详情也保留停止按钮。**准备数据、下载权重和训练阶段都可请求停止，等待状态显示“已停止”。停止不删除记录，也不保证额外保存一次权重，只能从已有 checkpoint 恢复；还未产生 checkpoint 但已创建实验目录时，请换实验名。重试/续训读取原任务保存的运行设置（包括当时覆盖的数据目录），不会应用当前表单修改；配置被改动时会拒绝续训。新增其他个人实验文件不影响本次续训；公共源码更新仍会触发严格校验，旧版本训练需保留原 checkout。共享磁盘上的任务需要回到原服务器 UI 操作。

UI 流程日志和状态在 `logs/training_ui/<任务ID>/`；正式训练仍按现有规则写入 `logs/`、`checkpoints/`、`wandb/`，不会写入公共环境。GPU 占用检查用于提示，本仓库 UI 会预留自己任务使用的卡；其他 clone、其他用户及命令行训练仍需协调。运行期间不要修改正在使用的源码或配置。

UI 只管理本仓库训练，不安装驱动或环境、不转换数据、不修改 W&B 登录凭据。除首次准备环境、登录 W&B 和启动 UI 外，日常数据准备检查、统计计算、参数选择及训练操作都可在页面完成。下面的命令行流程仍可使用。

UI 启动 GPU 训练时默认传入 `XLA_PYTHON_CLIENT_PREALLOCATE=true` 和 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`，无需每次手动 export；实际值记录在任务状态的 `training_environment` 和流程日志中。若启动 UI 的环境已显式设置这两个变量，则保留其值。此默认值仅作用于 UI 后续启动的 GPU 训练，不改变命令行入口、已有任务或 baseline 配置。预分配不能保证更大的 batch 能运行，仍须根据实际训练验证显存需求。

## 自己的电脑：安装本地环境


适用于 Linux + NVIDIA GPU 训练机。先确认 `nvidia-smi` 正常，并将本机 SSH 公钥加入 GitLab 账号。环境使用 Python 3.11、JAX CUDA 12 和 PyTorch CUDA 12.8，驱动需支持对应版本。

Ubuntu/Debian 安装基础工具和视频依赖：

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates openssh-client ffmpeg libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/0.12.10/install.sh | sh
```

重新打开终端，确认 `uv --version` 可用，然后下载仓库、安装环境：

```bash
git clone --branch main \
  https://github.com/LTT-BIT/tomato-openpi-public.git
cd tomato-openpi-public
unset PI05_SHARED_ENV
bash scripts/setup_training_env.sh --local
```

安装脚本按 `uv.lock` 在当前仓库创建 `.venv/`，并打印 JAX 设备和 Torch CUDA 状态。正式训练前应看到 GPU 设备及 `CUDA=True`。以后使用 `bash scripts/python.sh` 自动读取本地 `.venv`。

baseline 默认基础权重为服务器路径 `/mnt/data/public_ckpt/pi05/pi05_base/params`。自己的电脑若没有该挂载，在 UI 的“基础权重”中填写已下载的本地 `params` 目录，或明确填写 `gs://openpi-assets/checkpoints/pi05_base/params` 以下载官方权重；命令行可用 `--weight-loader.params-path` 覆盖。离线训练前还需准备 tokenizer 缓存。默认不会自动下载基础权重。

## 任务与 baseline

| 任务 | 说明和配置 | 当前状态 |
| --- | --- | --- |
| 番茄-Wuji | [tomato_wuji](training_configs/experiments/tomato_wuji/README.md) | LoRA / 全参 v1 baseline |
| 番茄-Sharpa | [tomato_sharpa](training_configs/experiments/tomato_sharpa/README.md) | 统一使用 baseline 下的 LoRA / 全参配置，支持转换器 116D→58D RGB 输入；完整训练和实机待验证 |
| 铲柠檬 | [lemon_scooping](training_configs/experiments/lemon_scooping/README.md) | 双 UR/O30 LoRA / 全参 v1 baseline；209 条有效数据全部训练 |
| 抓取杯子 | [cup_grasping](training_configs/experiments/cup_grasping/README.md) | 单臂 Franka LoRA / 全参 v1 baseline；v2.1 数据待准备 |

每个任务下的 `baseline/` 存放团队发布的参考配置，`template/` 供每个人在自己的机器上保存个人实验配置。`template/` 内容由 `.gitignore` 排除，不上传 GitLab；数据、统计和训练结果也不随仓库分发。

当前八份 baseline：

| 任务 / 方案 | baseline 文件 | 动作维度 | 总步数 | 默认数据目录 |
| --- | --- | ---: | ---: | --- |
| 番茄-Wuji LoRA | [tomato_fr3_lora_v1.py](training_configs/experiments/tomato_wuji/baseline/tomato_fr3_lora_v1.py) | 54 | 30,000 | `data/tomato_fulltask_v1` |
| 番茄-Wuji 全参 | [tomato_fr3_fullft_v1.py](training_configs/experiments/tomato_wuji/baseline/tomato_fr3_fullft_v1.py) | 54 | 30,000 | `data/tomato_all_full_v1` |
| 番茄-Sharpa LoRA | [tomato_fr3_sharpa_lora_v1.py](training_configs/experiments/tomato_sharpa/baseline/tomato_fr3_sharpa_lora_v1.py) | 58 | 30,000 | `/mnt/data/users/tairan/workspace/data/10.16_2_data_lerobot` |
| 番茄-Sharpa 全参 | [tomato_fr3_sharpa_fullft_v1.py](training_configs/experiments/tomato_sharpa/baseline/tomato_fr3_sharpa_fullft_v1.py) | 58 | 30,000 | `/mnt/data/users/tairan/workspace/data/10.16_2_data_lerobot` |
| 抓取杯子 LoRA | [cup_single_franka_lora_v1.py](training_configs/experiments/cup_grasping/baseline/cup_single_franka_lora_v1.py) | 7（模型补零至 32） | 30,000 | `data/cup_grasping_v1` |
| 抓取杯子 全参 | [cup_single_franka_fullft_v1.py](training_configs/experiments/cup_grasping/baseline/cup_single_franka_fullft_v1.py) | 7（模型补零至 32） | 30,000 | `data/cup_grasping_v1` |
| 铲柠檬 LoRA | [lemon_ur_o30_lora_v1.py](training_configs/experiments/lemon_scooping/baseline/lemon_ur_o30_lora_v1.py) | 12（模型 32，状态 52） | 30,000 | `data/lemon_wrist_reset_v1` |
| 铲柠檬 全参 | [lemon_ur_o30_fullft_v1.py](training_configs/experiments/lemon_scooping/baseline/lemon_ur_o30_fullft_v1.py) | 12（模型 32，状态 52） | 30,000 | `data/lemon_wrist_reset_v1` |

番茄与杯子配置采用以下公共训练参数；Sharpa 全参覆盖为 batch 32、FSDP 8（默认 8 卡），Sharpa LoRA 保持 batch 8、FSDP 1，两者从转换器 116D 状态提取 58D 位置并重排动作，详见 [Sharpa 说明](training_configs/experiments/tomato_sharpa/README.md)。柠檬 LoRA 复用公共训练配方，柠檬全参学习率与公共配方一致，日志与 checkpoint 保存设置沿用来源任务记录，详见[柠檬说明](training_configs/experiments/lemon_scooping/README.md)。以下公共参数中的频率与动作语义以各任务协议为准。

本体协议、数据身份、统计及英文 prompt 见对应任务说明。

## 公共训练参数

| 参数 | 默认值 |
| --- | --- |
| 模型 / 精度 | π0.5，bfloat16；max_token_len=256，discrete_state_input=True |
| LoRA | VLM `gemma_2b_lora` rank/alpha=16/16；Expert `gemma_300m_lora` rank/alpha=32/32；视觉及投影可训练 |
| 全参 | `gemma_2b` + `gemma_300m`，所有参数可训练 |
| 频率 / horizon | 30 Hz / 50；所有 baseline 使用 absolute 动作目标；数据 action 必须为绝对目标值 |
| batch / workers / FSDP | 全局 batch：LoRA 8、全参 64 / workers 8 / fsdp_devices=1 |
| 采样 / seed | epoch_shuffle；seed=42、model_seed=42、data_seed=42001、augmentation_seed=42002 |
| 学习率 | cosine；warmup=1,000，peak=2.5e-5，decay_steps=30,000，end=2.5e-6 |
| 优化器 | AdamW；β1=0.9、β2=0.95、eps=1e-8、weight_decay=1e-10、clip=1.0 |
| EMA | 关闭 |
| 日志 / checkpoint | log_interval=10、save_interval=2,000、keep_period=10,000 |
| W&B | 启用，project=openpi；联网 online，离线显式 offline |

番茄 Wuji / Sharpa 的 UI“动作监督”默认 absolute。选择 delta 时，仅将左右机械臂各 7 维绝对关节目标减去当前 state，手部目标仍为 absolute；两种模式均要求原始数据的 action 是绝对目标，不能把已经是 delta 的原始数据再转换一次。切换模式会生成新的个人配置和实验名，并要求新的 asset_id 与独立归一化统计；UI 会自动建议独立统计名称。必须核对数据语义后重新统计和开始新实验，不能复用另一模式的统计或 checkpoint。杯子与柠檬暂不提供切换。

需要 A/B 按 7:3 等比例有放回采样时，只需在本地个人配置中设置 `sampling_mode="weighted_replacement"` 和 `sampling_groups`，见[采样配置示例](多人开发README.md#可选-ab-采样比例)。不设置时沿用 baseline 的逐轮打乱采样。

LoRA 和全参默认均训练 30,000 步，学习率计划的 decay_steps 同为 30,000；步数不随数据量自动调整。个人实验可按实际训练计划覆盖。

以下以 Wuji LoRA 为例，选择模块并展开配置，无需数据、权重或 GPU：

```bash
export MODULE=training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1
export EXP=alice_tomato_wuji_lora_s42_run01
bash scripts/python.sh -m scripts.run_training "$MODULE" --exp-name "$EXP" --dry-run
```

更换方案时修改 MODULE。个人修改数据路径、学习率或预算时，按 [多人开发 README](多人开发README.md) 在本地复制 baseline，再将 MODULE 指向本地模块；不需要创建分支或同步 GitLab。

## 数据与统计

将数据仓库产出的 LeRobot v2.1 数据放入 `data/`，包含 `meta/`、`data/`、`videos/`。在所选实验的 `build_dataset()` 中确认 root、repo_id、来源、版本、assets_dir、asset_id 和英文 prompt 示例。`root` 是实际目录，`repo_id` 是数据身份；训练实际英文指令来自 `meta/tasks.jsonl`。

数据应满足任务的关节顺序、单位、相机映射和采样频率：番茄与杯子为 30 Hz，柠檬为 10 Hz。新数据使用新的版本和 asset_id；同一内容仅搬迁路径时保留身份。

新数据先计算统计：

```bash
bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 512
```

统计保存到 `<assets_dir>/<asset_id>/`，包含 `norm_stats.json` 和 `norm_metadata.json`。已有配套统计时跳过计算，迁移时两个文件一起复制。不要提前创建最终 asset_id 目录：工具拒绝覆盖已有目录，包括空目录；同一目标的重复计算进程也会被拒绝。

统计绑定数据文件、输入变换和 horizon。改变这些内容后重新计算；仅改学习率或 batch 可复用。常量关节允许 std=0，归一化使用 epsilon。番茄 Wuji / Sharpa 已切换为 absolute，四份 baseline 使用带 absolute 的新 asset_id 与默认实验名。必须重新计算统计，不能复用旧 delta 统计或对旧 checkpoint 直接 resume。

## 本机检查与训练

服务器启动命令见上方“500028 服务器”部分，包含公共基础权重路径。以下命令用于本机默认基础权重。

```bash
bash scripts/python.sh -m scripts.run_training "$MODULE" --exp-name "$EXP" --check
bash scripts/python.sh -m wandb login
CUDA_VISIBLE_DEVICES=0 bash scripts/python.sh -u -m scripts.run_training \
  "$MODULE" --exp-name "$EXP" --wandb-mode online
```

`--check` 验证数据元信息、fps、维度、相机、英文 task、统计数值及数据/变换绑定，需要已有统计，不构建模型。正式启动还核对实际帧数并记录数据文件与采样审计。它们不能代替关节语义、单位及视频/动作时间对齐的验收。

GPU 编号按分配修改，全局 batch 必须能被可见设备数整除；单 GPU 命令仅为示例。可临时用 `--batch-size 4` 等 CLI 参数调试，正式实验将设置写入本地个人模块。个人模块不提交；训练启动器会把它自动复制到日志目录，并在运行清单中记录文件哈希和当前公共仓库 commit。训练期间不要修改正在运行的目录。

多卡训练启动时，W&B 样例图只从一个本地 GPU 分片复制到主机，不会为了记录图片跨卡聚合整个 batch；这不影响实际训练数据和 FSDP 分片。

无网络时使用 `--wandb-mode offline`，本地日志和 W&B 仍保存。恢复联网后逐个同步离线段：

```bash
bash scripts/python.sh -m wandb login
bash scripts/python.sh -m wandb sync wandb/offline-run-实际编号
```

## 输出与续训

训练输出和可写缓存默认位于当前仓库；服务器公共环境与基础权重使用上方服务器部分列出的共享路径：

| 内容 | 位置 |
| --- | --- |
| 本地环境 | 自己电脑使用 `.venv/`；服务器使用公共环境 |
| 缓存 | `.cache/`；JAX 编译缓存 `.cache/jax/` |
| 数据 / 统计 | `data/`、`assets/` |
| 本地日志、运行清单及配置源码快照 | `logs/<配置名>/<实验名>/` |
| checkpoint | `checkpoints/<配置名>/<实验名>/<step>/` |
| W&B | `wandb/` |

每个 checkpoint 的 `params/` 保存权重，`train_state/` 保存优化器等恢复状态，`assets/` 保存配套统计。运行清单记录 commit、最终配置、数据来源、实际 prompt、统计摘要和运行结果。

恢复未完成实验时使用原 commit、模块、配置、数据、统计和实验名。服务器使用上方服务器训练命令并加 `--resume`；本机执行：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/python.sh -u -m scripts.run_training \
  "$MODULE" --exp-name "$EXP" --resume --wandb-mode online
```

迁移续训保留完整实验目录，包括 checkpoint、`reproducibility/`、`wandb_id.txt` 和 `wandb_sessions/`，以及未同步的 W&B 日志。只有 params 不足以精确续训。缺少有效 checkpoint 会报错；已完成原预算直接退出。改变代码、数据或预算后创建新实验，不能在新 main 上直接恢复旧运行。

每次启动或续训创建一个独立 W&B run，同一实验通过固定 group 关联；云端不会自动拼成一条曲线。各段的恢复步数和目录记录在 `wandb_sessions/`，离线段需逐个 sync。

历史训练参数、checkpoint 与指标见 [历史实验记录](docs/records/历史实验记录.md)。本仓库不包含这些历史运行产物。

GitLab 自动验证目前暂停。本地代码检查命令为 `bash scripts/check_training.sh`，协作与验证规则见 [多人开发 README](多人开发README.md)。本项目遵循 [LICENSE](LICENSE) 和 [Gemma 许可](LICENSE_GEMMA.txt)。

## 显式选择的 Delta 论文训练实验

Sharpa可在本地 `template/<负责人>/` 选择双臂Delta、逐未来时间位置归一化和Correlated Noise的新实验；八份baseline及recipe默认值不变。共享代码与个人配置分开，个人文件不提交、不上传；首次可用 `scripts.create_delta_paper_experiments --owner <负责人>` 在固定任务目录创建。对照设计、独立统计、训练命令和新权重部署限制见 [Sharpa论文算法实验](training_configs/experiments/tomato_sharpa/README.md#本地-delta-论文算法实验不修改-baseline)。目前仅完成离线代码验证，未执行真实数据完整训练。

论文训练实验现支持原目录中的no-block-one六份摘番茄数据联合加载，逐源完整窗口、统一统计及独立个人配置；使用方法见[任务说明](training_configs/experiments/tomato_sharpa/README.md#no-block-one-六份摘番茄数据联合实验)。原baseline保持原样。

## 普通训练与 T3

选择正式 Sharpa 全参 baseline 可运行普通 π0.5；在「03 训练参数 → 论文功能」勾选逐时间步归一化和相关噪声即可运行 T3。两项功能可独立选择，T0/T1/T2/T3/TG 对照与单源、六源数据入口保留。切换算法需独立统计和新实验名，不能混用 checkpoint 或续训合同。

## 跟踪原仓库 main

公开版使用 `origin` 推送到本仓库，使用 `upstream` 跟踪原仓库 main：

```bash
git remote add upstream https://github.com/LTT-BIT/tomato-openpi.git
git fetch upstream main
git branch --set-upstream-to=upstream/main main
git config branch.main.pushRemote origin
git config push.default current
```

运行 `git fetch upstream` 获取后续更新，审查后执行 `git merge upstream/main`，处理公开版差异并运行 `bash scripts/check_training.sh` 后再 `git push origin main`。上游包含 Train RTC，合并时须保持本公开版的完整窗口训练行为和接口约束。统计实现如有变化，应按资产合同重新生成统计，不能跳过来源和摘要检查。
