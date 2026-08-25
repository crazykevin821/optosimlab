# OptoSimLab

基于 PyTorch 的可微分光信号级仿真平台，用于构建光通信链路、光计算模块和后续的光神经网络。

本项目模拟的是复包络波形和器件传输函数，不是 FDTD 或有限元电磁场仿真器。

## 1. 当前状态

当前开发环境和基础模块已经验证可用：

```text
64 passed, 1 skipped (CUDA unavailable)
```

已完成：

| 类别 | 模块 | 状态 |
| --- | --- | --- |
| 核心 | `SimulationGrid`、FFT 频率约定、功率/SNR/NMSE 指标 | 已完成 |
| 调制 | `MachZehnderModulator` | 已完成 |
| 调制 | `MZMWithElectricalFilter` | 已完成 |
| 滤波 | `FrequencyDomainLowPass` | 已完成 |
| 传输 | `LinearDispersiveFiber`，损耗和二阶色散 | 已完成 |
| 整形 | `WaveShaper`，可训练衰减和相位遮罩 | 已完成 |
| 干涉 | `FourInputMZI` | 已完成 |
| 非理想 | `OpticalAttenuator`、`AdditiveComplexGaussianNoise` | 已完成 |
| 功能块 | `OpticalChain`、`ComplexConv1d` | 已完成 |
| 长序列 | `OverlapSaveFIR`，可训练复 FIR 的线性卷积 | 已完成 |
| 示例 | MZM 偏置的端到端自动微分训练 | 已完成 |
| 光源 | `OpticalFrequencyComb`，离散谱线、每线功率和相位 | 已完成 |
| 光源 | `ElectroOpticComb`，时域相位调制、级联 RF 驱动与总插损 | 已完成 |
| 光源 | `PracticalElectroOpticComb`，RF 一阶带宽与 Wiener 相位噪声 | 已完成 |
| 放大 | `SmallSignalEDFA`，电流映射、频率增益带宽和插损 | 已完成 |
| 放大 | `SaturatedNoisyEDFA`，小信号增益、ASE、饱和和噪声系数 | 已完成 |
| 网络层 | 调制器非线性功能块 | 暂缓，待物理模型确定 |

详细的物理约定和公式自检记录见 `docs/OptoSimLab_使用说明.pdf`。

## 2. 每次开始开发时

打开 PowerShell，逐行执行：

```powershell
cd F:\OptoSimLab
.\.venv\Scripts\Activate.ps1
python --version
python -c "import torch, numpy; print('torch =', torch.__version__); print('numpy =', numpy.__version__)"
python -m pytest
```

成功标准：最后一条命令显示 `64 passed` 或更多，且没有失败项目；没有 CUDA 时会额外显示 1 项明确跳过的 GPU 一致性测试。

如果 PowerShell 不允许激活虚拟环境，只对当前窗口执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. 安装或恢复环境

只有在 `.venv` 丢失或换电脑时才执行这一节。

```powershell
cd F:\OptoSimLab
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 仅为当前项目配置镜像，不影响其他 Python 项目
python -m pip config --site set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --site set global.timeout 120
python -m pip config --site set global.retries 5

python -m pip install --upgrade pip
python -m pip install torch numpy pytest
python -m pip install -e . --no-deps
python -m pytest
```

如普通镜像无法提供 PyTorch，可单独安装 CPU wheel：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --timeout 120 --retries 5
```

## 4. 日常验证命令

### 全部验证

```powershell
cd F:\OptoSimLab
.\.venv\Scripts\Activate.ps1
python -m pytest
python examples\train_mzm_bias.py
```

成功标准：

- 所有测试通过。
- MZM 示例最后的 `loss` 小于 `1e-5`。

### 只验证一个模块

```powershell
# MZM
python -m pytest tests\test_mzm.py -q

# 低通滤波器
python -m pytest tests\test_filters.py -q

# 光纤
python -m pytest tests\test_fiber.py -q

# WaveShaper
python -m pytest tests\test_waveshaper.py -q
```

### 只做语法检查

```powershell
python -m compileall -q src tests examples
```

成功标准：命令没有错误输出。语法检查不能替代 `pytest`。

## 5. 开发新器件的强制流程

每新增一个模块，都必须按以下顺序完成，不能跳步：

1. 明确模型边界：输入/输出形状、单位、复数或实数 dtype、是否为时域或频域模型。
2. 写出公式和符号约定，并至少找一个可解析验证点。
3. 在 `src/optosimlab/devices/` 或 `src/optosimlab/blocks/` 中实现为 `torch.nn.Module`。
4. 所有可调物理量注册为 `nn.Parameter`；正值参数必须有物理约束或非法值检测。
5. 在 `tests/test_<模块名>.py` 中写单元测试，至少包含：正常值、解析点、非法输入和梯度检查。
6. 运行模块测试，再运行全部测试。
7. 更新本 README 和 PDF 使用说明，写明模型假设、公式、限制和测试结果。

每个模块完成后的 PowerShell 验证模板：

```powershell
python -m compileall -q src tests examples
python -m pytest tests\test_<模块名>.py -q
python -m pytest
```

三个命令都成功，模块才可以接入 `OpticalChain` 或神经网络层。

## 6. 已完成：离散频谱光梳

已实现确定性的离散频谱光梳。当前范围：

- 输入：采样网格、线间隔、谱线数量、每根谱线的功率和相位。
- 输出：复包络时域波形，频域谱线位于正确的 FFT bin。
- 不做：锁模激光器腔内动力学、相位噪声和泵浦噪声。

验收测试必须覆盖：

- 谱线数和谱线位置正确。
- 总平均功率等于各谱线功率之和。
- 相位参数存在有效梯度。
- 非 FFT-bin 对齐的线间隔必须抛出异常，或明确使用插值模型。

验证光梳模块：

```powershell
cd F:\OptoSimLab
.\.venv\Scripts\Activate.ps1
python -m pytest tests\test_optical_comb.py -q
```

本模块验证完成后，再运行全量验证：

```powershell
python -m pytest
```

## 7. 已完成：时域电光梳

`ElectroOpticComb` 对复光包络施加一台或多台级联相位调制器：

```text
E_out(t) = E_in(t) exp(j Σ beta_k sin(2π f_k t + phi_k)) 10^(-L_dB/20)
```

- `modulation_indices_rad`、`rf_phases_rad` 与 `insertion_loss_db` 是 `nn.Parameter`，可直接由 PyTorch 优化器训练。
- `modulation_frequency_hz` 是离散仿真结构参数，必须是 `sample_rate_hz / samples` 的整数倍，并且低于 Nyquist 频率；这样有限记录内的梳齿才不会由采样窗引入频谱泄漏。
- 不带插损时，指数相位因子的模恒为 1，因此逐点和平均功率都严格守恒；插损 `L_dB` 使用正确的场系数 `10^(-L_dB/20)`。
- 第一版假定 RF 正弦驱动理想且相干；RF 相噪、调制器响应与电带宽仍是后续扩展，不应当当作已建模效果。

验证这个模块：

```powershell
cd F:\OptoSimLab
.\.venv\Scripts\Activate.ps1
python -m pytest tests\test_electro_optic_comb.py -q
```

预期结果为 `6 passed`。测试覆盖零调制恒等性、逐采样点公式、Jacobi-Anger/Bessel 载波和一阶边带、功率守恒与 dB 插损、梯度，以及非复输入、非整数 FFT bin 和 Nyquist 驱动的拒绝逻辑。

### 实用 RF 扩展

`PracticalElectroOpticComb` 在理想电光梳上加入一阶 RF 响应和相位噪声：

```text
H_rf(f) = 1 / (1 + j f / f_c)
Var[phi_noise[n+1] - phi_noise[n]] = D_phi × dt
```

- RF 调制深度自动乘以 `|H_rf|`，RF 相位自动叠加 `arg(H_rf)`；因此幅度和相位滞后同时符合一阶低通。
- `rf_phase_noise_diffusion_rad2_per_s` 是单位为 rad²/s 的 Wiener 相位扩散系数；传入相同随机数生成器种子即可复现实验。
- 它不是完整的振荡器相位噪声掩膜模型，但理想退化、带宽与随机游走统计均已测试。

```powershell
python -m pytest tests\test_practical_electro_optic_comb.py -q
```

预期结果为 `5 passed`。

## 8. 已完成：小信号 EDFA

`SmallSignalEDFA` 是无噪声、无饱和的确定性频域放大器，使用以下功率增益：

```text
G_peak_dB = gain_slope_db_per_ma × max(pump_current_ma - transparency_current_ma, 0) - insertion_loss_db
G_power(f) = 10^(G_peak_dB / 10) × exp[-4 ln(2) ((f - f0) / B)^2]
E_out(f) = sqrt(G_power(f)) × E_in(f)
```

- `B` 是功率增益相对峰值的 FWHM：在 `f0 ± B/2` 处，功率增益严格为峰值的一半，即相对下降 3.0103 dB。
- 泵浦电流、透明电流、增益斜率、带宽、中心失谐和插损均已封装成 `nn.Parameter`，支持训练与 `state_dict` 保存。
- 当前仅适用于小信号线性工作区；ASE、增益饱和、噪声系数、泵浦动态和偏振效应尚未建模。

验证这个模块：

```powershell
cd F:\OptoSimLab
.\.venv\Scripts\Activate.ps1
python -m pytest tests\test_edfa.py -q
```

预期结果为 `6 passed`。测试覆盖电流-增益映射、功率 dB 插损、-3 dB FWHM、中心失谐、无额外相位、梯度，以及非法参数和 dtype 的拒绝逻辑。

## 9. 已完成：饱和与 ASE EDFA

`SaturatedNoisyEDFA` 继承小信号 EDFA，并在同一封装中加入增益饱和和 ASE：

```text
G_eff(f) = G_ss(f) / (1 + P_in / P_sat)
S_ASE(f) = n_sp × h × nu × max(G_eff(f) - 1, 0)
n_sp = 10^(noise_figure_db / 10) / 2
```

- `P_in` 是输入复包络的平均功率；`P_sat`、噪声系数与基础 EDFA 的所有物理量均为 `nn.Parameter`。
- ASE 使用一偏振复包络、双边 PSD 约定，单位为 W/Hz。每一个离散频点的随机频谱方差按 `N^2 × S_ASE × df` 生成，因此 IFFT 后的期望平均噪声功率是 `sum(S_ASE × df)`。
- 该饱和模型是用于信号级训练的现象学标量模型：保留频谱形状、压缩整体增益；不是掺铒能级布居动力学模型。

验证这个模块：

```powershell
cd F:\OptoSimLab
.\.venv\Scripts\Activate.ps1
python -m pytest tests\test_saturated_noisy_edfa.py -q
```

预期结果为 `6 passed`。测试包含饱和公式、低功率极限、ASE PSD 与离散 FFT 噪声功率统计一致性、零增益 ASE、所有参数梯度和非法输入。

## 10. 系统级指标工具

已加入以下通用指标函数：

- `power_spectrum` 与 `centered_frequency_axis_hz`：fftshift 频谱，谱线功率和严格等于时域平均功率。
- `eye_diagram`：按一个符号间隔重叠折叠为两符号眼图轨迹。
- `error_vector_magnitude`：复包络的 RMS EVM。
- `gaussian_q_factor` 与 `gaussian_ber_estimate`：仅适用于等先验、实值、近似高斯的二元判决样本。

```powershell
python -m pytest tests\test_system_metrics.py -q
```

预期结果为 `5 passed`。

## 11. 已完成：长序列 overlap-save

`OverlapSaveFIR` 是可训练复 FIR 的线性、因果卷积模块：

- 输入左侧补 `M-1` 个零，`M` 是脉冲响应长度；每个 FFT 块丢弃前 `M-1` 个受循环卷积污染的输出。
- 每块保留 `N_fft-M+1` 个有效样本，末块自动补零后再裁剪回原信号长度。
- 输出严格等于零初始条件下的直接线性卷积，而不是整段 FFT 的循环卷积。

```powershell
python -m pytest tests\test_overlap_save.py -q
```

预期结果为 `5 passed`。

## 12. GPU 一致性门禁

`tests\test_cuda_consistency.py` 会在安装 CUDA 版 PyTorch 且 GPU 可用时，比较光纤、两类电光梳、EDFA 和 overlap-save 的 CPU/GPU 结果；当前电脑是 CPU 版 PyTorch，因此该测试显示为 `skipped`，不是通过。

## 13. 后续路线

### 阶段 A：补齐器件库

1. 实验曲线标定和器件测量数据拟合。
2. 端到端小型光计算训练示例。

### 阶段 B：系统级可用性

1. 给频域器件加入零填充或 overlap-save，消除循环卷积伪影。
2. 增加批处理、GPU 与 CPU 一致性测试。
3. 增加保存/加载、参数校准和实验曲线拟合示例。
4. 增加链路级指标：EVM、BER 估计、频谱和眼图。

### 阶段 C：光神经网络

1. 基于 `OpticalChain` 定义可训练光学层。
2. 为复值卷积补充批处理、梯度和数值一致性测试。
3. 建立小型分类或回归任务，验证端到端反向传播。
4. 使用器件测量数据替换理想传输曲线，比较仿真与实验。

## 8. 全部完成的验收标准

只有同时满足以下条件，才把平台标为“完整的第一版”：

- [x] README 计划中的电光梳和 EDFA 模块全部完成。
- [ ] 每个器件均有公式、单位、输入输出契约和失败用例。
- [ ] 每个可训练参数都是 `nn.Parameter`，并通过梯度测试。
- [ ] CPU 上全部测试通过；如使用 GPU，CPU/GPU 结果在约定误差内一致。
- [ ] 长序列的频域处理不再依赖未说明的循环卷积。
- [ ] 至少有一个端到端光计算或光神经网络训练示例。
- [ ] 至少有一个带非理想器件参数的链路示例。
- [ ] `docs/OptoSimLab_使用说明.pdf` 与真实实现、测试结果一致。

## 9. 工程结构

```text
F:\OptoSimLab
├── src\optosimlab\
│   ├── config.py       # 采样网格和物理常数
│   ├── devices\        # 器件模型
│   ├── blocks\         # 组合模块和复值卷积
│   └── metrics.py       # 功率、SNR、NMSE
├── tests\              # 每个模块的自动测试
├── examples\           # 可运行示例
├── docs\               # PDF 使用说明和生成源
└── pyproject.toml       # 打包与依赖定义
```
