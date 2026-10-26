# OptoSimLab 中文 API 与使用样例

本文对应 v0.10.1。所有公开接口默认采用 SI 单位：光场为 `sqrt(W)`，功率为 W，电压为 V，时间为 s，频率为 Hz，长度为 m，相位为 rad，增益和损耗为功率 dB。

## 1. 核心与单位

### `SimulationGrid`

| 参数 | 含义 |
| --- | --- |
| `sample_rate_hz` | 复包络采样率，必须大于零 |
| `center_wavelength_m` | 中心光波长，默认 1550 nm |

提供采样间隔属性 `dt_s` 和 `frequencies_hz(samples)`；频率轴遵循 PyTorch FFT 顺序，并表示相对光载波的频偏。

### `units.py`

- `dbm_to_w`、`w_to_dbm`：dBm 与 W 转换。
- `db_to_power_ratio`：`10**(dB/10)`。
- `db_to_field_ratio`：`10**(dB/20)`。
- `field_from_power_w`、`field_power_w`：功率与复场幅度转换。

## 2. MZM、分束器和耦合器

### `PowerSplitter1x2`

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `splitting_ratio` | `0.5` | through 端功率比例，内部用 sigmoid 保持在 `(0,1)` |
| `insertion_loss_db` | `0` | 功率插损 dB |
| `trainable` | `True` | 是否训练分光比和插损 |

返回 `(through, cross)`，cross 端保留定向耦合器的 `j` 正交相位。

### `DirectionalCoupler2x2`

参数同分束器。输入两路复光场，返回两路复光场；无损时矩阵满足 `CᴴC=I`。

### `IdealMZM`、`PracticalMZM`、`MeasuredMZM`

- `IdealMZM(v_pi, bias_voltage, trainable)`：理想单入单出调制器。
- `PracticalMZM(v_pi, bias_voltage, insertion_loss_db, extinction_ratio_db, trainable)`：增加峰值插损和有限消光泄漏。
- `MeasuredMZM(voltage_points_v, power_transmission, phase_rad, trainable_curve)`：对实测电压-功率-相位曲线分段线性插值。
- `MeasuredMZM.from_csv()` 默认列名为 `voltage_v,power_transmission`，可选 `phase_rad`。

### `DifferentialDualPortMZM`、`FourPortMZM`

- `DifferentialDualPortMZM`：1 路光输入、push-pull 差分相移、2 路互补光输出。
- `FourPortMZM`：2 路光输入和 2 路光输出，由输入/输出 2x2 耦合器与差分相移组成。
- 两者均支持 `v_pi`、`bias_voltage`、`splitting_ratio`、`insertion_loss_db` 和 `trainable`。

### `IntensityMZM`

独立的无符号强度调制器，不调用差分 MZM。功率传输为：

```text
T(V) = eta * [epsilon + (1-epsilon) * cos²(pi*(V+Vbias)/(2*Vpi))]
```

参数为 `v_pi`、`bias_voltage`、`insertion_loss_db`、`extinction_ratio_db` 和 `trainable`。

### `IntensityMZMActivation`

把实值网络张量乘以 `input_scale_v_per_unit` 后送入 `IntensityMZM`，输出非负功率透射率。默认偏置位于斜率较大的非线性区，可用于与 ReLU 对照。

```python
activation = IntensityMZMActivation(
    v_pi=1.0,
    bias_voltage=-0.5,
    input_scale_v_per_unit=0.25,
)
y = activation(x)
```

## 3. 光源、滤波、光纤和 WaveShaper

### `OpticalFrequencyComb`

| 参数 | 含义 |
| --- | --- |
| `grid` | 仿真网格 |
| `line_spacing_hz` | 梳齿间隔 |
| `line_count` | 梳齿数量 |
| `line_powers` | 标量或逐梳齿功率 W |
| `line_phases_rad` | 逐梳齿相位 |
| `trainable` | 功率和相位是否训练 |

`channel_envelopes()` 返回分离的基带通道，适合 WDM 器件链；`forward()` 返回合成时域光梳并要求 FFT bin 对齐。

### `ElectroOpticComb`、`PracticalElectroOpticComb`

- 理想模型参数：`modulation_frequency_hz`、`modulation_indices_rad`、`rf_phases_rad`、`insertion_loss_db`。
- 实用模型额外加入 `rf_bandwidth_hz` 和 `rf_phase_noise_diffusion_rad2_per_s`。

### `FrequencyDomainLowPass`

参数：`grid`、`cutoff_hz`、`order`、`trainable`。截止频率定义为半功率点。

### `MZMWithElectricalFilter`

参数：`grid`、`cutoff_hz`、`order`、`filter_trainable`、`mzm`。它先用
`FrequencyDomainLowPass` 处理实驱动电压，再调用所选 MZM；省略 `mzm`
时使用实际解析 MZM。输入为复光场和实电压，输出为同形状复光场。

```python
device = MZMWithElectricalFilter(grid, cutoff_hz=20e9, order=2)
field_out = device(field_in, drive_voltage)
```

### `LinearDispersiveFiber`

| 参数 | 含义 |
| --- | --- |
| `length_m` | 光纤长度 m |
| `attenuation_db_per_km` | 功率衰减 dB/km |
| `beta2_s2_per_m` | 二阶色散 s²/m |
| `dispersion_ps_nm_km` | 可代替 beta2 的常用 D 参数 |
| `trainable` | 物理参数是否训练 |

`power_transmission(reference)` 返回可微的公共功率透射率；`adjacent_channel_delay_s()` 返回相邻 WDM 通道延时。

### `WaveShaper`

参数：`control_frequencies_hz`、`attenuation_db`、`phase_rad`、`trainable`。衰减是功率 dB，但作用到场上的系数是 `10**(-A/20)`。训练后应由上层调用物理参数投影，避免无源衰减变成负值。

## 4. 探测、噪声和放大器

### `PhotodetectorSum`

对形状 `(..., channels, samples)` 的 WDM 复场逐通道平方后求和，返回 W。`responsivity_a_per_w` 用于 `current_a()`。

### `NoisyPhotodetectorSum`

额外参数 `noise_std_w` 是平方律探测与求和之后的等效高斯读出噪声标准差，单位 W；它表示探测器/TIA/ADC 读出噪声，不表示光场 ASE。

### `BalancedPhotodetector`

输入正、负两组 WDM 光场，输出两路平方律功率之差。

### `OpticalAttenuator`、`AdditiveComplexGaussianNoise`

- 衰减器参数 `attenuation_db` 是功率 dB。
- 复高斯噪声参数 `sigma` 是复场 RMS，输出期望噪声功率为 `sigma**2`。

### `GainControlledEDFA`

推荐的商用级接口：`gain_db`、`gain_bandwidth_hz`、`gain_center_offset_hz`、`output_saturation_power_dbm`、`noise_figure_db`、`enable_saturation`、`enable_ase` 和 `trainable`。

`SmallSignalEDFA`、`SaturatedNoisyEDFA` 的泵浦电流接口仅为旧版本兼容，不建议用于新商用链路。

### `SmallSignalEDFA`、`SaturatedNoisyEDFA`（兼容接口）

- `SmallSignalEDFA` 参数：`grid`、`pump_current_ma`、`transparency_current_ma`、
  `gain_slope_db_per_ma`、`gain_bandwidth_hz`、`gain_center_offset_hz`、`trainable`。
- `SaturatedNoisyEDFA` 额外参数：`saturation_power`（与输入平均功率同单位）、
  `noise_figure_db`、`enable_ase`。其饱和关系为 `G_eff=G_ss/(1+P_in/P_sat)`。
- 新实验优先使用 `GainControlledEDFA`，避免把泵浦电流经验映射误当成器件通用规律。

```python
amplifier = GainControlledEDFA(grid, gain_db=20.0, gain_bandwidth_hz=40e9)
amplified_field = amplifier(field_in)
```

## 5. 链路与实值光卷积

### `OpticalChain`

- `OpticalChain(device1, device2, ...)`：简单串联。
- `OpticalChain.from_graph(nodes, outputs)`：显式命名数据流图。
- `OpticalChain.from_device_graph(seed, outputs=...)`：从器件 `connect()`、`connect_inputs()`、`connect_outputs()` 声明的端口连接自动发现拓扑。

器件图中的节点应继承 `GraphDevice`，通过 `graph_input_ports`、`graph_output_ports` 声明端口；`named(name)` 设置稳定节点名，`connect()`、`connect_inputs()`、`connect_outputs()` 建立连接。连接只保存拓扑元数据，`OpticalChain` 会检查重复输入和环路，并只注册每个器件一次。

### `PhotonicRealConvolution`

| 参数 | 含义 |
| --- | --- |
| `power_weights` | 单支路非负权重，范围 `[0,1]` |
| `sample_rate_hz` | 输入符号/采样率 |
| `line_spacing_hz` | 光梳间隔 |
| `line_power_w` | 每梳齿输入功率 |
| `tap_delay_samples` | 相邻抽头延时样本数 |
| `dispersion_ps_nm_km` | 自动设计光纤长度时使用的 D |
| `modulator` | 理想、实际或实测 MZM |
| `fiber` | 可选自定义匹配色散光纤 |
| `detector` | 可选理想或带噪功率探测器 |
| `trainable` | WaveShaper 权重是否训练 |

`forward()` 返回校准后的无量纲因果卷积；`detected_power_w()` 返回包含插损、消光底值、光纤损耗和探测噪声的功率读数；`forward_chunked()` 拼接长序列分块结果。需要真正限制训练峰值内存时使用 `iter_chunked()`，对每块立即计算损失并反向传播，不保留全部分块计算图。

本版还实际执行了 `batch=2`、每条 `200000` 点的有符号光卷积分块前向与反向传播；输出、损失、输入梯度和全部可训练器件参数梯度均为有限值。

### `DifferentialPhotonicConvolution`

使用两条独立的正/负物理支路和平衡探测差值表示有符号核。若原始核为 `w`、`s=max(1,max(abs(w)))`、差分深度为 `d`，初始化功率权重为 `p+=(1+d*w/s)/2`、`p-=(1-d*w/s)/2`，输出乘 `s/d`，因此 `(s/d)*(p+-p-)=w`。尺度下限 1 保证全零或极小初始核仍有可用动态范围。默认 `d=0.8` 使两支路都远离 0/1 边界，避免旧式正负截断在零点产生近乎消失的梯度，并允许训练时权重跨越零点。`effective_weights` 返回当前两路 WaveShaper 表示的有效核。

```python
x = torch.rand(16, 2, 100_000)
conv = DifferentialPhotonicConvolution(
    torch.tensor([-0.3, 0.2, 0.7]), differential_depth=0.8
)
optimizer = torch.optim.Adam(conv.parameters(), lr=1e-3)
optimizer.zero_grad()
for y_chunk in conv.iter_chunked(x, 8192):
    (y_chunk.square().sum() / x.numel()).backward()
optimizer.step()
conv.project_physical_parameters_()
```

## 6. 复值光学卷积

### `MZIComplexWeightBank`

输入复核形状为 `(out_channels, in_channels, kernel_size)`。内部参数：

- `theta_rad`：MZI 内部相移，控制 `cos(theta/2)` 幅度。
- `phi_rad`：外部相移，控制复权重相位。
- `weight_scale`：每个输出行的二范数标定常数。
- `insertion_loss_db`：公共场插损。
- `phase_error_std_rad`：每次前向的独立高斯相移误差。

训练后调用 `project_physical_parameters_()`：公共插损投影到非负区间；负的 `cos(theta/2)` 等价并入 `phi+pi`；每个输出行的归一化幅度二范数投影到不超过 1，保证无源 MZI 网络可实现。

### `MZIComplexConv1d`

参数：`complex_weights`、`tap_delay_samples`、`bias`、`insertion_loss_db`、`phase_error_std_rad`、`trainable`。输入为复数 `(batch,in_channels,samples)`；输出为同长度因果复卷积。`forward_chunked()` 保持因果历史和 autograd。

该模型依据 Nature Communications 12, 457 (2021) 中 MZI 内外相移实现复权重、相干探测恢复幅相的路线。它是器件参数化模型，不是波导 FDTD，也没有声称复刻特定芯片版图。

## 7. MNIST 光电混合模型

### `PhotonicMNISTClassifier`

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `image_size` | `28` | 方形灰度图边长 |
| `kernel_size` | `5` | 一维光卷积抽头数 |
| `mode` | `ideal` | `ideal` 或 `nonideal` |
| `activation` | `relu` | `relu` 或 `mzi` |
| `sample_rate_hz` | `100e9` | 串行采样率 |
| `line_spacing_hz` | `10e9` | 光梳间隔 |
| `detector_noise_std_w` | `1e-6` | 非理想模式探测读出噪声 |
| `chunk_size` | `None` | 长序列分块大小 |

输入可为 `(batch,1,28,28)` 或 `(batch,784)`，输出 `(batch,10)` logits。训练循环必须在 `optimizer.step()` 后调用 `model.project_physical_parameters_()`；该方法同时投影光卷积 WaveShaper 与强度 MZI 激活器的 `input_scale_v_per_unit`、`v_pi`、插损和消光比。

```powershell
python examples\train_mnist_photonic.py --mode ideal --activation relu --epochs 2
python examples\train_mnist_photonic.py --mode nonideal --activation relu --epochs 2
python examples\train_mnist_photonic.py --mode ideal --activation mzi --epochs 2
```

在固定随机种子、4096 条训练样本、1024 条测试样本、CPU、2 epoch 快速验收中，实测测试准确率分别为：理想/ReLU `77.83%`，非理想/ReLU `78.32%`，理想/强度 MZI `62.50%`。这些数字只用于证明三条链路可训练且损失下降，不是完整 MNIST 基准精度声明。

## 8. 兼容模块与系统指标

### `ComplexConv1d`

参数：`in_channels`、`out_channels`、`kernel_size`、`bias`。输入形状为
`(batch,in_channels,samples)` 的复张量。它用四次实卷积实现数学复卷积，
仅用于兼容，不代表物理复值光芯片。

### `OverlapSaveFIR`

参数：复数 `impulse_response`、`fft_size`、`trainable`。最后一维是样本；实现带零初始
状态的线性因果 FIR，不把循环卷积伪影带入有效输出。

```python
fir = OverlapSaveFIR(torch.tensor([1+0j, 0.2-0.1j]), fft_size=64)
y = fir(torch.ones(4, 4096, dtype=torch.complex64))
```

### 系统指标

- `mean_power(field)`、`power_spectrum(field)`、`centered_frequency_axis_hz(grid, samples)`。
- `snr_db(signal, noise)`、`normalized_mean_square_error(reference, estimate)`、
  `error_vector_magnitude(reference, estimate)`。
- `eye_diagram(signal, samples_per_symbol, offset_samples=0)`。
- `gaussian_q_factor(level_zero, level_one)`、
  `gaussian_ber_estimate(level_zero, level_one)`；二者假设二元电平近似高斯分布。

```python
evm = error_vector_magnitude(reference_symbols, received_symbols)
ber = gaussian_ber_estimate(samples_for_zero, samples_for_one)
```

## 9. 每个公开模型/模块的最短调用样例

下面的样例与前文参数表配套，覆盖包顶层导出的全部模型类。`GraphDevice` 是端口图基类，不应单独前向；它的 `named/connect/connect_inputs/connect_outputs` 用法见第 5 节。

```python
import torch
from optosimlab import *

grid = SimulationGrid(100e9)
field = torch.ones(100, dtype=torch.complex64) * 0.01
voltage = torch.linspace(-1, 1, 100)
wdm = field.repeat(3, 1)                       # (channels, samples)

# 分束、耦合和全部 MZM
splitter = PowerSplitter1x2(0.5, insertion_loss_db=0.2)
through, cross = splitter(field)
coupler = DirectionalCoupler2x2(0.5)
out0, out1 = coupler(through, cross)
bar, cross = DifferentialDualPortMZM(v_pi=1.0)(field, voltage)
out0, out1 = FourPortMZM(v_pi=1.0)(field, field, voltage)
y_ideal = IdealMZM(v_pi=1.0)(field, voltage)
y_practical = PracticalMZM(insertion_loss_db=3, extinction_ratio_db=20)(field, voltage)
y_alias = MachZehnderModulator()(field, voltage)
measured = MeasuredMZM(
    torch.tensor([-1.0, 0.0, 1.0]),
    torch.tensor([0.0, 0.5, 1.0]),
)
y_measured = measured(field, voltage)
y_filtered = MZMWithElectricalFilter(grid, cutoff_hz=20e9)(field, voltage)
y_intensity = IntensityMZM()(field, voltage)
y_activation = IntensityMZMActivation()(torch.randn(8, 16))

# 光源、传输、频域器件、噪声和探测
comb = OpticalFrequencyComb(grid, 10e9, 3, line_powers=1e-3)
comb_field = comb(100)
eoc = ElectroOpticComb(grid, 10e9, modulation_indices_rad=0.8)
eoc_field = eoc(field)
practical_eoc = PracticalElectroOpticComb(
    grid, 10e9, rf_bandwidth_hz=20e9,
    rf_phase_noise_diffusion_rad2_per_s=1e3,
)
practical_eoc_field = practical_eoc(field)
filtered = FrequencyDomainLowPass(grid, cutoff_hz=20e9, order=2)(field)
fiber_out = LinearDispersiveFiber(grid, 1000, dispersion_ps_nm_km=17)(field)
shaped = WaveShaper(grid, control_points=17)(field)
attenuated = OpticalAttenuator(3.0)(field)
noisy = AdditiveComplexGaussianNoise(sigma=1e-5)(field)
power_w = PhotodetectorSum()(wdm)
noisy_power_w = NoisyPhotodetectorSum(noise_std_w=1e-6)(wdm)
balanced_w = BalancedPhotodetector()(wdm, 0.5 * wdm)

# 三种 EDFA；新实验优先使用 GainControlledEDFA
amplified = GainControlledEDFA(grid, gain_db=20)(field)
small_signal = SmallSignalEDFA(grid, pump_current_ma=100)(field)
saturated = SaturatedNoisyEDFA(grid, saturation_power=1e-3)(field)

# 串行/图式组合和卷积模块
serial = OpticalChain(OpticalAttenuator(1.0), OpticalAttenuator(2.0))
serial_out = serial(field)
real_conv = PhotonicRealConvolution(torch.tensor([0.2, 0.5, 0.8]))
real_y = real_conv(torch.rand(4, 256))
signed_conv = DifferentialPhotonicConvolution(
    torch.tensor([-0.2, 0.5, 0.8]), differential_depth=0.8
)
signed_y = signed_conv(torch.rand(4, 256))
complex_weights = torch.tensor([[[0.3+0.2j, -0.1+0.4j]]])
bank = MZIComplexWeightBank(complex_weights)
bank_y = bank(torch.ones(2, 1, 32, 2, dtype=torch.complex64))
complex_conv = MZIComplexConv1d(complex_weights, tap_delay_samples=1)
complex_y = complex_conv(torch.ones(2, 1, 256, dtype=torch.complex64))
math_conv = ComplexConv1d(1, 2, 3, padding=1)
math_y = math_conv(torch.ones(2, 1, 256, dtype=torch.complex64))
fir = OverlapSaveFIR(torch.tensor([1+0j, 0.2-0.1j]), fft_size=64)
fir_y = fir(field)

# Issue #3 的混合光电 MNIST 模型
mnist = PhotonicMNISTClassifier(mode="nonideal", activation="mzi", chunk_size=256)
logits = mnist(torch.rand(8, 1, 28, 28))
```

所有带 `trainable=True` 的物理量都由 PyTorch 优化器管理。对含无源约束的光卷积、强度 MZI 或复 MZI 权重，必须在 `optimizer.step()` 后调用对应模型的 `project_physical_parameters_()`。

## 10. 最小自检命令

```powershell
# 在仓库根目录执行
.\.venv\Scripts\Activate.ps1
python -m compileall -q src tests examples
python -m pytest -q -p no:cacheprovider tests
python examples\train_mnist_photonic.py --synthetic-smoke --epochs 1
```

最终测试数量以仓库当前运行结果为准；跳过项仅应是无 CUDA 环境下的 GPU 一致性测试。真实 MNIST 下载失败属于网络问题，不代表模型前向/反向测试失败。
