"""Build the Chinese OptoSimLab usage manual PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "OptoSimLab_使用说明.pdf"


def build_styles() -> dict[str, ParagraphStyle]:
    pdfmetrics.registerFont(TTFont("YaHei", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
    base = getSampleStyleSheet()
    chinese = "YaHei"
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName=chinese, fontSize=24, leading=32, alignment=TA_CENTER, textColor=colors.HexColor("#12304A")),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontName=chinese, fontSize=11, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#52616B")),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName=chinese, fontSize=17, leading=25, spaceBefore=12, spaceAfter=8, textColor=colors.HexColor("#12304A")),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=chinese, fontSize=13, leading=20, spaceBefore=9, spaceAfter=5, textColor=colors.HexColor("#176B87")),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName=chinese, fontSize=9.5, leading=16, spaceAfter=6),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName=chinese, fontSize=8, leading=12, textColor=colors.HexColor("#4B5563")),
        "table": ParagraphStyle("table", parent=base["BodyText"], fontName=chinese, fontSize=8, leading=12),
        "callout": ParagraphStyle("callout", parent=base["BodyText"], fontName=chinese, fontSize=9.5, leading=16, leftIndent=8, rightIndent=8, borderColor=colors.HexColor("#8BC4D8"), borderWidth=0.8, borderPadding=8, backColor=colors.HexColor("#EEF8FB")),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def code(text: str) -> Preformatted:
    return Preformatted(text.strip(), ParagraphStyle("code", fontName="Courier", fontSize=7.5, leading=11, leftIndent=7, rightIndent=7, textColor=colors.HexColor("#16202A"), backColor=colors.HexColor("#F3F5F7"), borderColor=colors.HexColor("#D5DCE2"), borderWidth=0.4, borderPadding=7))


def page_number(canvas, document) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#C7D1D8"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("YaHei", 8)
    canvas.setFillColor(colors.HexColor("#52616B"))
    canvas.drawString(18 * mm, 8.5 * mm, "OptoSimLab | 可微分光信号级仿真平台")
    canvas.drawRightString(192 * mm, 8.5 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def status_table(styles: dict[str, ParagraphStyle]) -> Table:
    rows = [
        ["模块", "当前交付", "验证状态"],
        ["核心采样", "SimulationGrid、FFT 频率轴与物理常数", "运行时测试通过"],
        ["MZM", "Vpi、偏置、插损、有限消光比、可训练参数", "峰值/消光/梯度测试通过；公式核验通过"],
        ["低通", "零相位 Butterworth 幅度响应", "解析增益与 -3 dB 功率点测试通过；公式核验通过"],
        ["线性光纤", "功率损耗、beta2 或 D 参数、二阶色散", "单位换算/能量守恒/相位测试通过；公式核验通过"],
        ["WaveShaper", "可训练频域衰减与相位控制点", "恒定衰减和相位测试通过；公式核验通过"],
        ["频域光梳", "离散谱线、每线功率/相位、严格 FFT bin 对齐", "位置、总功率、梯度与非法输入测试通过"],
        ["电光梳", "理想与 RF 一阶响应、Wiener 相位噪声、级联驱动", "Bessel/带宽/相噪/功率/梯度与非法输入测试通过"],
        ["小信号 EDFA", "电流-增益映射、Gaussian 功率增益带宽、插损", "峰值、-3 dB 带宽、失谐、梯度与非法输入测试通过"],
        ["饱和/ASE EDFA", "标量饱和、噪声系数、ASE 频谱随机噪声", "饱和、PSD/FFT 归一化、梯度与非法输入测试通过"],
        ["非理想与模块", "衰减、复高斯噪声、复卷积、器件链、四入单出 MZI", "运行时测试通过；复卷积偏置重复计入问题已修正"],
        ["系统指标", "Parseval 频谱、眼图、EVM、Gaussian-Q BER", "频谱/眼图/解析 EVM 与 BER 测试通过"],
        ["长序列", "OverlapSaveFIR、可训练复 FIR、线性因果卷积", "直接卷积/批量/梯度/块几何测试通过"],
        ["GPU 门禁", "核心模块 CPU/CUDA 一致性比较", "当前 CPU 环境明确跳过；CUDA 环境自动执行"],
        ["后续扩展", "实验标定与端到端训练示例", "明确列为后续范围，未伪称已实现"],
    ]
    table = Table([[p(cell, styles["table"]) for cell in row] for row in rows], colWidths=[28 * mm, 83 * mm, 63 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B87")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D1D8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build() -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=20 * mm, title="OptoSimLab 使用说明")
    story = []
    story += [Spacer(1, 28 * mm), p("OptoSimLab", styles["title"]), p("可微分光信号级仿真平台 使用说明与自检记录", styles["subtitle"]), Spacer(1, 12 * mm)]
    story += [p("版本 0.7.0 | 工程位置：F:\\OptoSimLab | 更新日期：2026-08-25", styles["callout"]), Spacer(1, 12 * mm)]
    story += [p("阅读对象", styles["h2"]), p("面向需要在 PyTorch 中搭建光通信链路、可训练光计算模块或光神经网络的开发者。本文将系统明确为信号级仿真器：它模拟复包络、器件传输函数和可微参数，不求解纳米尺度电磁场。", styles["body"])]
    story += [p("快速判断", styles["h2"]), p("若你需要调制器偏置优化、光纤色散、可编程频谱整形和端到端梯度训练，本项目适用。若你需要波导几何、模式求解、耦合效率或 FDTD 电磁场分布，应使用专门的电磁仿真工具，而不是本项目。", styles["body"]), PageBreak()]

    story += [p("1. 范围与当前交付", styles["h1"]), p("平台遵循“一个器件一个 PyTorch Module”的封装原则。所有可调的物理量都注册为 nn.Parameter，以便被 optimizer、state_dict 和 device 迁移机制识别。", styles["body"]), status_table(styles), Spacer(1, 6 * mm)]
    story += [p("真实性说明", styles["h2"]), p("PyTorch、NumPy 与 pytest 已通过项目虚拟环境安装。当前版本在 Windows CPU 环境中执行 64 项测试全部通过，另有 1 项 CUDA 一致性测试因本机为 CPU 版 PyTorch 而明确跳过。每个模块还执行 Python 语法编译；关键物理公式另外做了独立算术核验。新增模块仍必须先通过专项测试，再通过完整回归测试。", styles["callout"]), PageBreak()]

    story += [p("2. 安装与第一个运行", styles["h1"]), p("需要 Python 3.10 或更高版本，以及 PyTorch 2.2 或更高版本。建议始终在项目虚拟环境中安装，避免把 GPU 或 CUDA 版本混入系统 Python。", styles["body"])]
    story += [code(r'''
cd F:\OptoSimLab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python examples\train_mzm_bias.py
''')]
    story += [p("最后一条示例把 MZM 偏置从初值优化到平均输出功率为 0.5 的工作点。它只把 bias_voltage 交给 Adam，其他已注册物理参数保持不变，从而演示“可训练”不等于“所有参数都必须同时训练”。", styles["body"])]
    story += [p("最小链路", styles["h2"]), code('''import torch
from optosimlab import SimulationGrid, MachZehnderModulator
from optosimlab import FrequencyDomainLowPass, LinearDispersiveFiber

grid = SimulationGrid(sample_rate_hz=100e9)
carrier = torch.ones(4096, dtype=torch.complex64)
drive = 0.5 * torch.sin(2 * torch.pi * 5e9 * torch.arange(4096) / grid.sample_rate_hz)
mzm = MachZehnderModulator(v_pi=1.0, bias_voltage=0.5)
fiber = LinearDispersiveFiber(grid, length_m=10e3, attenuation_db_per_km=0.2, dispersion_ps_nm_km=17.0)
field_out = fiber(mzm(carrier, drive))
'''), PageBreak()]

    story += [p("3. 信号与公式约定", styles["h1"]), p("时域光场使用复包络 E(t)。不直接采样 THz 量级的光载波；因此频率 f 代表相对中心载波的偏移。频域设备采用 PyTorch 的 FFT 约定：正变换含 exp(-j 2 pi f t)，反变换含 exp(+j 2 pi f t)。", styles["body"])]
    story += [p("采样网格", styles["h2"]), p("SimulationGrid(sample_rate_hz, center_wavelength_m) 是频域器件唯一的采样约定来源。它输出未移位的 FFT bin 顺序：0、正频率、负频率。需要以载波为中心的可视化或 WaveShaper 控制点，内部使用 fftshift。不要在器件内复制全局采样率常量。", styles["body"])]
    story += [p("功率", styles["h2"]), p("在归一化仿真中，瞬时功率与 |E|² 成正比。所有 dB 损耗都是功率 dB：功率系数为 10^(-L_dB/10)，因此场幅度系数为 10^(-L_dB/20)。这是本项目检查过的常见错误点。", styles["body"])]
    story += [p("MZM", styles["h2"]), p("理想推挽 MZM 使用 E_out = E_in cos[pi(V+V_bias)/(2 V_pi)]。有限消光比 ER 采用正交泄漏项，谷/峰功率比严格为 10^(-ER/10)。插损 eta 以 sqrt(eta) 作用于场，而不是以 eta 作用于场。", styles["body"])]
    story += [p("时域电光梳", styles["h2"]), p("ElectroOpticComb 级联相位调制器：E_out(t)=E_in(t) exp[j Σ beta_k sin(2 pi f_k t + phi_k)] 10^(-L_dB/20)。beta_k 是调制深度（rad），phi_k 是 RF 初相；无插损时指数因子模长恒为 1，故逐点功率和平均功率均守恒。对连续波，Jacobi-Anger 展开给出第 n 根梳齿的场系数 J_n(beta) exp(j n phi)，频率偏移为 n f。为使有限记录内的频谱严格周期化，f 必须为 sample_rate/samples 的整数倍且低于 Nyquist 频率。当前第一版只表示理想、相干的正弦 RF 驱动；相噪、响应曲线和电带宽尚未建模。", styles["body"])]
    story += [p("实用 RF 电光梳", styles["h2"]), p("PracticalElectroOpticComb 用 H_rf=1/(1+j f/f_c) 表示每级一阶 RF 响应，故调制深度乘以 |H_rf|，RF 相位叠加 -atan(f/f_c)。其可选 Wiener 相位噪声满足 Var[phi(n+1)-phi(n)]=D_phi dt，D_phi 的单位为 rad²/s；通过传入固定随机数生成器可复现实验。它是可控的相位扩散近似，不应被解释为完整的实际振荡器相噪掩膜。", styles["body"])]
    story += [p("小信号 EDFA", styles["h2"]), p("SmallSignalEDFA 的峰值净功率增益为 G_peak_dB=slope×max(I_pump-I_transparency,0)-L。频率响应采用 G_power(f)=10^(G_peak_dB/10) exp[-4 ln(2)((f-f0)/B)^2]，因此 f0±B/2 处功率增益恰好是峰值的一半（相对 -3.0103 dB）；光场乘以 sqrt(G_power)，不引入额外相位。本版只覆盖确定性小信号线性区：不包含 ASE、饱和、噪声系数、泵浦动态或偏振依赖。", styles["body"])]
    story += [p("饱和与 ASE EDFA", styles["h2"]), p("SaturatedNoisyEDFA 使用 G_eff=G_ss/[1+P_in/P_sat] 压缩小信号增益；P_in 为输入平均功率，P_sat 为可训练饱和功率。它采用一偏振复包络、双边 PSD 约定：S_ASE=n_sp h nu max(G_eff-1,0)，n_sp=10^(NF_dB/10)/2。离散频谱 bin 的方差取 N^2 S_ASE df，故 IFFT 后期望平均噪声功率严格等于 sum(S_ASE df)。这是一种可微分的信号级现象学模型，不代表掺铒离子布居、泵浦瞬态或偏振动力学。", styles["body"])]
    story += [p("低通与光纤", styles["h2"]), p("低通采用零相位 Butterworth 幅度响应 H(f)=[1+(|f|/f_c)^(2n)]^(-1/2)，在 f_c 处功率降为一半。线性光纤使用 H(f)=exp(-alpha_power L/2) exp(+j beta2 (2 pi f)^2 L/2)。D 和 beta2 的换算为 beta2=-D lambda²/(2 pi c)。", styles["body"])]

    story += [p("4. 器件 API", styles["h1"])]
    api_rows = [
        ["类", "关键参数", "输入与输出"],
        ["MachZehnderModulator", "v_pi, bias_voltage, insertion_loss_db, extinction_ratio_db", "(complex carrier, real voltage) -> complex field"],
        ["FrequencyDomainLowPass", "grid, cutoff_hz, order", "complex field -> complex field"],
        ["MZMWithElectricalFilter", "grid, cutoff_hz, mzm", "先对 real voltage 低通，再调制 carrier"],
        ["LinearDispersiveFiber", "length_m, attenuation_db_per_km, beta2 或 D", "complex field -> complex field"],
        ["OpticalFrequencyComb", "grid, line_spacing_hz, line_powers, line_phases_rad", "samples -> complex field"],
        ["ElectroOpticComb", "grid, modulation_frequency_hz, modulation_indices_rad, rf_phases_rad", "complex field -> complex field"],
        ["PracticalElectroOpticComb", "ElectroOpticComb 参数, rf_bandwidth_hz, phase diffusion", "complex field -> complex field"],
        ["SmallSignalEDFA", "grid, pump_current_ma, gain_slope_db_per_ma, gain_bandwidth_hz", "complex field -> complex field"],
        ["SaturatedNoisyEDFA", "SmallSignalEDFA 参数, saturation_power, noise_figure_db", "complex field -> complex field (含可选 ASE)"],
        ["OverlapSaveFIR", "complex impulse_response, fft_size", "complex field -> 线性因果卷积"],
        ["WaveShaper", "control_points, attenuation_db, phase_rad", "fftshift 轴上的可训练频谱遮罩"],
        ["FourInputMZI", "4 个 phase_rad, insertion_loss_db", "(..., 4, samples) -> (..., samples)"],
        ["OpticalAttenuator / Noise", "attenuation_db / sigma", "非理想损耗或复高斯噪声"],
    ]
    api = Table([[p(cell, styles["table"]) for cell in row] for row in api_rows], colWidths=[43 * mm, 68 * mm, 63 * mm], repeatRows=1)
    api.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B87")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#C7D1D8")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [api, Spacer(1, 7 * mm)]
    story += [p("组合与复值模块", styles["h2"]), p("OpticalChain 以 ModuleList 顺序串联只接收一个 field 参数的器件，例如光纤、WaveShaper 和衰减器。调制器因为额外需要电压输入，应在自定义 nn.Module.forward 中显式调用。ComplexConv1d 接收形状为 (batch, channel, samples) 的复数张量，并以四次实卷积实现复卷积。其偏置只加一次到对应实部或虚部，避免双加偏置。", styles["body"]), PageBreak()]

    story += [p("5. 训练与物理约束", styles["h1"]), p("参数确实注册为 nn.Parameter，但某些量必须满足物理边界：Vpi、截止频率、长度、衰减和噪声标准差不可为负。当前版本在前向计算时检测非法的负参数并抛出错误，避免悄悄跑出不物理结果。训练这些正值参数时，建议设置较小学习率并使用参数化约束或优化后投影。", styles["body"])]
    story += [code('''mzm = MachZehnderModulator(v_pi=1.0, bias_voltage=0.1)
optimizer = torch.optim.Adam([mzm.bias_voltage], lr=3e-2)

optimizer.zero_grad()
loss = (mzm(carrier, drive).abs().square().mean() - 0.5).square()
loss.backward()
optimizer.step()
''')]
    story += [p("参数校准", styles["h2"]), p("把实验测得的 V-I 或 V-传输曲线拟合为器件初值，然后冻结或微调。对实际 MZM，优先标定 Vpi、偏置、插损、消光比和电带宽。对光纤，明确 D 的单位是 ps/(nm km)，长度是 m，中心波长默认 1550 nm。", styles["body"])]
    story += [p("数值注意事项", styles["h2"]), p("频域滤波默认对应循环卷积。短记录中要为脉冲留出零填充，或者扩展为 overlap-save。过窄的滤波器、极长光纤或极高频率会增大相位变化，需提高采样率和记录长度以防混叠。", styles["body"]), PageBreak()]

    story += [p("6. 模块自检清单", styles["h1"]), p("每增加一个器件模块，至少完成以下闭环，才允许它进入主链路。", styles["body"])]
    checks = [
        "1. 写清输入/输出张量形状、复数或实数 dtype、单位和 FFT 约定。",
        "2. 把可调物理量封装为 nn.Parameter，并显式处理物理取值范围。",
        "3. 写解析可验证点：例如 MZM 的峰值/零点，滤波器的截止点，光纤的无损能量守恒。",
        "4. 写失败用例：错误 dtype、错误样本维度、非法负参数或相互冲突的物理输入。",
        "5. 做梯度检查：至少确认目标参数的 grad 非空；高风险公式用有限差分对照。",
        "6. 在 CPU 执行后，再在 GPU 上比较误差；最后测试批量维度和不同 sample 数。",
    ]
    story += [p(item, styles["body"]) for item in checks]
    story += [p("当前已做的自检", styles["h2"]), p("全部源码和测试已用 compileall 编译，并在 Windows CPU 环境中通过 64 项运行时测试，另有 1 项 CUDA 一致性测试按环境明确跳过。已独立核对：MZM 的 3 dB 插损与 20 dB 消光功率关系；Butterworth 截止点；D-beta2 的单位往返和 10 dB/km 损耗；复高斯噪声两正交分量方差；WaveShaper 的幅度与相位；四路 MZI 的 1/2 幅度归一化；频域光梳的 FFT bin 位置、谱线总功率和可训练相位；理想与实用电光梳的 Bessel 边带、RF 一阶响应、Wiener 相位扩散和功率守恒；两类 EDFA 的电流映射、带宽、饱和、ASE PSD 与 FFT 归一化随机统计；系统指标的 Parseval 频谱、眼图折叠、EVM、Gaussian-Q BER；OverlapSaveFIR 与直接线性卷积、批量末块和梯度的一致性。", styles["body"]), PageBreak()]

    story += [p("7. 运行时验证与下一步", styles["h1"]), p("每次新增器件前后都运行下面命令。当前 CPU 环境预期为 64 项测试通过和 1 项 CUDA 测试跳过；若失败，先按测试名定位到单一模块，不在未通过时继续叠加新器件。", styles["body"]), code('''cd F:\\OptoSimLab
.\\.venv\\Scripts\\Activate.ps1
python -m pytest
python examples\\train_mzm_bias.py
''')]
    story += [p("建议的下一批模块", styles["h2"]), p("器件库、实用 RF 电光梳、频谱/眼图/EVM/BER、长序列 overlap-save 和 CUDA 一致性门禁均已交付。后续属于实验适配：器件测量曲线标定和端到端小型光计算训练示例。每一步都必须附带可查的物理来源或实验标定数据。", styles["body"])]
    story += [p("交付结构", styles["h2"]), code('''F:\\OptoSimLab
  src\\optosimlab\\   核心、器件、功能模块、指标
  tests\\              每个模块的单元测试
  examples\\           可微 MZM 偏置训练示例
  docs\\               本 PDF 及其生成源
  pyproject.toml        可安装项目与依赖声明
''')]
    story += [p("结论", styles["h2"]), p("当前版本已完成器件库和功能模块清单：每个已交付模块均保存明确的公式、边界和测试，运行时测试已全绿。后续开发应转向系统级可靠性与实验标定，并继续坚持“公式、封装、专项测试、全量回归”的门禁。", styles["callout"])]

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(OUTPUT)


if __name__ == "__main__":
    build()
