# SAEDIndex 可靠性与 FFT 扩展 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> 本项目执行约定：本轮只制定计划。后续采用当前会话逐项执行，可用技能为 `executing-plans`；未获用户或适用指令授权时不派生子代理。该目录当前不是 Git 仓库，不创建 worktree，不执行 Git 提交。

**Goal:** 先提高衍射图自动分组和标定结果的可核对性，再接入 HRTEM FFT、两相取向关系和经仪器校准的倾转辅助。

**Architecture:** 保留现有 PySide6 桌面、Session 数据模型、CIF 几何计算与论文出图链路。按独立阶段交付，新增科学计算放入无 Qt 的模块，界面负责输入、核对和显示；测量来源、算法参数、确认状态随项目保存。

**Tech Stack:** 现有 Python、PySide6、NumPy、SciPy、Pillow、tifffile 与 PyInstaller。FFT 使用 NumPy/SciPy；首轮不新增依赖。

---

## 1. 本轮范围与已核实基础

- 本文是分阶段路线图和验收计划，后续每阶段单独展开实现任务并验证，不一次性实施全部功能。
- 代码目录：`D:\科研论文\SAEDIndex`。
- 已有标尺识别证据显示、人工定标、固定比例 CIF 叠加、手工建组删组、标注拖动、撤销重做、标记显隐、PNG/TIFF/CSV 导出和项目保存。
- 已有 `dist\SAEDIndex\SAEDIndex.exe` 和 `dist\SAEDIndex-win64.zip`；本轮不覆盖。
- `progress.md` 记录上一轮 61 项测试通过。本次仅查阅代码和文档，没有重新运行测试，不把该记录当作新一轮测试结果。
- `pattern.py` 已有多点最小二乘晶格拟合和顺序分组；主要增量是异常点处理、多组接受证据及误分组控制。
- `indexing.py` 已比较角度、长度比例、共同尺度偏差、允许反射和缺失反射；分数是排序指标，不是概率。
- 当前 `Session.px_per_unit` 是单一倒易比例；`dmfile.read_dm()` 当前只返回第一维标定，接入 FFT 前必须核对二维像素尺度。
- 当前 `project.py` 只接受版本 1；增加 FFT 来源和新状态时需要兼容旧项目。

## 2. 借鉴依据

已核实的官方页面及由出版方提交的论文摘要：

1. Klinger & Jäger, 2015, *Crystallographic Tool Box (CrysTBox): automated tools for transmission electron microscopists and crystallographers*, Journal of Applied Crystallography 48, 2012–2018，DOI `10.1107/S1600576715017252`。本次经 Crossref 读取出版方摘要，确认 `diffractGUI` 的晶带轴及反射索引、`ringGUI` 的环分析，以及 `diffractGUI` 与 `cellViewer` 配合计算目标晶带轴倾转。全文访问受限，未核实具体算法章节；RANSAC 等仅作为本项目待比较的方法，不据此归因于原文。
2. Klinger, 2017, *More features, more tools, more CrysTBox*, Journal of Applied Crystallography 50, 1226–1234，DOI `10.1107/S1600576717006793`。本次核实摘要中的双相界面可视化、几何相位分析和可视化扩展；不据此声称其已有本计划提出的自动两相取向关系工作流。
3. NumPy 官方 `numpy.fft.fftfreq` 文档确认频率采样间隔为 `1 / (N * sample_spacing)`，用于 FFT 单位与数值回归。
4. FZU 官方 `CrysTBox diffractGUI - Diffraction and HRTEM analysis` 页面确认 HRTEM/衍射图输入、晶面间距/夹角测量及每一步可手工调整的工作流。

实际读取地址记录：

```text
https://www.fzu.cz/en/crystbox
https://www.fzu.cz/en/crystbox_diffractgui
https://api.crossref.org/works/10.1107/S1600576715017252
https://api.crossref.org/works/10.1107/S1600576717006793
https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
```

采用公开方法和工作流思路，使用本项目代码独立实现。没有安装运行 CrysTBox 做同图对测，也没有核实其当前版本的全部能力；此处不作性能优劣或精度等同承诺。此前提及的独立 `tiltGUI` 名称未得到上述来源支持，计划统一称“倾转辅助”。

## 3. 交付顺序

| 阶段 | 用户得到的能力 | 验收重点 | 优先级 |
| --- | --- | --- | --- |
| P0 基准 | 固定的合成样本、人工复核样本和旧版基线 | 开发集、留出集分开；有明确真值或歧义标记 | 必须先做 |
| P1 可靠性 | 更稳的组数判断、可解释的候选及残差图 | 单组误分、多组漏检、标尺冲突、矢量闭合 | 第一版 |
| P2 FFT | 框选 HRTEM 区域后完成 FFT 与 CIF 标定 | 实空间/倒易空间换算、中心、处理来源、项目往返 | 第二版 |
| P3 取向关系 | 两组已确认标定间的平行关系及偏差 | 晶体坐标、屏幕坐标、对称等价、多解 | 第三版 |
| P4 倾转辅助 | 从当前取向到目标晶带轴的理论倾角 | 样品杆轴向、旋转顺序、正负号、限位、回算 | 仪器参数齐备后 |
| P5 专项扩展 | 衍射环、CBED 圆盘、晶体可视化 | 分别建立数据与验收集 | 按实际数据需求排期 |

推荐先交付 P0+P1，再交付 P2。P3 使用已有多组数据，优先于依赖仪器参数的 P4。各阶段工期在该阶段基线和样本核对后确定，不用未经验证的准确率或时间承诺代替验收。

## 4. 公共约束

- 科学定标与出图标尺保持独立；CIF 尺度拟合仅作为偏差诊断，用户显式应用校准后才改变测量比例，并记录来源、撤销及重新确认。
- 亮度、对比度、伽马仅影响显示。背景扣除若参与检测，记录参数并保留原始数据；标记处理后的强度来源。
- 保留检测点、晶格预测点的区别；仅使用独立实测点评价矢量闭合。
- 自动结果维持候选状态。缺少比例、中心或独立斑点证据时，显示具体原因。
- 新分析操作与现有撤销、标记显隐、待应用标尺保护、干净导出一起验收。
- 科学参数变更使相关确认失效；仅移动标注或改字体不改变科学结果。
- 缺失、变动的源图或 CIF 继续由来源哈希检查拦截；旧项目须可读取。
- 新计算遵循现有后台 worker 模式；重分析结果不可覆盖后来打开的图或更新后的 ROI。

## 5. P0：建立可比较基线

实施状态：合成基准和 8 张按系列划分的实测对照已建立，见
`verification/reliability-validation.md`。实测图组数/物相真值仍待材料实验人员复核，
不将这一项标记完成，也不报告实测识别准确率。

**文件：** 新建 `tests/test_grouping.py`、`tests/fixtures/pattern_cases.py`、`tests/verify_accuracy.py`；验证输出放 `verification/accuracy-baseline/`。

- [x] 阅读现有 `test_core.py`、`test_workbench.py`、`test_editing.py`，沿用 `unittest` 和合成 Session 的做法。
- [x] 用固定随机种子生成：完整单组、缺点单组、噪点单组、饱和中心、被遮挡中心、旋转双组、共用节点双组、接近共线的稀疏点及无晶格噪声。
- [x] 每个合成样本保存真实中心、基矢、组别、噪声参数和遮挡区域。评估时按预测节点匹配，不直接比较可能对称等价的基矢编号。
- [x] 从现有实测图整理开发集合；按拍摄系列划分独立留出集合，避免同一图的裁剪或亮度副本跨集合。
- [ ] 人工复核每张图的组数、中心和可接受候选集合；难以唯一判断的图标记“有歧义”，不强行指定唯一物相。
- [ ] 同一配置下记录旧版的误分组率、多组漏检率、待核对率、候选覆盖、尺度误差、角度误差及耗时。

**验收：** 基线可重复生成；报告分清自动分组正确率、标尺识别误差和物相候选表现。已有 155 张图的标尺统计不充当物相准确率。

## 6. P1：可靠性与证据展示

实施状态：已实现稳健分组、诊断页、评分分项、固定尺度残差叠加、标尺冲突门控、
可撤销 CIF 辅助比例及诊断导出。重复位移候选与确定性多候选重拟合解决了已复现病例，
本次未增加另一套随机 RANSAC 求解器；真实数据科学验收依赖上述人工真值复核。

**文件：** 修改 `saedindex/pattern.py`、`saedindex/indexing.py`、`saedindex/session.py`、`saedindex/window.py`、`saedindex/canvas.py`、`saedindex/project.py`；新增 `saedindex/diagnostics.py`、`tests/test_diagnostics.py`；扩展 `tests/test_grouping.py`、`tests/test_editing.py`、`tests/test_project.py`。

- [x] 先以 P0 病例复现单组误拆和第二组漏检，记录触发条件；对没有复现的猜测不直接调整阈值。
- [ ] 在现有全图多点拟合基础上比较稳健重拟合与有固定种子的 RANSAC 候选策略。复用现有 NumPy/SciPy 求解，记录内点与异常点，优先保留验证更稳定的方案。
- [x] 第二组接受检查包括新增独立实测点数、二维非共线性、空间覆盖、节点残差及与所有已有组的重复性。保留重合点的多组解释，防止硬分配吞掉真实第二组。
- [x] 重复晶格检查覆盖所有已有组；基矢方向变化和整数换基不能产生假新组。不能仅用“能落在更细网格上”就合并真实组。
- [x] 比较单组与多组模型，展示“接受第 2 组”的具体证据。证据不足的弱组列为待核对，保留手工建组、删组和组数上限。
- [x] 新增诊断模块生成实测支持点数、覆盖率、RMS、角度误差、d 间距误差、共同尺度偏差、实测闭合残差以及候选接近程度。
- [ ] 检查 `indexing.evaluate()` 中强度/缺失反射项对排序的影响，输出各项贡献；避免强度近似掩盖几何冲突。
- [x] 在现有侧栏显示“定标 → 检测 → 晶格 → CIF → 确认”的状态、冲突原因及可点击的异常点。CIF 叠加提供匹配/未匹配点区分和残差显示开关。
- [x] 标尺识别与 DM 元数据明显冲突时转为待核对，显示双方数值。拟合建议与原比例并列，应用后重新索引并可撤销。
- [x] 诊断图层随标记总开关隐藏，不进入论文导出；验证分组调整和定标调整的撤销重做。

交付补充：RANSAC 对照未实施，采用上文说明的确定性稳健重拟合；评分分项已输出，
但强度项对真实物相排序的科学验证仍待人工真值。上述两项保留未勾选。
84 项单元测试、原生界面回归和打包启动检查已通过；免安装包已更新并保留旧版备份。

**首批测试写法示例，代码只写入计划，执行阶段再建立测试文件：**

```python
import unittest
import numpy as np
from saedindex.pattern import Spot, find_lattices


class GroupingAcceptanceTests(unittest.TestCase):
    def test_complete_single_lattice_stays_one_group(self):
        center = np.array([256.0, 256.0])
        g1 = np.array([44.0, 3.0])
        g2 = np.array([10.0, -51.0])
        spots = [Spot(*(center + i * g1 + j * g2), 100.0)
                 for i in range(-3, 4) for j in range(-3, 4)
                 if (i, j) != (0, 0)]
        groups = find_lattices(spots, center, max_sets=3)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].members), len(spots))
```

该测试保护基础行为；实际算法修改必须有 P0 中对应的失败病例。禁止只靠降低默认组数让单组样本通过。

**验收：** 所有明确单组的确定性合成病例保持一组；旋转双组与共享点双组保留两组；纯噪声不自动确认。留出实测集合同时报告误分和漏检，不能以增加漏检换取表面上的少分组。低置信度原因能在图中复核。

## 7. P2：HRTEM FFT 标定

**文件：** 新建 `saedindex/fft.py`、`saedindex/fft_panel.py`、`tests/test_fft.py`、`tests/verify_fft.py`；修改 `saedindex/calibration.py`、`saedindex/dmfile.py`、`saedindex/session.py`、`saedindex/indexing.py`、`saedindex/project.py`、`saedindex/window.py`、`saedindex/canvas.py`、`saedindex/rendering.py`。

- [ ] 首批测试定义三类源图：有可信实空间元数据、有烧录实空间标尺、没有可信比例。确保原有 SAED 路径仍拒绝把 `nm` 当成 `1/nm`。
- [ ] 在 DM 读取的 `info` 中增加二维像素尺寸与单位，保留 `read_dm()` 现有返回值，避免破坏衍射图读取调用。逐轴归一为 `nm/px`。
- [ ] 实空间标尺支持自动读取、原图裁片核对、手工端点/长度/单位编辑。没有可靠比例时可显示 FFT，但不产生已定标的绝对 d 间距。
- [ ] 第一版采用方形 ROI；对已知各向异性像素禁用标定并说明原因。普通图片没有二维元数据时明确记录等比例像素假设，经过核对后使用。矩形 ROI 和各向异性度量作为后续独立任务。
- [ ] 定义 `compute_fft(raw, roi, nm_per_px, window="hann")`：输入原始二维数据和 `(x, y, size)`，验证有限正比例、ROI 边界和最小尺寸；返回频谱数据、显示灰度、固定零频中心、倒易比例和处理参数。
- [ ] 去均值、Hann 窗与 NumPy FFT 使用统一处理顺序。检测使用数值频谱，显示使用对数拉伸；亮度拖动不能更改频率坐标或检测输入。首版不做重采样和零填充。
- [ ] 按 `dg = 1 / (N * nm_per_px)` 换算为 `1/nm`，复用现有接口时 `px_per_unit = N * nm_per_px`。中心固定为 `fftshift` 的零频索引，不调用 SAED 透射束搜索。
- [ ] FFT 来源通过明确的分析类型进入 Session；`indexing.py` 对 FFT 采用几何主导排序，停用直接沿用 SAED 反射强度和缺失强峰的罚分，保留该来源的判读限制。
- [ ] UI 提供原图、ROI 与 FFT 切换；默认 ROI 避开烧录标尺和文字。记录 ROI 及加窗方式，原图始终保留。
- [ ] 支持 ROI/实空间标尺修改后重算、确认失效与撤销重做。历史存储来源和参数，避免每次拖动复制整张 FFT 数组。
- [ ] 项目格式升级到版本 2，继续读取版本 1；保存源图引用、ROI、实空间标定、FFT 参数和处理版本，打开时验证来源并重建派生图。
- [ ] 复用标注拖动、上划线、字体线宽与干净导出；FFT 图标尺显示倒易单位，原图显示实空间单位。

**核心数值验收向量：** `N=256`，像素尺寸 `0.02 nm/px`，x/y 方向整数频率分别为 16/32 时，峰值为 `3.125/6.25 1/nm`，相应 d 间距为 `0.32/0.16 nm`，夹角为 90 度。以下原始信号用于独立验证：

```python
import numpy as np

n = 256
pixel_nm = 0.02
yy, xx = np.mgrid[:n, :n]
raw = (100.0 + 10.0 * np.cos(2.0 * np.pi * 16.0 * xx / n)
       + 8.0 * np.cos(2.0 * np.pi * 32.0 * yy / n))
power = np.abs(np.fft.fftshift(np.fft.fft2(raw - raw.mean()))) ** 2
freq = np.fft.fftshift(np.fft.fftfreq(n, d=pixel_nm))
assert np.isclose(abs(freq[n // 2 + 16]), 3.125)
assert np.isclose(abs(freq[n // 2 + 32]), 6.25)
assert np.argmax(power[n // 2]) in (n // 2 - 16, n // 2 + 16)
```

**验收：** 整数频率合成样本的峰坐标和单位通过独立数值比较；另测奇数 ROI、斜向条纹、裁剪后比例、无标尺、错误单位、非有限输入与各向异性元数据。至少一张有可信标定的真实 HRTEM 图完成“ROI → FFT → 标定 → 保存重开 → 干净导出”；真实图需人工复核后才作为科学验收通过。

## 8. P3：两相取向关系

**文件：** 新建 `saedindex/orientation.py`、`tests/test_orientation.py`；修改 `saedindex/session.py`、`saedindex/window.py`、`saedindex/project.py` 及 CSV 导出逻辑。

- [ ] 从同一张图中选两组已确认结果，各组需要两个不共线、独立实测的已索引向量；缺少条件时返回具体原因。
- [ ] 使用 CIF 晶胞度量转换晶体向量，统一屏幕 y 轴、手性、束向和旋转矩阵约定。采用 SciPy 旋转工具和 NumPy 线性代数计算，矩阵定义写入函数文档。
- [ ] 在输入假设成立的情况下拟合各组取向，显示候选平行晶面、平行晶向及夹角偏差，并生成两相关系表。
- [ ] 枚举已知晶体对称等价结果，区分正常旋转与反演；CIF 对称信息不足时说明等价搜索不完整。
- [ ] 将单图束向符号、镜像和近轴投影的不确定性显示为候选解及假设。提供二维投影夹角，即使不足以唯一确定三维关系也可核对已有测量。
- [ ] 通过一致旋转、已知旋转角、非正交晶胞、对称等价、镜像和退化输入测试；改变任一组标定后关系失效并可重算。
- [ ] 支持复制关系文本、CSV 导出和项目保存；关系标注如加入图中，也可拖动并使用现有出图样式。

**验收：** 合成真值下能恢复约定内的相对旋转或等价解集合；实际两相图输出每条关系的证据和偏差。不能把两个晶带轴的简单叉乘或二维角度直接当作已确定的完整三维取向关系。

## 9. P4：经校准的倾转辅助

**文件：** 新建 `saedindex/tilt.py`、`saedindex/tilt_panel.py`、`tests/test_tilt.py`；修改项目保存及入口界面。

- [ ] 记录当前 alpha/beta 角、样品杆轴序与正方向、图像中倾转轴方向、相机旋转/镜像关系和机械角度范围。
- [ ] 复用 P3 的取向表示，用 `scipy.spatial.transform.Rotation` 进行正向旋转，用带边界的 `scipy.optimize.least_squares` 求解目标轴。
- [ ] 对多个解按角度变化量排序；显示目标绝对角、角度增量、剩余角误差和是否超限。
- [ ] 缺少仪器坐标校准时仅提供晶体几何旋转量，不输出可直接操作仪器的确定倾转值。
- [ ] 用已知正向旋转生成数据，检查求解与回算一致；覆盖零移动、不可达目标、正负号反转和双解。
- [ ] 用同一区域至少两个已知倾角下的实测图验证仪器参数与误差，再开放“已校准”状态。第一版只计算与显示，不连接仪器控制。

**验收：** 合成回算残差达标、限位判断正确；实测路径有明确坐标校准和误差记录。仅输入当前 alpha/beta 不算完成校准。

## 10. P5：后续独立子项目

- **衍射环：** 中心拟合、束挡/文字排除、径向积分、背景处理、峰拟合、CIF 环位置对照。以真实多晶环数据建立单独计划；不默认自动恢复椭圆畸变。
- **CBED/纳米束圆盘：** 圆盘边缘与中心检测、盘大小控制、重叠盘处理。先建立几何索引，不在同一阶段承诺厚度、结构因子或动力学强度反演。
- **晶体可视化：** 晶胞、晶面及倒易点联动，作为索引核对辅助。等几何索引稳定后再选择渲染实现。

## 11. 每阶段验证与发布

以下命令在 `D:\科研论文\SAEDIndex` 执行；本轮不执行。

先运行对应阶段的聚焦测试，例如 P1：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:QT_QPA_PLATFORM = 'offscreen'
.\venv\Scripts\python.exe -m unittest discover -s tests -p 'test_grouping.py' -v
```

预期：新增行为先由明确失败病例约束，实现后全部通过。完成该阶段后执行完整测试：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:QT_QPA_PLATFORM = 'offscreen'
.\venv\Scripts\python.exe -m unittest discover -s tests -q
```

预期：全部通过，跳过的真实数据测试逐条说明，不把跳过计为验证通过。涉及 UI 的阶段还需 Windows 原生验证：

```powershell
$env:PYTHONPATH = 'D:\科研论文\SAEDIndex'
$env:QT_QPA_PLATFORM = 'windows'
.\venv\Scripts\python.exe tests\verify_revision.py
.\venv\Scripts\python.exe tests\verify_native.py
```

检查正常/紧凑窗口、组操作、标尺待应用保护、撤销重做、标记显隐、标注拖动、CIF 叠加与论文导出。P2 增加 `tests\verify_fft.py` 原生交互检查。

通过后更新 README、阶段记录和验证截图，使用现有脚本重新打包：

```powershell
powershell -ExecutionPolicy Bypass -File .\Build-SAEDIndex.ps1
```

以打包 exe 进行导入图片/CIF、保存重开、调整标注和导出测试；至少检查一个没有开发目录和默认 CIF 路径的环境。只有实际完成后才报告独立环境验证通过。保留上一版包以便回退，不关闭用户正在使用的旧窗口。

## 12. 本计划完成检查

- [x] 查阅现有代码、测试结构、项目格式和打包产物。
- [x] 用官方功能页及论文摘要核实参考方向，纠正未经核实的模块名称；注明全文未读与未进行实机对测。
- [x] 确定可靠性、FFT、取向关系、倾转辅助的依赖与交付顺序。
- [x] 为各阶段列出文件范围、输入条件和验收要求。
- [x] 将科学定标、撤销、人工核对及论文出图纳入回归要求。
- [x] 用户已授权并开始 P0/P1 实施；软件验证记录见 `verification/reliability-validation.md`。
