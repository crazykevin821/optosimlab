# LaTeX 公式版 PDF 构建说明

当前正式手册由 latex_formula_manual.tex 生成。请不要再使用旧的 ReportLab 脚本生成 PDF；该脚本不能可靠排版数学上下标、希腊字母和分式。

在 PowerShell 中执行：

~~~powershell
# 先在 PowerShell 中进入仓库根目录
Set-Location .\docs\source
xelatex -interaction=nonstopmode -halt-on-error latex_formula_manual.tex
xelatex -interaction=nonstopmode -halt-on-error latex_formula_manual.tex
Copy-Item .\latex_formula_manual.pdf ..\OptoSimLab_使用说明.pdf -Force
New-Item -ItemType Directory -Force ..\..\output\pdf
Copy-Item .\latex_formula_manual.pdf ..\..\output\pdf\OptoSimLab_使用说明_v0.10.2.pdf -Force
~~~

第二次编译用于刷新目录和交叉引用。输出后，用 Poppler 将 PDF 渲染为 PNG，检查中文、公式、代码块、表格、页眉和页码。正式发布同时保留 `docs` 稳定文件名和 `output/pdf` 带版本文件名。
