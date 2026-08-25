# LaTeX 公式版 PDF 构建说明

当前正式手册由 latex_formula_manual.tex 生成。请不要再使用旧的 ReportLab 脚本生成 PDF；该脚本不能可靠排版数学上下标、希腊字母和分式。

在 PowerShell 中执行：

~~~powershell
cd F:\OptoSimLab\docs\source
F:\texlive\2026\bin\windows\xelatex.exe -interaction=nonstopmode -halt-on-error latex_formula_manual.tex
F:\texlive\2026\bin\windows\xelatex.exe -interaction=nonstopmode -halt-on-error latex_formula_manual.tex
Copy-Item .\latex_formula_manual.pdf ..\OptoSimLab_使用说明.pdf -Force
~~~

第二次编译用于刷新目录和交叉引用。输出后，用 Poppler 将 PDF 渲染为 PNG，检查中文、公式、代码块、表格、页眉和页码。
