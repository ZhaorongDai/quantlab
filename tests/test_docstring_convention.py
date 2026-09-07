"""文档字符串约定闸门：用 AST 扫描全部非测试模块，强制 Google 风格中文文档字符串。

这个模块有两个入口。作为 pytest 测试运行时，它是防止约定回退的永久护栏；
直接用 ``python tests/test_docstring_convention.py`` 运行时，它按文件分组打印
完整的违规报告，最后一行输出 ``REMAINING: N``——转换期间唯一需要人读的数字。
两个入口共用同一套采集逻辑，因此报告和测试不可能各说各话。

闸门只能证明**结构**：文档字符串存在、摘要是中文、参数被列全并带类型、有返回
值就写了 ``Returns:``。它证明不了文档字符串是**对的**或**有用的**——类型抄错、
写成"设置 x 的值"这种复述代码的摘要、或者 ``x (int): 参数`` 这种敷衍描述，都能
通过。所以闸门变绿只说明文档字符串齐了，不说明它们写得好。

约定本身见 CLAUDE.md。一条容易踩的坑：仓库里有多个测试直接读模块**源文本**，
且只过滤 ``#`` 开头的注释行，因此文档字符串的每一行都会被当成活代码扫描——
文档字符串里不能出现赋值语句、导入路径或调用表达式的原文。
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: 摘要的中文字符下限。仓库里最短的真实中文摘要是 5 个汉字
#: （utils/nautilus.py::generate_bar_type_str -> "生成bar type字符串"），
#: 所以门槛设在 4：低于任何一条真实摘要，高于任何一个可信的占位符。
_MIN_SUMMARY_CJK = 4

_CJK_RE = re.compile(r"[一-鿿]")

_SECTION_HEADER_RE = re.compile(r"^[ \t]*(Args|Returns|Raises|Yields):[ \t]*$")

#: 匹配 ``Args:`` 段里的一条参数条目：``name (type): 说明``。
#: 星号是条目的一部分，因为 ``*args`` / ``**kwargs`` 也是参数，也要立条目。
_ARG_ENTRY_RE = re.compile(r"^[ \t]*(\*{0,2}[A-Za-z_]\w*)[ \t]*(\([^)]*\))?[ \t]*:")

_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n")

#: 尚未转换的模块路径（相对仓库根，POSIX 风格）。这是一个只减不增的棘轮：
#: 每个批次转换完自己的文件后必须把它们从这里删掉，而
#: ``test_the_conversion_allowlist_only_shrinks`` 保证一个已经转换完的文件
#: 无法继续留在这里。全部转换完成后，这个白名单连同棘轮测试一起被删除。
_CONVERSION_ALLOWLIST: frozenset[str] = frozenset(
    {
        "acquisition/alpaca.py",
        "acquisition/tiingo.py",
        "acquisition/universe.py",
        "base/acquisition.py",
        "base/chunking.py",
        "base/constituent.py",
        "base/data.py",
        "base/pageledger.py",
        "config/__init__.py",
        "dataset/backend.py",
        "dataset/cleaning.py",
        "dataset/constituent.py",
        "dataset/masking.py",
        "dataset/spot.py",
        "dataset/stock.py",
        "dl_model/mlp.py",
        "dl_model/rnn.py",
        "dl_model/rnn_classification.py",
        "enums/constant.py",
        "enums/data.py",
        "factor/alpha101.py",
        "factor/alpha158.py",
        "factor/momentum.py",
        "get_binance_instruments.py",
        "ingest_alpaca.py",
        "ingest_binance_spot.py",
        "ingest_tiingo.py",
        "ingest_us_equity.py",
        "label/spot.py",
        "main.py",
        "ml_model/backend.py",
        "my_ops/preprocess.py",
        "read_mock_data_sink.py",
        "scripts/download_stock_data_from_tiingo.py",
        "utils/asdict.py",
        "utils/binance.py",
        "utils/cli.py",
        "utils/file.py",
        "utils/module.py",
        "utils/nautilus.py",
        "utils/timer.py",
        "vecbt/bt.py",
    }
)


def discover_modules() -> list[Path]:
    """收集所有受本约定管辖的源码模块，即除测试外的全部 Python 文件。

    刻意不调用 git：闸门必须在裸检出、打包分发或任何没有 git 的环境里给出同样
    的结果，否则"哪些文件受管"就成了环境的函数而不是仓库的事实。

    Returns:
        list[Path]: 按路径排序的模块绝对路径列表。
    """
    modules: list[Path] = []
    for path in sorted(_REPO_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if rel.parts[0] == "tests":
            continue
        if path.name.startswith("test_") or path.name == "conftest.py":
            continue
        modules.append(path)
    return modules


def _iter_units(tree: ast.Module) -> Iterator[ast.AST]:
    """遍历一个模块里所有受管的类与函数定义。

    函数体内直接定义的闭包被排除：闭包是其外层函数的实现细节，而外层函数本身
    已经受管，要求闭包单独写一份 Google 段落只会制造噪音。

    Args:
        tree (ast.Module): 已解析的模块语法树。

    Yields:
        ast.AST: 一个 ClassDef、FunctionDef 或 AsyncFunctionDef 节点。
    """
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        parent = parents.get(node)
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield node


def _parse_sections(doc: str) -> tuple[list[str], dict[str, list[str]]]:
    """把文档字符串切成 Google 段落，返回段落出现顺序与各段落正文。

    段落头必须独占一行（``Args:``），这样才能和正文里顺口提到的"Returns:"区分
    开。段落在遇到缩进不深于段落头的非空行时结束，因此像 ``Contract:`` 这样的
    自定义段落会正确地终止上一段而不被吞进去。

    Args:
        doc (str): 已经 cleandoc 过的文档字符串全文。

    Returns:
        tuple[list[str], dict[str, list[str]]]: 段落名按出现顺序组成的列表
            （重复出现会重复计入，供查重使用），以及段落名到正文行的映射。
    """
    headers: list[str] = []
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    current_indent = 0
    for line in doc.splitlines():
        match = _SECTION_HEADER_RE.match(line)
        if match:
            current = match.group(1)
            current_indent = len(line) - len(line.lstrip())
            headers.append(current)
            bodies.setdefault(current, [])
            continue
        if current is None:
            continue
        if line.strip() and (len(line) - len(line.lstrip())) <= current_indent:
            current = None
            continue
        bodies[current].append(line)
    return headers, bodies


def _arg_entries(body: list[str]) -> dict[str, bool]:
    """解析 ``Args:`` 段正文，得到每个被记录的参数名以及它有没有带类型。

    只有处在段内最浅缩进层的行才算一条条目；更深的行是上一条条目的续行描述，
    否则一段跨行说明里出现的"冒号短语"会被误认成新参数。

    Args:
        body (list[str]): ``Args:`` 段的正文行。

    Returns:
        dict[str, bool]: 参数名到"是否带括号类型"的映射，参数名已去掉星号。
    """
    entries: dict[str, bool] = {}
    filled = [line for line in body if line.strip()]
    if not filled:
        return entries
    base_indent = min(len(line) - len(line.lstrip()) for line in filled)
    for line in filled:
        if (len(line) - len(line.lstrip())) != base_indent:
            continue
        match = _ARG_ENTRY_RE.match(line)
        if match:
            entries[match.group(1).lstrip("*")] = match.group(2) is not None
    return entries


def _documented_parameters(node: ast.AST) -> list[str]:
    """列出一个函数必须在 ``Args:`` 里交代的参数名。

    ``self`` 与 ``cls`` 被排除（它们是调用协议而不是入参），``*args`` 与
    ``**kwargs`` 则包含在内——它们是真正的参数，调用者需要知道往里传什么。

    Args:
        node (ast.AST): 一个 FunctionDef 或 AsyncFunctionDef 节点。

    Returns:
        list[str]: 需要被记录的参数名，已去掉星号。
    """
    args = node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return [name for name in names if name not in ("self", "cls")]


def _own_scope_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """遍历一个函数自身作用域内的节点，不下潜进嵌套的函数、类或 lambda。

    ``return`` 的归属必须按作用域算：闭包里的 ``return`` 交回的是闭包的值，把它
    算到外层函数头上会逼外层写一段它根本不产出的返回说明。

    Args:
        node (ast.AST): 作用域的根节点。

    Yields:
        ast.AST: 属于该作用域的一个子节点。
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        yield child
        yield from _own_scope_nodes(child)


def _is_generator(node: ast.AST) -> bool:
    """判断一个函数定义里是否出现了 yield，出现即按生成器要求它交代产出。

    这里刻意连嵌套闭包一起看：一个函数内部藏着生成器闭包时，读者同样需要知道
    它产出什么。这条规则只放宽要求（有 ``Yields:`` 或 ``Returns:`` 皆可），不会
    因为多算而逼出一份错误的文档。

    Args:
        node (ast.AST): 一个 FunctionDef 或 AsyncFunctionDef 节点。

    Returns:
        bool: 定义中出现 yield 为 True。
    """
    return any(
        isinstance(child, (ast.Yield, ast.YieldFrom)) for child in ast.walk(node)
    )


def _returns_a_value(node: ast.AST) -> bool:
    """判断函数是否会向调用者交回一个值，从而必须写 ``Returns:``。

    有返回标注时以标注为准：作者写下 ``-> None`` 就是在声明"这里没有值给你"，
    哪怕函数体里有一句用于提前退出的 return，也不该被要求编造一段 ``Returns:``。
    没有标注时才退回到扫描函数体——仓库里最旧的一批代码普遍不标注返回类型，只
    看标注会把它们整批漏掉。

    Args:
        node (ast.AST): 一个 FunctionDef 或 AsyncFunctionDef 节点。

    Returns:
        bool: 有返回值为 True。
    """
    annotation = node.returns
    if annotation is not None:
        is_none = (
            isinstance(annotation, ast.Constant) and annotation.value is None
        ) or (isinstance(annotation, ast.Name) and annotation.id == "None")
        return not is_none
    return any(
        isinstance(child, ast.Return) and child.value is not None
        for child in _own_scope_nodes(node)
    )


def _summary_cjk_count(doc: str) -> int:
    """数出文档字符串首段里的中文字符个数。

    只看首段：中文摘要是给读者的第一句话，下面保留的英文正文里出现中文并不能
    替代它。

    Args:
        doc (str): 已经 cleandoc 过的文档字符串全文。

    Returns:
        int: 首段中的中文字符数量。
    """
    for chunk in _PARAGRAPH_SPLIT_RE.split(doc):
        if chunk.strip():
            return len(_CJK_RE.findall(chunk))
    return 0


def _check_unit(node: ast.AST) -> list[str]:
    """对单个类或函数逐条检查约定，返回它违反的规则名。

    没有文档字符串时只报 ``MISSING``：其余规则都是在描述一份已存在文档的形状，
    对着不存在的文档再列一串缺失段落，只会把同一个问题数很多遍。

    Args:
        node (ast.AST): 一个 ClassDef、FunctionDef 或 AsyncFunctionDef 节点。

    Returns:
        list[str]: 违反的规则名列表，全部通过时为空列表。
    """
    doc = ast.get_docstring(node)
    if not doc:
        return ["MISSING"]

    failures: list[str] = []
    if _summary_cjk_count(doc) < _MIN_SUMMARY_CJK:
        failures.append("SUMMARY-NOT-CHINESE")

    headers, bodies = _parse_sections(doc)
    for name, count in sorted(Counter(headers).items()):
        if count > 1:
            failures.append(f"DUP-{name}")

    if isinstance(node, ast.ClassDef):
        # 类不写 Args:/Returns:——构造参数记在 __init__ 上，而类本身不返回值。
        return failures

    parameters = _documented_parameters(node)
    if parameters:
        if "Args" not in headers:
            failures.append("NO-ARGS")
        else:
            entries = _arg_entries(bodies["Args"])
            if any(name not in entries for name in parameters):
                failures.append("ARGS-MISSING")
            if any(
                name in entries and not entries[name] for name in parameters
            ):
                failures.append("ARGS-UNTYPED")

    if _is_generator(node):
        if "Yields" not in headers and "Returns" not in headers:
            failures.append("NO-YIELDS")
    elif _returns_a_value(node) and "Returns" not in headers:
        failures.append("NO-RETURNS")

    return failures


def collect_failures() -> list[tuple[str, int, str, str]]:
    """扫描全部受管模块，汇总所有违反约定的地方。

    Returns:
        list[tuple[str, int, str, str]]: 每项为 (相对路径, 行号, 单元名, 规则名)，
            按路径与行号排序。
    """
    results: list[tuple[str, int, str, str]] = []
    for path in discover_modules():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _iter_units(tree):
            for rule in _check_unit(node):
                results.append((rel, node.lineno, node.name, rule))
    results.sort(key=lambda item: (item[0], item[1], item[3]))
    return results


def test_every_non_test_unit_carries_a_google_style_chinese_docstring() -> None:
    """守住已经转换完成的模块：任何一个单元退回旧写法都会让这条测试变红。

    白名单里的文件被跳过，因为它们还没轮到转换；除此之外的每个类和函数都必须
    满足约定。转换全部完成后白名单会被删除，这条测试随之变成无条件的。
    """
    failures = [
        item for item in collect_failures() if item[0] not in _CONVERSION_ALLOWLIST
    ]
    if failures:
        shown = "\n".join(
            f"  {path}:{lineno}:{name} — {rule}"
            for path, lineno, name, rule in failures[:40]
        )
        more = "" if len(failures) <= 40 else f"\n  ... 另有 {len(failures) - 40} 条"
        raise AssertionError(
            f"{len(failures)} 处不满足中文 Google 风格文档字符串约定：\n"
            f"{shown}{more}\n"
            "约定见 CLAUDE.md；运行 "
            "`uv run python tests/test_docstring_convention.py` 查看完整报告。"
        )


def test_the_conversion_allowlist_only_shrinks() -> None:
    """保证白名单是只减不增的棘轮，转换完的文件无法继续赖在里面。

    两件事必须同时成立：白名单里的每个路径都指向一个真实存在且被扫描到的文件
    （否则一次重命名就会留下一条永远生效的静默豁免），以及白名单里的每个文件都
    还至少有一处违规（否则一个已经转换完的文件可以永久免检，批次进度也就无从
    校验）。
    """
    discovered = {
        path.relative_to(_REPO_ROOT).as_posix() for path in discover_modules()
    }
    stale = sorted(_CONVERSION_ALLOWLIST - discovered)
    assert not stale, (
        f"白名单里有 {len(stale)} 个路径不再是受管模块，应当删除：{stale}"
    )

    failing_files = {item[0] for item in collect_failures()}
    converted = sorted(_CONVERSION_ALLOWLIST - failing_files)
    assert not converted, (
        f"这 {len(converted)} 个文件已经满足约定，必须从 _CONVERSION_ALLOWLIST "
        f"中移除：{converted}"
    )


def _main() -> int:
    """打印按文件分组的完整违规报告，供转换过程中读取剩余数量。

    Returns:
        int: 进程退出码，恒为 0——这是一份报告，不是断言。
    """
    failures = collect_failures()
    current = None
    for path, lineno, name, rule in failures:
        if path != current:
            current = path
            counted = sum(1 for item in failures if item[0] == path)
            print(f"\n{path}  ({counted})")
        print(f"  {lineno}:{name} — {rule}")
    print(f"\nREMAINING: {len(failures)}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
