from pathlib import Path

from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import BRICS, FragmentCatalog, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold


# 教程第一个配体：1,10-菲咯啉-2,9-二羧酸
SMILES = "O=C(O)c1ccc2ccc3ccc(C(=O)O)nc3c2n1"
OUTPUT_DIR = Path.home() / "rdkit_tutorial_results" / "task3_molecule1"
REPORT = []

# 按名称对应教程中的“不含连接原子”的官能团 SMARTS。
FG_WITHOUT_CA = {
    "-NC(=O)CH3": "[N;D2]-[C;D3](=O)-[C;D1;H3]",
    "-C(=O)O": "C(=O)[O;D1]",
    "-C(=O)OMe": "C(=O)[O;D2]-[C;D1;H3]",
    "-C(=O)H": "C(=O)-[H]",
    "-C(=O)N": "C(=O)-[N;D1]",
    "-C(=O)CH3": "C(=O)-[C;D1;H3]",
    "-N=C=O": "[N;D2]=[C;D2]=[O;D1]",
    "-N=C=S": "[N;D2]=[C;D2]=[S;D1]",
    "-NO2": "[N;D3](=[O;D1])[O;D1]",
    "-N=O": "[N;R0]=[O;D1]",
    "=N-O": "[N;R0]-[O;D1]",
    "=NCH3": "[N;R0]-[C;D1;H3]",
    "-N=CH2": "[N;R0]=[C;D1;H2]",
    "-N=NCH3": "[N;D2]=[N;D2]-[C;D1;H3]",
    "-N=N": "[N;D2]=[N;D1]",
    "-N#N": "[N;D2]#[N;D1]",
    "-C#N": "[C;D2]#[N;D1]",
    "-SO2NH2": "[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
    "-NHSO2CH3": "[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]",
    "-SO3H": "[S;D4](=O)(=O)-[O;D1]",
    "-SO3CH3": "[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]",
    "-SO2CH3": "[S;D4](=O)(=O)-[C;D1;H3]",
    "-SO2Cl": "[S;D4](=O)(=O)-[Cl]",
    "-SOCH3": "[S;D3](=O)-[C;D1]",
    "-SCH3": "[S;D2]-[C;D1;H3]",
    "-S": "[S;D1]",
    "=S": "[S;D1]",
    "-X": "[#9,#17,#35,#53]",
    "-tBu": "[C;D4]([C;D1])([C;D1])-[C;D1]",
    "-CF3": "[C;D4](F)(F)F",
    "-C#CH": "[C;D2]#[C;D1;H]",
    "-cPropyl": "[C;D3]1-[C;D2]-[C;D2]1",
    "-OEt": "[O;D2]-[C;D2]-[C;D1;H3]",
    "-OMe": "[O;D2]-[C;D1;H3]",
    "-O": "[O;D1]",
    "=O": "[O;D1]",
    "-N": "[N;D1]",
    "=N": "[N;D1]",
    "#N": "[N;D1]",
}


def log(*items):
    line = " ".join(str(item) for item in items)
    print(line)
    REPORT.append(line)


def original_indices(m):
    """读取原始原子编号，排除断键时新增的 *。"""
    return sorted(
        atom.GetIntProp("original_idx")
        for atom in m.GetAtoms()
        if atom.HasProp("original_idx")
    )


def boundary_bonds(m, selected_atoms):
    """寻找所选部分与其余部分之间的键。"""
    selected = set(selected_atoms)
    pairs = []

    for bond in m.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if (a in selected) != (b in selected):
            pairs.append((a, b))

    return pairs


def find_fragment_atom_indices(m, bond_pairs):
    """参考教程示例：沿邻居搜索，但不跨越指定断键。"""
    cut_edges = {frozenset(pair) for pair in bond_pairs}
    visited = set()
    fragments = []

    for start in range(m.GetNumAtoms()):
        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        group = []

        while stack:
            current = stack.pop()
            group.append(current)

            for neighbor in m.GetAtomWithIdx(current).GetNeighbors():
                other = neighbor.GetIdx()
                edge = frozenset((current, other))

                if other in visited or edge in cut_edges:
                    continue

                visited.add(other)
                stack.append(other)

        fragments.append(sorted(group))

    collected = sorted(i for group in fragments for i in group)
    assert collected == list(range(m.GetNumAtoms()))
    return fragments


def cut_into_parts(m, bond_pairs):
    """断开指定键，在断口两侧保留普通 * 标记。"""
    bond_ids = []

    for a, b in bond_pairs:
        bond = m.GetBondBetweenAtoms(a, b)
        if bond is None:
            raise ValueError(f"原子 {a} 和 {b} 之间没有键")
        bond_ids.append(bond.GetIdx())

    bond_ids = sorted(set(bond_ids))

    if bond_ids:
        broken = Chem.FragmentOnBonds(
            m,
            bond_ids,
            addDummies=True,
            dummyLabels=[(0, 0)] * len(bond_ids),
        )
    else:
        broken = Chem.Mol(m)

    return list(
        Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
    )


def generate_smiles_by_removing_masks(m, smask):
    """返回：其余部分 SMILES、选出部分 SMILES；均保留断口。"""
    selected = set()
    for mask in smask:
        selected.update(mask)

    if not selected <= set(range(m.GetNumAtoms())):
        raise ValueError("smask 中存在无效的原子编号")

    mol_copy = Chem.Mol(m)
    for atom in mol_copy.GetAtoms():
        atom.SetIntProp("original_idx", atom.GetIdx())

    pairs = boundary_bonds(mol_copy, selected)
    parts = cut_into_parts(mol_copy, pairs)
    remaining_smiles = []
    selected_smiles = []

    for part in parts:
        indices = set(original_indices(part))
        smiles = Chem.MolToSmiles(part)

        if indices <= selected:
            selected_smiles.append(smiles)
        elif indices.isdisjoint(selected):
            remaining_smiles.append(smiles)
        else:
            raise RuntimeError("边界键未完全断开")

    return ".".join(remaining_smiles), ".".join(selected_smiles)


def report_parts(parts):
    for number, part in enumerate(parts, start=1):
        log(f"  片段 {number}，原始编号：{original_indices(part)}")
        log("  SMILES：", Chem.MolToSmiles(part))


def report_mask(m, selected_atoms):
    """每次从完整原分子独立选取，不接着上次结果继续删除。"""
    selected = sorted(set(selected_atoms))
    remaining = sorted(set(range(m.GetNumAtoms())) - set(selected))
    kept_smiles, selected_smiles = generate_smiles_by_removing_masks(
        m, [selected]
    )

    log("  选出的原子编号：", selected)
    log("  保留的原子编号：", remaining)
    log("  选出部分（保留 *）：", selected_smiles)
    log("  其余部分（保留 *）：", kept_smiles)


def identify_functional_groups(m):
    """通过子集比较，排除官能团匹配中的连接原子。"""
    filename = str(Path(RDConfig.RDDataDir) / "FunctionalGroups.txt")
    params = FragmentCatalog.FragCatParams(1, 6, filename)
    records = []
    seen = set()

    for i in range(params.GetNumFuncGroups()):
        full_query = params.GetFuncGroup(i)
        name = full_query.GetProp("_Name")

        if name not in FG_WITHOUT_CA:
            raise ValueError(f"未知官能团模板：{name}，请核对数据版本")

        core_query = Chem.MolFromSmarts(FG_WITHOUT_CA[name])
        if core_query is None:
            raise ValueError(f"SMARTS 读取失败：{name}")

        full_matches = m.GetSubstructMatches(full_query)
        core_matches = m.GetSubstructMatches(core_query)

        for full_match in full_matches:
            for core_match in core_matches:
                if not set(core_match) <= set(full_match):
                    continue

                core = tuple(sorted(core_match))
                key = (name, core)
                if key in seen:
                    continue

                seen.add(key)
                records.append({
                    "name": name,
                    "full": sorted(full_match),
                    "core": list(core),
                    "connection": sorted(set(full_match) - set(core_match)),
                })

    return records


def save_structure(m, filename, highlighted_bonds=()):
    """画出原分子编号，将指定断键位置标红。"""
    drawer = rdMolDraw2D.MolDraw2DSVG(1000, 700)
    drawer.drawOptions().addAtomIndices = True
    bond_ids = [
        m.GetBondBetweenAtoms(a, b).GetIdx()
        for a, b in highlighted_bonds
    ]

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        Chem.Mol(m),
        highlightBonds=bond_ids,
        highlightBondColors={i: (0.9, 0.2, 0.2) for i in bond_ids},
    )
    drawer.FinishDrawing()
    (OUTPUT_DIR / filename).write_text(
        drawer.GetDrawingText(), encoding="utf-8"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.clear()
    log("任务 3：1,10-菲咯啉-2,9-二羧酸")
    log("RDKit 版本：", rdBase.rdkitVersion)
    log("编号始终对应原分子，从 0 开始，不按输出 SMILES 的位置判断。")

    # 流程 1：读取分子
    log("\n流程 1：读取原分子")
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        raise ValueError("分子读取失败，请检查 SMILES")

    for atom in mol.GetAtoms():
        atom.SetIntProp("original_idx", atom.GetIdx())

    formula = rdMolDescriptors.CalcMolFormula(mol)
    log("输入 SMILES：", SMILES)
    log("分子式：", formula)
    log("非氢原子数：", mol.GetNumAtoms())
    assert formula == "C14H8N2O4", "分子式与第一个配体不符"
    assert mol.GetNumAtoms() == 20, "非氢原子数应为 20"

    for atom in mol.GetAtoms():
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        log(
            f"原子 {atom.GetIdx():2d}：{atom.GetSymbol():2s}，"
            f"邻居编号：{neighbors}"
        )

    save_structure(mol, "01_original_atom_indices.svg")

    # 流程 2：获取去重后的 BRICS 片段种类
    log("\n流程 2：BRICS 分解")
    unique_fragments = sorted(
        BRICS.BRICSDecompose(
            mol,
            minFragmentSize=1,
            keepNonLeafNodes=False,
            singlePass=False,
        )
    )
    log("去重后的片段种类数：", len(unique_fragments))
    for number, fragment in enumerate(unique_fragments, start=1):
        log(f"种类 {number}：{fragment}")

    # 流程 3：找键并实际断开，保留重复出现的片段
    log("\n流程 3：查找并断开 BRICS 键")
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    brics_pairs = [pair for pair, types in brics_bonds]

    for (a, b), (type_a, type_b) in brics_bonds:
        log(f"断开 {a}—{b}，两端 BRICS 类型：{type_a}、{type_b}")

    broken = BRICS.BreakBRICSBonds(mol)
    brics_parts = list(Chem.GetMolFrags(broken, asMols=True))
    log("实际片段块数（不去重）：", len(brics_parts))
    log("[6*] 等标记中的数字是 BRICS 类型，不是原子编号。")
    report_parts(brics_parts)
    save_structure(mol, "03_BRICS_cut_positions.svg", brics_pairs)

    # 流程 4：自己搜索片段编号，再调用掩码拆分函数
    log("\n流程 4：根据片段编号进行掩码拆分")
    masks = find_fragment_atom_indices(mol, brics_pairs)
    actual_masks = [original_indices(part) for part in brics_parts]
    assert sorted(masks) == sorted(actual_masks), "片段编号核对失败"

    log("各片段的原始编号：", masks)
    log("下面每次都从完整原分子独立选取一块。")
    for number, mask in enumerate(masks, start=1):
        log(f"\n第 {number} 次独立选取")
        report_mask(mol, mask)

    # 流程 5：提取骨架，并找出骨架与支链的连接键
    log("\n流程 5：Murcko 骨架及其连接键")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    match = mol.GetSubstructMatch(scaffold)
    if not match:
        raise ValueError("没有在原分子中找到对应骨架")

    scaffold_atoms = sorted(match)
    scaffold_pairs = boundary_bonds(mol, scaffold_atoms)
    log("骨架 SMILES：", Chem.MolToSmiles(scaffold))
    log("骨架在原分子中的编号：", scaffold_atoms)
    log("骨架与支链之间的连接键：", scaffold_pairs)

    # 流程 6：沿连接键拆分骨架和支链
    log("\n流程 6：拆分骨架和支链")
    murcko_masks = find_fragment_atom_indices(mol, scaffold_pairs)
    murcko_parts = cut_into_parts(mol, scaffold_pairs)
    report_parts(murcko_parts)

    sidechain_masks = [
        mask for mask in murcko_masks
        if set(mask).isdisjoint(scaffold_atoms)
    ]
    backbone_smiles, sidechain_smiles = (
        generate_smiles_by_removing_masks(mol, sidechain_masks)
    )
    log("全部支链的原始编号：", sidechain_masks)
    log("留下的骨架（保留 *）：", backbone_smiles)
    log("选出的支链（保留 *）：", sidechain_smiles)
    save_structure(mol, "06_Murcko_cut_positions.svg", scaffold_pairs)

    # 流程 7：识别官能团，排除连接原子，再进行拆分
    log("\n流程 7：官能团识别与拆分")
    records = identify_functional_groups(mol)
    log("官能团匹配记录数：", len(records))
    log("小模板可能匹配羧基内部的 O，因此匹配记录允许重叠。")

    for number, item in enumerate(records, start=1):
        log(f"\n匹配 {number}：{item['name']}")
        log("  含连接原子的编号：", item["full"])
        log("  排除的连接原子：", item["connection"])
        report_mask(mol, item["core"])

    # 汇总时，去掉完全包含在较大匹配内部的小匹配。
    cores = {frozenset(item["core"]) for item in records}
    main_group_masks = sorted(
        sorted(core)
        for core in cores
        if not any(core < other for other in cores)
    )
    log("\n排除子集重复后的主要基团：", main_group_masks)
    assert main_group_masks == [[0, 1, 2], [13, 14, 15]], (
        "主要基团应为两个羧基，请核对结构和模板匹配"
    )

    remaining_smiles, group_smiles = (
        generate_smiles_by_removing_masks(mol, main_group_masks)
    )
    log("同时选出的两个羧基：", group_smiles)
    log("留下的主体（保留 *）：", remaining_smiles)

    log("\n完成检查")
    log("BRICS 片段种类数：", len(unique_fragments))
    log("BRICS 实际片段块数：", len(brics_parts))
    log("Murcko 支链数量：", len(sidechain_masks))
    log("主要官能团数量：", len(main_group_masks))
    log("普通 * 表示断口，点号 . 表示互不连接。")
    log("原分子 mol 未被修改；以上是结构拆分，不是反应预测。")
    log("输出文件夹：", OUTPUT_DIR)

    report_path = OUTPUT_DIR / "results.txt"
    log("文字报告：", report_path)
    report_path.write_text("\n".join(REPORT), encoding="utf-8")


if __name__ == "__main__":
    main()