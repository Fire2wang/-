from pathlib import Path
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import BRICS, FragmentCatalog, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold

SMILES = "O=C(N(CCCCCCC)CCCCCCC)c1ccc2ccc3ccc(C(=O)N(CCCCCCC)CCCCCCC)nc3c2n1"
OUTPUT_DIR = Path(__file__).resolve().parent / "task3_molecule2_output"
REPORT = []

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


def section(title):
    log("\n" + "=" * 60)
    log(title)
    log("=" * 60)


def original_indices(m):
    return sorted(atom.GetIntProp("original_idx") for atom in m.GetAtoms() if atom.HasProp("original_idx"))


def boundary_bonds(m, selected_atoms):
    selected = set(selected_atoms)
    return [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in m.GetBonds()
        if (bond.GetBeginAtomIdx() in selected) != (bond.GetEndAtomIdx() in selected)
    ]


def find_fragment_atom_indices(m, bond_pairs):
    cut_edges = {frozenset(pair) for pair in bond_pairs}
    visited, fragments = set(), []

    for start in range(m.GetNumAtoms()):
        if start in visited:
            continue

        stack, group = [start], []
        visited.add(start)

        while stack:
            current = stack.pop()
            group.append(current)

            for neighbor in m.GetAtomWithIdx(current).GetNeighbors():
                other = neighbor.GetIdx()
                if other not in visited and frozenset((current, other)) not in cut_edges:
                    visited.add(other)
                    stack.append(other)

        fragments.append(sorted(group))

    assert sorted(i for group in fragments for i in group) == list(range(m.GetNumAtoms()))
    return fragments


def cut_into_parts(m, bond_pairs):
    bond_ids = []

    for a, b in bond_pairs:
        bond = m.GetBondBetweenAtoms(a, b)
        if bond is None:
            raise ValueError(f"原子 {a} 和 {b} 之间没有键")
        bond_ids.append(bond.GetIdx())

    bond_ids = sorted(set(bond_ids))

    if bond_ids:
        broken = Chem.FragmentOnBonds(m, bond_ids, addDummies=True, dummyLabels=[(0, 0)] * len(bond_ids))
    else:
        broken = Chem.Mol(m)

    return list(Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True))


def generate_smiles_by_removing_masks(m, masks):
    selected = {atom_idx for mask in masks for atom_idx in mask}
    if not selected <= set(range(m.GetNumAtoms())):
        raise ValueError("原子编号超出分子范围")

    mol_copy = Chem.Mol(m)
    for atom in mol_copy.GetAtoms():
        atom.SetIntProp("original_idx", atom.GetIdx())

    parts = cut_into_parts(mol_copy, boundary_bonds(mol_copy, selected))
    remaining, chosen = [], []

    for part in parts:
        indices = set(original_indices(part))
        assert indices <= selected or indices.isdisjoint(selected)
        target = chosen if indices <= selected else remaining
        target.append(Chem.MolToSmiles(part))

    return ".".join(remaining), ".".join(chosen)


def report_parts(parts):
    for number, part in enumerate(parts, start=1):
        log(f"  片段 {number}，原始原子编号：{original_indices(part)}")
        log("  SMILES：", Chem.MolToSmiles(part))


def report_mask(m, selected):
    selected = sorted(set(selected))
    remaining = sorted(set(range(m.GetNumAtoms())) - set(selected))
    remaining_smi, selected_smi = generate_smiles_by_removing_masks(m, [selected])
    log("  选出原子编号：", selected)
    log("  其余原子编号：", remaining)
    log("  选出部分：", selected_smi)
    log("  其余部分：", remaining_smi)


def save_structure(m, filename, highlighted_bonds=()):
    drawer = rdMolDraw2D.MolDraw2DSVG(1000, 700)
    drawer.drawOptions().addAtomIndices = True
    bond_ids = [m.GetBondBetweenAtoms(a, b).GetIdx() for a, b in highlighted_bonds]
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, Chem.Mol(m), highlightBonds=bond_ids, highlightBondColors={i: (0.9, 0.2, 0.2) for i in bond_ids})
    drawer.FinishDrawing()
    (OUTPUT_DIR / filename).write_text(drawer.GetDrawingText(), encoding="utf-8")


def identify_functional_groups(m):
    filename = str(Path(RDConfig.RDDataDir) / "FunctionalGroups.txt")
    params = FragmentCatalog.FragCatParams(1, 6, filename)
    records, seen = [], set()

    for i in range(params.GetNumFuncGroups()):
        full_query = params.GetFuncGroup(i)
        name = full_query.GetProp("_Name")

        if name not in FG_WITHOUT_CA:
            raise ValueError(f"模板名称不在字典中：{name}")

        core_query = Chem.MolFromSmarts(FG_WITHOUT_CA[name])
        if core_query is None:
            raise ValueError(f"SMARTS 读取失败：{name}")

        for full_match in m.GetSubstructMatches(full_query):
            for core_match in m.GetSubstructMatches(core_query):
                if not set(core_match) <= set(full_match):
                    continue

                core = tuple(sorted(core_match))
                key = (name, core)

                if key not in seen:
                    seen.add(key)
                    records.append({
                        "name": name,
                        "full": tuple(sorted(full_match)),
                        "core": core,
                        "connection": tuple(sorted(set(full_match) - set(core_match))),
                    })

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.clear()
    log("任务3 / 第二个分子：N,N,N′,N′-四庚基-1,10-菲咯啉-2,9-二甲酰胺")
    log("RDKit 版本：", rdBase.rdkitVersion)

    section("流程1：读取分子并记录原子编号")
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        raise ValueError("SMILES 读取失败")

    for atom in mol.GetAtoms():
        atom.SetIntProp("original_idx", atom.GetIdx())

    formula = rdMolDescriptors.CalcMolFormula(mol)
    log("输入 SMILES：", SMILES)
    log("分子式：", formula)
    log("非氢原子数：", mol.GetNumAtoms())
    assert formula == "C42H66N4O2" and mol.GetNumAtoms() == 48

    for atom in mol.GetAtoms():
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        log(f"  原子 {atom.GetIdx()}，元素 {atom.GetSymbol()}，相邻原子 {neighbors}")

    save_structure(mol, "01_original_atom_indices.svg")

    section("流程2：按 BRICS 规则生成片段种类")
    unique_fragments = sorted(BRICS.BRICSDecompose(mol, minFragmentSize=1))
    log("片段种类数：", len(unique_fragments))
    for fragment in unique_fragments:
        log(" ", fragment)

    section("流程3：查找 BRICS 断键并实际拆分")
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    brics_pairs = [pair for pair, _ in brics_bonds]

    for (a, b), (type_a, type_b) in brics_bonds:
        log(f"  断开 {a}—{b}，BRICS 类型为 {type_a} 和 {type_b}")

    brics_parts = list(Chem.GetMolFrags(BRICS.BreakBRICSBonds(mol), asMols=True))
    log("BRICS 断键数：", len(brics_pairs))
    log("实际片段块数：", len(brics_parts))
    report_parts(brics_parts)
    save_structure(mol, "03_BRICS_cut_positions.svg", brics_pairs)

    section("流程4：列出每块片段的原始原子编号")
    masks = find_fragment_atom_indices(mol, brics_pairs)
    expected_masks = [
        [0, 1],
        [2],
        [3, 4, 5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 44, 45, 46, 47],
        [27, 28],
        [29],
        [30, 31, 32, 33, 34, 35, 36],
        [37, 38, 39, 40, 41, 42, 43],
    ]
    assert masks == expected_masks

    for number, mask in enumerate(masks, start=1):
        log(f"\n第 {number} 块")
        report_mask(mol, mask)

    section("流程5：提取 Murcko 骨架并查找连接键")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    match = mol.GetSubstructMatch(scaffold)
    if not match:
        raise ValueError("没有找到骨架匹配")

    scaffold_atoms = sorted(match)
    scaffold_pairs = boundary_bonds(mol, scaffold_atoms)
    log("骨架 SMILES：", Chem.MolToSmiles(scaffold))
    log("骨架原子编号：", scaffold_atoms)
    log("骨架与侧链的连接键：", scaffold_pairs)

    section("流程6：切开骨架与侧链")
    murcko_masks = find_fragment_atom_indices(mol, scaffold_pairs)
    murcko_parts = cut_into_parts(mol, scaffold_pairs)
    report_parts(murcko_parts)

    sidechains = [mask for mask in murcko_masks if set(mask).isdisjoint(scaffold_atoms)]
    remaining, branches = generate_smiles_by_removing_masks(mol, sidechains)
    log("侧链原子编号：", sidechains)
    log("骨架 SMILES：", remaining)
    log("侧链 SMILES：", branches)
    save_structure(mol, "06_Murcko_cut_positions.svg", scaffold_pairs)

    section("流程7：识别官能团模板并汇总")
    records = identify_functional_groups(mol)

    for number, item in enumerate(records, start=1):
        log(f"\n匹配 {number}：{item['name']}")
        log("  完整匹配原子：", list(item["full"]))
        log("  连接原子：", list(item["connection"]))
        report_mask(mol, list(item["core"]))

    cores = {frozenset(item["core"]) for item in records}
    main_groups = sorted(sorted(core) for core in cores if not any(core < other for other in cores))
    assert main_groups == [[0], [28]]

    remaining, groups = generate_smiles_by_removing_masks(mol, main_groups)
    log("\n模板筛出的核心原子编号：", main_groups)
    log("模板筛出的核心 SMILES：", groups)
    log("剩余分子的 SMILES：", remaining)
    log("注意：本组模板匹配到两个羰基氧，没有匹配到完整的三级酰胺结构。")

    section("完成")
    log("BRICS：", len(unique_fragments), "种片段；", len(brics_parts), "块实际片段。")
    log("Murcko：", len(scaffold_atoms), "个骨架原子；", len(sidechains), "条侧链。")
    log("流程7结果：当前模板匹配到两个羰基氧。")
    log("输出文件夹：", OUTPUT_DIR)
    (OUTPUT_DIR / "results.txt").write_text("\n".join(REPORT), encoding="utf-8")


if __name__ == "__main__":
    main()