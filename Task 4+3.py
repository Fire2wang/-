from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import BRICS, FragmentCatalog, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold


SMILES = "O=C(O)c1ccc2ccc3ccc(C(=O)O)nc3c2n1"
OUTPUT_DIR = Path(__file__).resolve().parent / "task3_molecule1_output"
REPORT = []

# 教程中的官能团核心 SMARTS 模板
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


def original_indices(mol):
    return sorted(
        atom.GetIntProp("original_idx")
        for atom in mol.GetAtoms()
        if atom.HasProp("original_idx")
    )


def boundary_bonds(mol, selected_atoms):
    selected = set(selected_atoms)
    return [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
        if (bond.GetBeginAtomIdx() in selected)
        != (bond.GetEndAtomIdx() in selected)
    ]


def find_fragment_atom_indices(mol, bond_pairs):
    cut_edges = {frozenset(pair) for pair in bond_pairs}
    visited = set()
    fragments = []

    for start in range(mol.GetNumAtoms()):
        if start in visited:
            continue

        stack = [start]
        group = []
        visited.add(start)

        while stack:
            current = stack.pop()
            group.append(current)

            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                other = neighbor.GetIdx()
                if other not in visited and frozenset((current, other)) not in cut_edges:
                    visited.add(other)
                    stack.append(other)

        fragments.append(sorted(group))

    assert sorted(i for group in fragments for i in group) == list(range(mol.GetNumAtoms()))
    return fragments


def cut_into_parts(mol, bond_pairs, label_connections=False):
    """按键切分；label_connections=True 时为每对断口设置相同编号。"""
    bond_ids = []
    unique_pairs = []

    for a, b in bond_pairs:
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            raise ValueError(f"原子 {a} 和 {b} 之间没有键")
        if bond.GetIdx() not in bond_ids:
            bond_ids.append(bond.GetIdx())
            unique_pairs.append((a, b))

    if not bond_ids:
        return [Chem.Mol(mol)]

    labels = [(i, i) for i in range(1, len(bond_ids) + 1)] if label_connections else [(0, 0)] * len(bond_ids)
    broken = Chem.FragmentOnBonds(mol, bond_ids, addDummies=True, dummyLabels=labels)
    return list(Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True))


def generate_smiles_by_removing_masks(mol, masks, encode_connections=False):
    """返回 (剩余部分 SMILES, 选出部分 SMILES)，可选择给断口自动编号。"""
    selected = {idx for mask in masks for idx in mask}
    if not selected <= set(range(mol.GetNumAtoms())):
        raise ValueError("原子编号超出分子范围")

    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetIntProp("original_idx", atom.GetIdx())

    cut_pairs = boundary_bonds(copy, selected)
    parts = cut_into_parts(copy, cut_pairs, label_connections=encode_connections)
    remaining, chosen = [], []

    for part in parts:
        indices = set(original_indices(part))
        assert indices <= selected or indices.isdisjoint(selected)
        target = chosen if indices <= selected else remaining
        target.append(Chem.MolToSmiles(part))

    return ".".join(remaining), ".".join(chosen)


def reassemble_fragments(fragment_smiles):
    """依据断口编号自动配对片段、加回连接键并删除断口标记。"""
    mols = []

    for smiles in fragment_smiles:
        if not smiles:
            continue
        part = Chem.MolFromSmiles(smiles)
        if part is None:
            raise ValueError(f"片段 SMILES 无法读取：{smiles}")
        mols.append(part)

    if not mols:
        raise ValueError("没有可用于拼接的片段")

    combined = mols[0]
    for part in mols[1:]:
        combined = Chem.CombineMols(combined, part)

    ports = defaultdict(list)
    for atom in combined.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() > 0:
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                raise ValueError("断口标记应当只连接一个原子")

            bond = combined.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
            ports[atom.GetIsotope()].append((atom.GetIdx(), neighbors[0].GetIdx(), bond.GetBondType()))

    editable = Chem.RWMol(combined)

    for label, ends in ports.items():
        if len(ends) != 2:
            raise ValueError(f"断口编号 {label} 应出现两次，实际出现 {len(ends)} 次")
        _, atom_a, bond_type_a = ends[0]
        _, atom_b, bond_type_b = ends[1]
        if bond_type_a != bond_type_b:
            raise ValueError(f"断口编号 {label} 两侧的键类型不一致")
        editable.AddBond(atom_a, atom_b, bond_type_a)

    dummy_indices = [
        atom.GetIdx()
        for atom in editable.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() > 0
    ]
    for idx in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(idx)

    result = editable.GetMol()
    Chem.SanitizeMol(result)
    return result


def save_structure(mol, filename, highlighted_bonds=()):
    drawer = rdMolDraw2D.MolDraw2DSVG(1000, 700)
    drawer.drawOptions().addAtomIndices = True
    bond_ids = [mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in highlighted_bonds]
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        Chem.Mol(mol),
        highlightBonds=bond_ids,
        highlightBondColors={idx: (0.9, 0.2, 0.2) for idx in bond_ids},
    )
    drawer.FinishDrawing()
    (OUTPUT_DIR / filename).write_text(drawer.GetDrawingText(), encoding="utf-8")


def identify_functional_groups(mol):
    filename = str(Path(RDConfig.RDDataDir) / "FunctionalGroups.txt")
    params = FragmentCatalog.FragCatParams(1, 6, filename)
    records = []
    seen = set()

    for i in range(params.GetNumFuncGroups()):
        full_query = params.GetFuncGroup(i)
        name = full_query.GetProp("_Name")
        if name not in FG_WITHOUT_CA:
            raise ValueError(f"模板名称不在字典中：{name}")

        core_query = Chem.MolFromSmarts(FG_WITHOUT_CA[name])
        if core_query is None:
            raise ValueError(f"SMARTS 读取失败：{name}")

        for full_match in mol.GetSubstructMatches(full_query):
            for core_match in mol.GetSubstructMatches(core_query):
                if not set(core_match) <= set(full_match):
                    continue

                core = tuple(sorted(core_match))
                key = (name, core)
                if key in seen:
                    continue

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
    log("任务3 / 第一个分子：1,10-菲咯啉-2,9-二羧酸")
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
    assert formula == "C14H8N2O4" and mol.GetNumAtoms() == 20

    for atom in mol.GetAtoms():
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        log(f"  原子 {atom.GetIdx()}，元素 {atom.GetSymbol()}，相邻原子 {neighbors}")

    save_structure(mol, "01_original_atom_indices.svg")

    section("流程2：生成不同的 BRICS 片段种类")
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
    log("实际片段块数：", len(brics_parts))
    for number, part in enumerate(brics_parts, start=1):
        log(f"  片段 {number} SMILES：", Chem.MolToSmiles(part))

    save_structure(mol, "03_BRICS_cut_positions.svg", brics_pairs)

    section("流程4：列出 BRICS 片段的原始原子编号")
    masks = find_fragment_atom_indices(mol, brics_pairs)
    assert masks == [
        [0, 1, 2],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19],
        [13, 14, 15],
    ]

    for number, mask in enumerate(masks, start=1):
        log(f"第 {number} 块原子编号：", mask)

    section("流程5：提取 Murcko 骨架并查找连接键")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_match = mol.GetSubstructMatch(scaffold)
    if not scaffold_match:
        raise ValueError("没有找到骨架匹配")

    scaffold_atoms = sorted(scaffold_match)
    scaffold_pairs = boundary_bonds(mol, scaffold_atoms)
    log("骨架 SMILES：", Chem.MolToSmiles(scaffold))
    log("骨架原子编号：", scaffold_atoms)
    log("骨架与侧链的连接键：", scaffold_pairs)

    section("流程6：切开骨架与侧链，并给断口自动编号")
    murcko_masks = find_fragment_atom_indices(mol, scaffold_pairs)
    sidechains = [mask for mask in murcko_masks if set(mask).isdisjoint(scaffold_atoms)]

    # 这里打开 encode_connections：同一条断键两端会得到相同的编号。
    remaining, branches = generate_smiles_by_removing_masks(
        mol, sidechains, encode_connections=True
    )
    log("带编号的骨架片段：", remaining)
    log("带编号的侧链片段：", branches)
    save_structure(mol, "06_Murcko_cut_positions.svg", scaffold_pairs)

    section("流程7：识别官能团并汇总")
    records = identify_functional_groups(mol)

    for number, item in enumerate(records, start=1):
        log(f"匹配 {number}：{item['name']}")
        log("  完整匹配原子：", list(item["full"]))
        log("  连接原子：", list(item["connection"]))
        log("  核心原子：", list(item["core"]))

    cores = {frozenset(item["core"]) for item in records}
    main_groups = sorted(
        sorted(core)
        for core in cores
        if not any(core < other for other in cores)
    )
    assert main_groups == [[0, 1, 2], [13, 14, 15]]
    log("汇总后的主要官能团原子编号：", main_groups)

    section("流程8：读取带编号的片段并自动拼接")
    fragments = [remaining, branches]
    rebuilt = reassemble_fragments(fragments)

    original_canonical = Chem.MolToSmiles(mol)
    rebuilt_canonical = Chem.MolToSmiles(rebuilt)
    log("自动拼接后的 SMILES：", rebuilt_canonical)
    log("拼接结果与原分子一致：", rebuilt_canonical == original_canonical)

    assert rebuilt_canonical == original_canonical
    (OUTPUT_DIR / "auto_reassembled.smi").write_text(
        rebuilt_canonical + "\n",
        encoding="utf-8",
    )

    section("完成")
    log("BRICS：2 种片段；3 块实际片段。")
    log("Murcko：1 个骨架；2 条羧基侧链。")
    log("官能团汇总：2 个羧基。")
    log("自动拼接：按断口编号配对，并验证还原结果与原分子一致。")
    log("输出文件夹：", OUTPUT_DIR)
    (OUTPUT_DIR / "results.txt").write_text("\n".join(REPORT), encoding="utf-8")


if __name__ == "__main__":
    main()