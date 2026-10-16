from rdkit import Chem
from rdkit.Chem import BRICS


def generate_smiles_by_removing_masks(m, smask):
    """按原子编号选出片段，同时得到剩余部分的 SMILES。"""
    removed_atoms = set()

    for mask in smask:
        removed_atoms.update(mask)

    all_atoms = set(range(m.GetNumAtoms()))
    remaining_atoms = sorted(all_atoms - removed_atoms)
    removed_atoms = sorted(removed_atoms)

    cleaned_smiles = (
        Chem.MolFragmentToSmiles(m, atomsToUse=remaining_atoms)
        if remaining_atoms else ""
    )

    removed_parts_smiles = (
        Chem.MolFragmentToSmiles(m, atomsToUse=removed_atoms)
        if removed_atoms else ""
    )

    return cleaned_smiles, removed_parts_smiles


def find_fragment_atom_indices(m, brics_bonds):
    """把 BRICS 键视为已断开，找出每个片段包含的原子编号。"""
    cut_edges = {
        frozenset(atom_pair)
        for atom_pair, _ in brics_bonds
    }

    visited = set()
    fragments = []

    for start_idx in range(m.GetNumAtoms()):
        if start_idx in visited:
            continue

        stack = [start_idx]
        visited.add(start_idx)
        fragment = []

        while stack:
            current_idx = stack.pop()
            fragment.append(current_idx)

            current_atom = m.GetAtomWithIdx(current_idx)

            for neighbor in current_atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                edge = frozenset((current_idx, neighbor_idx))

                if edge in cut_edges or neighbor_idx in visited:
                    continue

                visited.add(neighbor_idx)
                stack.append(neighbor_idx)

        fragments.append(sorted(fragment))

    return fragments


# ============================================================
# 流程 1：用 SMILES 读取苯佐卡因
# ============================================================

smiles = "CCOC(=O)c1ccc(N)cc1"
mol = Chem.MolFromSmiles(smiles)

if mol is None:
    raise ValueError("SMILES 读取失败，请检查输入。")

print("流程 1：读取分子")
print("原始 SMILES：", smiles)
print("原子总数：", mol.GetNumAtoms())

for atom in mol.GetAtoms():
    print(f"原子 {atom.GetIdx()}：{atom.GetSymbol()}")


# ============================================================
# 流程 2：用 BRICSDecompose 查看分解得到的片段
# ============================================================

print("\n流程 2：BRICS 分解片段")

brics_smiles = BRICS.BRICSDecompose(mol)

for i, fragment_smiles in enumerate(sorted(brics_smiles), start=1):
    print(f"片段 {i}：{fragment_smiles}")


# ============================================================
# 流程 3：查找 BRICS 键，并实际断开这些键
# ============================================================

print("\n流程 3：查找 BRICS 可拆分键")

brics_bonds = list(BRICS.FindBRICSBonds(mol))

for atom_pair, brics_types in brics_bonds:
    print(f"原子 {atom_pair} 之间可拆；两端类型：{brics_types}")

broken_mol = BRICS.BreakBRICSBonds(mol)

actual_fragments = Chem.GetMolFrags(
    broken_mol,
    asMols=True,
    sanitizeFrags=True
)

print("实际断键后的片段：")

for i, fragment in enumerate(actual_fragments, start=1):
    print(f"片段 {i}：{Chem.MolToSmiles(fragment)}")


# ============================================================
# 流程 4：根据 BRICS 断键，找每个片段的原子编号
#         再调用自己写的函数，分别提取片段和剩余部分
# ============================================================

print("\n流程 4：片段的原子编号与 SMILES")

fragment_masks = find_fragment_atom_indices(mol, brics_bonds)

for i, fragment_atoms in enumerate(fragment_masks, start=1):
    smask = [fragment_atoms]

    remaining_smiles, fragment_smiles = (
        generate_smiles_by_removing_masks(mol, smask)
    )

    print(f"\n片段 {i}")
    print("原子编号：", fragment_atoms)
    print("选出的片段：", fragment_smiles)
    print("其余部分：", remaining_smiles)