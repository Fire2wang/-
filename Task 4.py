from pathlib import Path
from rdkit import Chem

SCAFFOLD_SMILES = "[Db]C1=NC2=C(C=C1)C=CC1=C2N=CC=C1"
SUBSTITUENT_SMILES = "C1CCNC1[Rf]"
OUTPUT_FILE = Path(__file__).resolve().parent / "task4_result.smi"


def find_marker(mol, symbol):
    matches = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == symbol]
    if len(matches) != 1:
        raise ValueError(f"标识符 {symbol} 应恰好出现一次，实际出现 {len(matches)} 次")
    marker = matches[0]
    neighbors = list(marker.GetNeighbors())
    if len(neighbors) != 1:
        raise ValueError(f"标识符 {symbol} 应只连接一个原子")
    return marker.GetIdx(), neighbors[0].GetIdx()


def main():
    scaffold = Chem.MolFromSmiles(SCAFFOLD_SMILES)
    substituent = Chem.MolFromSmiles(SUBSTITUENT_SMILES)

    if scaffold is None or substituent is None:
        raise ValueError("输入 SMILES 无法读取")

    db_idx, scaffold_atom_idx = find_marker(scaffold, "Db")
    rf_idx, substituent_atom_idx = find_marker(substituent, "Rf")

    print("骨架中的 [Db] 编号：", db_idx)
    print("与 [Db] 相邻的骨架原子编号：", scaffold_atom_idx)
    print("取代基中的 [Rf] 编号：", rf_idx)
    print("与 [Rf] 相邻的取代基原子编号：", substituent_atom_idx)

    combined = Chem.CombineMols(scaffold, substituent)
    substituent_offset = scaffold.GetNumAtoms()
    combined_substituent_atom_idx = substituent_atom_idx + substituent_offset
    combined_rf_idx = rf_idx + substituent_offset

    editable = Chem.RWMol(combined)
    editable.AddBond(
        scaffold_atom_idx,
        combined_substituent_atom_idx,
        Chem.BondType.SINGLE
    )

    for atom_idx in sorted([db_idx, combined_rf_idx], reverse=True):
        editable.RemoveAtom(atom_idx)

    product = editable.GetMol()
    Chem.SanitizeMol(product)

    product_smiles = Chem.MolToSmiles(product)
    print("拼接后的 SMILES：")
    print(product_smiles)

    OUTPUT_FILE.write_text(product_smiles + "\n", encoding="utf-8")
    print("结果已保存到：", OUTPUT_FILE)


if __name__ == "__main__":
    main()