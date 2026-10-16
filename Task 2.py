from rdkit import Chem
from rdkit.Chem import AllChem
smiles = "c1ccccc1"
benzene = Chem.MolFromSmiles(smiles)
print("苯的SMILES：", Chem.MolToSmiles(benzene))
print("补氢前原子数：", benzene.GetNumAtoms())
benzene = Chem.AddHs(benzene)
print("补氢后原子数：", benzene.GetNumAtoms())

embed_result = AllChem.EmbedMolecule(benzene, randomSeed=42)
optimize_result = AllChem.MMFFOptimizeMolecule(benzene)

print("三维坐标生成结果：", embed_result)
print("结构优化结果：", optimize_result)

conf = benzene.GetConformer()

print("原子编号  元素        x          y          z")
for atom in benzene.GetAtoms():
    i = atom.GetIdx()
    pos = conf.GetAtomPosition(i)
    print(i, atom.GetSymbol(), f"{pos.x:.4f}", f"{pos.y:.4f}", f"{pos.z:.4f}")

Chem.MolToXYZFile(benzene, "benzene.xyz")
print("benzene.xyz 已保存")
