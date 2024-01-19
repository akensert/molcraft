from rdkit import Chem
from rdkit.Chem import Draw 


def visualize_smiles(
    smiles: list[str], 
    grid_size: tuple[int, int] = (5, 5), 
    save_path: str = None
) -> None:
    rdkit_mols = []
    for s in smiles:
        rdkit_mol = Chem.MolFromSmiles(s)
        if rdkit_mol is not None:
            rdkit_mols.append(rdkit_mol)

    n = grid_size[0] * grid_size[1]
    img = Draw.MolsToGridImage(
        rdkit_mols[:n], molsPerRow=grid_size[0], returnPNG=False)
    if save_path:
        img.save(save_path)
    else:
        return img