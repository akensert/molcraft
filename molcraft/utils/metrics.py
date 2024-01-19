import numpy as np

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer

# TODO: Create a Callback which computes these metrics?

def validity(smiles: list[str]) -> float:
    n_total = len(smiles)
    n_valid = 0
    for s in smiles:
        if Chem.MolFromSmiles(s) is not None:
            n_valid += 1 
    return n_valid / n_total

def diversity(smiles: list[str]) -> float:
    n_total = len(smiles)
    n_unique = len(set(smiles))
    return n_unique / n_total

def novelty(smiles: list[str], smiles_database: list[str]) -> float:
    n_total = len(smiles)
    n_novel = len(set(smiles) - set(smiles_database))
    return n_novel / n_total

def qed(smiles: list[str]) -> tuple[float, float]:
    qed_scores = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            qed_score = QED.qed(mol)
            qed_scores.append(qed_score)

    if not len(qed_scores):
        return None 
        
    qed_mean = np.mean(qed_scores)
    qed_stderr = np.std(qed_scores) / np.sqrt(len(qed_scores))
    return qed_mean, qed_stderr

def sas(smiles: list[str]) -> tuple[float, float]:
    sas_scores = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            sas_score = sascorer.calculateScore(mol)
            sas_scores.append(sas_score)
    
    if not len(sas_scores):
        return None 
        
    sas_mean = np.mean(sas_scores)
    sas_stderr = np.std(sas_scores) / np.sqrt(len(sas_scores))
    return sas_mean, sas_stderr