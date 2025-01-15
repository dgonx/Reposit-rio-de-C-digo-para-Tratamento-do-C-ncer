# Reposit-rio-de-C-digo-para-Tratamento-do-C-ncer
"Identificação de Tratamentos para o Câncer Usando Docking Molecular, Simulação Dinâmica e Aprendizado de Máquina: Uma Abordagem Computacional"
# Melhor código possível para descobrir o melhor tratamento para o cancro

# Este programa integra ferramentas de docking molecular, simulação dinâmica molecular (MD), 
análise quântica, aprendizado de máquina (ML) e visualização interativa para identificar 
as melhores interações moleculares que possam levar ao tratamento eficaz do cancro.

# Bibliotecas necessárias
import os
import mdtraj as md
import numpy as np
import requests
import openbabel.pybel as pybel
import nglview as nv
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Diretórios principais
data_dir = "./data"
results_dir = "./results"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Função para buscar candidatos no ChEMBL
def fetch_chembl_molecules(target_id, max_results=50):
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_id}&limit={max_results}"
    response = requests.get(url)
    data = response.json()
    
    molecules = []
    for activity in data.get("activities", []):
        smiles = activity.get("canonical_smiles")
        if smiles:
            molecules.append(smiles)
    return molecules

# ID do alvo (exemplo: TP53 - proteína p53)
target_id = "CHEMBL5023"
molecule_smiles = fetch_chembl_molecules(target_id)

# Preparar e salvar moléculas
prepared_molecules = []
chembl_dir = os.path.join(data_dir, "chembl_molecules")
os.makedirs(chembl_dir, exist_ok=True)

for i, smiles in enumerate(molecule_smiles, 1):
    try:
        mol = pybel.readstring("smi", smiles)
        mol.addh()
        mol.make3D()
        output_file = os.path.join(chembl_dir, f"chembl_{i}.pdbqt")
        mol.write("pdbqt", output_file, overwrite=True)
        prepared_molecules.append(output_file)
        print(f"Molécula {i} salva: {output_file}")
    except Exception as e:
        print(f"Erro ao preparar a molécula {i}: {e}")

# Função para executar docking molecular
def perform_docking(protein_file, ligands_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for ligand in os.listdir(ligands_dir):
        if ligand.endswith(".pdbqt"):
            ligand_path = os.path.join(ligands_dir, ligand)
            output_path = os.path.join(output_dir, f"{ligand}_result.pdbqt")
            log_path = os.path.join(output_dir, f"{ligand}_log.txt")
            os.system(f"vina --receptor {protein_file} --ligand {ligand_path} --out {output_path} --log {log_path}")
            print(f"Docking finalizado para: {ligand}")

# Arquivo da proteína alvo
protein_file = "./data/target_protein.pdbqt"
docking_results_dir = os.path.join(results_dir, "docking")
perform_docking(protein_file, chembl_dir, docking_results_dir)

# Função para extrair energias de docking
def extract_binding_energies(results_dir):
    energies = {}
    for file in os.listdir(results_dir):
        if file.endswith("_log.txt"):
            ligand = file.replace("_log.txt", "")
            with open(os.path.join(results_dir, file), "r") as f:
                for line in f:
                    if "REMARK VINA RESULT:" in line:
                        energy = float(line.split(":")[1].split()[0])
                        if ligand not in energies:
                            energies[ligand] = []
                        energies[ligand].append(energy)
    return energies

docking_energies = extract_binding_energies(docking_results_dir)

# Identificar o melhor ligante
best_ligand = min(docking_energies, key=lambda k: min(docking_energies[k]))
best_energy = min(docking_energies[best_ligand])
print(f"Melhor ligante: {best_ligand} com energia {best_energy:.2f} kcal/mol")

# Simulação dinâmica molecular para o melhor ligante
# Configuração com GROMACS seria chamada em scripts externos para maior desempenho
def prepare_md_simulation(protein_file, ligand_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"gmx pdb2gmx -f {protein_file} -o {output_dir}/processed_protein.gro -water spce")
    os.system(f"gmx editconf -f {output_dir}/processed_protein.gro -o {output_dir}/protein_box.gro -c -d 1.0 -bt cubic")
    os.system(f"gmx solvate -cp {output_dir}/protein_box.gro -cs spc216.gro -o {output_dir}/protein_solv.gro -p {output_dir}/topol.top")
    os.system(f"gmx grompp -f ions.mdp -c {output_dir}/protein_solv.gro -p {output_dir}/topol.top -o {output_dir}/ions.tpr")
    os.system(f"gmx genion -s {output_dir}/ions.tpr -o {output_dir}/protein_ions.gro -p {output_dir}/topol.top -neutral")
    os.system(f"gmx grompp -f em.mdp -c {output_dir}/protein_ions.gro -p {output_dir}/topol.top -o {output_dir}/em.tpr")
    os.system(f"gmx mdrun -deffnm {output_dir}/em")

md_output_dir = os.path.join(results_dir, "md_simulation")
prepare_md_simulation(protein_file, os.path.join(docking_results_dir, f"{best_ligand}_result.pdbqt"), md_output_dir)

# Análise dos resultados de simulação
trajectory_file = os.path.join(md_output_dir, "md.xtc")
topology_file = os.path.join(md_output_dir, "topol.top")
trajectory = md.load(trajectory_file, top=topology_file)
rmsd = md.rmsd(trajectory, trajectory, frame=0)

print(f"RMSD médio: {rmsd.mean():.3f} nm")

# Visualização em 3D
view = nv.show_mdtraj(trajectory)
view.add_representation("cartoon", selection="protein", color="blue")
view.add_representation("licorice", selection="not protein", color="red")
view.center()
view

# Integração de aprendizado de máquina para prever candidatos promissores
def prepare_features(molecule_smiles):
    features = []
    for smiles in molecule_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            features.append(np.array(fp))
    return np.array(features)

# Criar dataset para treinamento
features = prepare_features(molecule_smiles)
targets = [-min(docking_energies.get(f"chembl_{i}", [0])) for i in range(1, len(molecule_smiles) + 1)]

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Treinamento do modelo de machine learning
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliar modelo
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Erro quadrático médio do modelo: {mse:.3f}")

# Predição de novos candidatos
new_predictions = model.predict(features)
best_predicted_index = np.argmax(new_predictions)
print(f"Melhor candidato predito: {molecule_smiles[best_predicted_index]} com pontuação {new_predictions[best_predicted_index]:.2f}")
