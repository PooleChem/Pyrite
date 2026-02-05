#%%
import sys
import pyrite
from scipy.optimize import differential_evolution, minimize
from tqdm.notebook import tqdm, trange
import numpy as np
import pandas as pd
from viztracer import VizTracer
# %%
#load data 
ligand = pyrite.Ligand.from_sdf('examples/input_files/factor_x_ligand.sdf')

print('Dihedral angles:', len(ligand.dihedral_angles))
ligand.set_draw_options({'highlight': 'center'})
ligand.svg
# %%
receptor = pyrite.Receptor.from_pdb('examples/input_files/factor_x.pdb')
receptor.viewer
# %%
#Pocket of protein
binding_site = pyrite.bounds.RectangularBounds.from_ligand(ligand, padding=1.0) 
grid_size = 1 #(0,1)
pocket = pyrite.bounds.Pocket.from_receptor(receptor, grid_size=1.4, neighbors=26, gt_distance_to_outside=14, solvent_accessible=False).intersect(binding_site,padding=1.0)
pyrite.Viewer(receptor, pocket).show()


# %%
#Initial placement
init_vars = ligand.place_in(pocket, n_positions=4000, n_conformations=20)
viewer = pyrite.Viewer(receptor)
viewer.add_v(ligand, init_vars[:59], slider=False, options={'colorscheme': 'magentaCarbon'})

viewer.show()
# %%
#First scoring fucntion based on the pocket
overlap_scoring = pyrite.scoring.DistanceToPocket(ligand, pocket) + 0.1 * pyrite.scoring.InternalEnergy(ligand)
# %%
#Filter out the poses and get top 100
k = 100

energies = np.zeros(len(init_vars))

for i, var in tqdm(enumerate(init_vars), total=len(init_vars)):
    energies[i] = overlap_scoring.step(var, ligand)

sorted_indices = np.argsort(energies)
filtered_vars = init_vars[sorted_indices[:k]]
# %%
#Searching 
searching_func = "pocket overlap"
if searching_func == "Vina":
    scoring = (-0.035579 * pyrite.Gaussian(ligand, receptor, offset=0.0, width=0.5, k=400)
                + -0.005156 * pyrite.scoring.Gaussian(ligand, receptor, offset=3.0, width=2.0, k=400)
                + 0.840245 * pyrite.scoring.Repulsion(ligand, receptor, offset=0.0, k=400)
                + -0.035069 * pyrite.scoring.Hydrophobic(ligand, receptor, good=0.5, bad=1.5, k=400)
                + -0.587439 * pyrite.scoring.NonDirHBond(ligand, receptor, good=-0.7, bad=0.0, k=400)
                + 1e-2 * pyrite.scoring.InternalEnergy(ligand)) / (1 + ((0.1 * (1.923 + 1)) * (NumTors(ligand))) / 5)
elif searching_func == "PLP":
    scoring = pyrite.scoring.PlantsPLP(ligand, receptor) + 1e-2 * pyrite.scoring.DistanceToPcket.InternalEnergy(ligand)
else: # use pocket overlap
    scoring = overlap_scoring

print(scoring.get_score())
# %%
n_poses = 4 # @param {"type":"slider","min":1,"max":20,"step":1}

results = []

crowding = pyrite.scoring.Crowding(ligand, offset=4)
func = scoring + 1e1 * crowding

for i in trange(n_poses):
    init_vars_idx = np.random.choice(len(filtered_vars), size=20, replace=True)
    init_vars_run = filtered_vars[init_vars_idx]

    res = differential_evolution(func.step, pocket.get_bounds(ligand), args={ligand}, rng=42, init=init_vars_run, strategy='best1bin', maxiter=1000, polish=False)
    results.append(res)

    crowding.register_pose(res.x)


# %%
rmsd = pyrite.scoring.RMSD(ligand)

rmsds = [rmsd.step(res.x, ligand) for res in results]
score = [scoring.step(res.x, ligand) for res in results]

df = pd.DataFrame(
    {'fun': [res.population_energies[0] for res in results], 'score': score, 'RMSD': rmsds,
     'v': [res.x for res in results]})
df.sort_values('score', inplace=True)
df.reset_index(inplace=True, drop=True)
pyrite.Viewer(receptor, options={'stickresidues': ['HEM']}).add_v(ligand, df['v'].values, slider=True,
                                                           options={'colorscheme': 'magentaCarbon'}).show()
print(df[['score', 'RMSD']])
# %%
refining_func = "PLP" 
if refining_func == "Vina":
    scoring = (-0.035579 *  pyrite.scoring.Gaussian(ligand, receptor, offset=0.0, width=0.5, k=400)
                + -0.005156 *  pyrite.scoring.Gaussian(ligand, receptor, offset=3.0, width=2.0, k=400)
                + 0.840245 *  pyrite.scoring.Repulsion(ligand, receptor, offset=0.0, k=400)
                + -0.035069 *  pyrite.scoring.Hydrophobic(ligand, receptor, good=0.5, bad=1.5, k=400)
                + -0.587439 *  pyrite.scoring.NonDirHBond(ligand, receptor, good=-0.7, bad=0.0, k=400)
                + 1e-2 *  pyrite.scoring.InternalEnergy(ligand)) / (1 + ((0.1 * (1.923 + 1)) * (NumTors(ligand))) / 5)
else:
    scoring =  pyrite.scoring.PlantsPLP(ligand, receptor) + 1e-2 *  pyrite.scoring.InternalEnergy(ligand)

print(scoring.get_score())
# %%
refine_results = []

for i, v in tqdm(enumerate(df['v'].values), total=len(df)):
    r = minimize(scoring.step, v, args=(ligand,), method='L-BFGS-B', options={'eps': 1e-2}, tol=1e-4)
    refine_results.append(r)
# %%
rmsds = [rmsd.step(res.x, ligand) for res in refine_results]

df = pd.DataFrame({'score': [res.fun for res in refine_results], 'RMSD': rmsds, 'v': [res.x for res in refine_results]})
df.sort_values('score', inplace=True)
pyrite.Viewer(receptor, ligand, options={'stickresidues': ['HEM']}).add_v(ligand, [res.x for res in refine_results], slider=True,
                               options={'colorscheme': 'magentaCarbon'}).show()
df[['score', 'RMSD']]
# %%
