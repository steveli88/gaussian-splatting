# python folder_psnr.py checkpoints_dual3dgs/color checkpoints_dual3dgs/gt-color
# export PYTHONPATH='/home/lorentz/Project/Code/dual3dgs_contrastive'
# conda env config vars set NERFBASELINES_BACKEND=python

nerfbaselines train --method gaussian-splatting --data datasets/phototourism/brandenburg-gate --output checkpoints_3dgs
nerfbaselines train --method gaussian-splatting --data datasets/phototourism/sacre-coeur --output checkpoints_3dgs
nerfbaselines train --method gaussian-splatting --data datasets/phototourism/trevi-fountain --output checkpoints_3dgs

export NERFBASELINES_REGISTER="dual_3dgs_spec.py"
nerfbaselines train --method dual-3dgs --data datasets/phototourism/brandenburg-gate --output checkpoints_dual3dgs_fullnerfw_merge
nerfbaselines train --method dual-3dgs --data datasets/phototourism/brandenburg-gate --output checkpoints_dual3dgs_fullnerfw_bgate_average
nerfbaselines train --method dual-3dgs --data datasets/phototourism/brandenburg-gate --output dual3dgs_fullnerfw_bgate_contrastive

# clone repo
git clone https://github.com/steveli88/gaussian-splatting.git 
git checkout dual3dgs
git submodule update --init --recursive

# conda environment?
pip install nerfbaselines
Check torch version, cuda version
Compile rasterization and simple knn

# download data
mkdir datasets
cd datasets
nerfbaselines download-dataset external://phototourism/sacre-coeur -o datasets
cd ..

