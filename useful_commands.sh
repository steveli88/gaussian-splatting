# clone repo
git clone https://github.com/steveli88/gaussian-splatting.git dual3dgs
Cd dual3dgs
git checkout dual3dgs
git submodule update --init --recursive

# conda environment
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

conda create -y -n 3dgs python=3.11
conda activate 3dgs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#python 3.7
#pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install plyfile
pip install tqdm
pip install -e ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
pip install nerfbaselines

# download data
nerfbaselines download-dataset external://phototourism/brandenburg-gate -o data/phototourism/brandenburg-gate
nerfbaselines download-dataset external://phototourism/sacre-coeur -o data/phototourism/sacre-coeur

$env:NERFBASELINES_REGISTER='dual_3dgs_spec.py'
$env:PYTHONPATH='F:\Codes\dual3dgs'
#export NERFBASELINES_REGISTER="dual_3dgs_spec.py"
#export PYTHONPATH='project folder'
nerfbaselines train --method dual-3dgs --data "F:\Codes\wild-gaussians\datasets\phototourism\brandenburg-gate" --output checkpoints_dual3dgs --backend python --eval-all-iters 2000::2000

# python folder_psnr.py checkpoints_dual3dgs/color checkpoints_dual3dgs/gt-color
# export PYTHONPATH='/home/lorentz/Project/Code/dual3dgs_contrastive'
# export NERFBASELINES_REGISTER="dual_3dgs_spec.py"
# conda env config vars set NERFBASELINES_BACKEND=python