pip install opencv-python
pip install tensorboardX
pip install numpy --upgrade
pip install SharedArray
conda install -c open3d-admin open3d
pip install addict
cd lib/pointgroup_ops/
python setup.py install
cd ../pointops2/
python setup.py install
cd ../
pip install easydict
pip install spconv-cu113
pip install einops

export CUDA_PATH=/cluster/public_datasets/nvidia/cuda-11.1
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib:$LD_LIBRARY_PATH

