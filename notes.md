# about installed FMs
they are located in ~/.cache/huggingface so better empty them when necessary
# launch the script
accelerate launch --cpu ./main_abei_training.py

#necessary
pip install torchvision diffusers accelerate numpy matplotlib datasets
#necessary on aws
eu west-1: nvidia instance not amd
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
sudo dnf clean all
sudo dnf install nvidia-gds
#post installation actions