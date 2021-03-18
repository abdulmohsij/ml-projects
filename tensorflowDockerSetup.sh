# Install docker and docker gpu support as per: https://www.tensorflow.org/install/docker
# Then run this script as:
# source tensorflowDockerSetup.sh

# To start a docker session in your current directory.
sudo docker run --gpus all --rm -v `pwd`:/my-working-directory -it tensorflow/tensorflow:2.1.0-gpu-py3-jupyter
cd my-working-directory

pip install -r requirements.txt
