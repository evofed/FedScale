source install.sh
git checkout eval-cifar
pip install networkx onnx
pip install -e .
chmod +x register.sh
./register.sh

