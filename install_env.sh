uv venv --python 3.11
source setup_env.sh
uv pip install --upgrade pip
uv pip install "isaacsim[all,ros2,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128