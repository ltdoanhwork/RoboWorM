curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync --extra=cu128 --python 3.10
source .venv/bin/activate
uv pip install torch
uv pip install openai tyro numpydantic albumentations tianshou git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
uv pip install pyarrow fastparquet piq lightning moviepy==2.2.1
uv pip install -U 'jsonargparse[signatures]>=4.27.7'
uv pip install torchcodec==0.5 --index-url=https://download.pytorch.org/whl/cu128