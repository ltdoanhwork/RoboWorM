import sys
sys.path.append('.')
from models.dreamgen.cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
config = get_cosmos_predict2_video2world_pipeline(model_size="14B", fastercache=True)
print(f"Fastercache config: {config.net.fastercache_config}")
