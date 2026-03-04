import sys
import types

lerobot = types.ModuleType('lerobot')
lerobot.common = types.ModuleType('lerobot.common')
lerobot.common.datasets = types.ModuleType('lerobot.common.datasets')
lrd = types.ModuleType('lerobot.common.datasets.lerobot_dataset')

class _Dummy:
    pass

lrd.LeRobotDataset = _Dummy
lrd.LeRobotDatasetMetadata = _Dummy

lerobot.common.datasets.lerobot_dataset = lrd
sys.modules['lerobot'] = lerobot
sys.modules['lerobot.common'] = lerobot.common
sys.modules['lerobot.common.datasets'] = lerobot.common.datasets
sys.modules['lerobot.common.datasets.lerobot_dataset'] = lrd

exec(open('/workspace/openpi/scripts/serve_policy.py').read())
