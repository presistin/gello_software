import numpy as np
import zarr
from pathlib import Path
from numcodecs import blosc

npz_file = Path("/home/zaq/bc_data/gello/merged_all_episodes.npz")
output_zarr = Path("/home/zaq/bc_data/gello_replay_buffer.zarr")

print("加载 npz...")
data = np.load(npz_file, allow_pickle=True)

obs_list = data['obs']
action = data['action'].astype(np.float32)
episode_ends = data.get('episode_ends', np.array([len(action)], dtype=np.int64))

print("提取观测字段...")
image0_list = [timestep['image0'] for timestep in obs_list]
image1_list = [timestep['image1'] for timestep in obs_list]
agent_pos_list = [timestep['agent_pos'] for timestep in obs_list]

image0 = np.stack(image0_list, axis=0).astype(np.uint8)
image1 = np.stack(image1_list, axis=0).astype(np.uint8)
agent_pos = np.stack(agent_pos_list, axis=0).astype(np.float32)

print("数据信息：")
print(f"  action shape: {action.shape}")
print(f"  image0 shape: {image0.shape}")
print(f"  agent_pos shape: {agent_pos.shape}")
print(f"  episode_ends: {episode_ends}")

print(f"\n创建 zarr: {output_zarr}")
root = zarr.open(str(output_zarr), mode='w')

data_grp = root.create_group('data')

data_grp.create_dataset(
    'action',
    data=action,
    shape=action.shape,                  # ← 必须显式指定
    chunks=(500, action.shape[1]),
    compressor=blosc.Blosc(cname='zstd', clevel=5)
)

data_grp.create_dataset(
    'agent_pos',
    data=agent_pos,
    shape=agent_pos.shape,
    chunks=(500, agent_pos.shape[1]),
    compressor=blosc.Blosc(cname='zstd', clevel=5)
)

data_grp.create_dataset(
    'image0',
    data=image0,
    shape=image0.shape,
    chunks=(1, 480, 640, 3),
    dtype='uint8',
    compressor=blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.BITSHUFFLE)
)

data_grp.create_dataset(
    'image1',
    data=image1,
    shape=image1.shape,
    chunks=(1, 480, 640, 3),
    dtype='uint8',
    compressor=blosc.Blosc(cname='zstd', clevel=5, shuffle=blosc.BITSHUFFLE)
)

meta_grp = root.create_group('meta')
meta_grp.create_dataset('episode_ends', data=episode_ends, shape=episode_ends.shape, dtype='int64')

print(f"\n转换完成！zarr 保存到：{output_zarr}")
print("现在可以用这个 zarr 路径开始训练了。")