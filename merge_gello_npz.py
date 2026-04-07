import numpy as np
from pathlib import Path
import os

# 修改为你的实际目录
base_dir = Path("/home/zaq/bc_data/gello")

# 找到所有 eposide.npz 文件（递归搜索）
npz_files = sorted(base_dir.rglob("episode.npz"))

if not npz_files:
    print("没有找到任何 episode.npz 文件！")
    exit(1)

print(f"找到 {len(npz_files)} 个 npz 文件：")
for f in npz_files:
    print(f"  {f}")

all_obs = []
all_action = []
episode_ends = []
cum_steps = 0

for npz_path in npz_files:
    print(f"\n加载 {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    obs = data['obs']
    action = data['action']
    
    num_steps = len(action)
    print(f"  步数: {num_steps}")
    
    all_obs.extend(obs)                  # obs 是 list of dict，不用 np.concatenate
    all_action.append(action)
    
    cum_steps += num_steps
    episode_ends.append(cum_steps)

# 拼接 action
action_all = np.concatenate(all_action, axis=0)

# obs 保持为 object array
obs_all = np.array(all_obs, dtype=object)

episode_ends = np.array(episode_ends, dtype=np.int64)

# 保存合并后的文件
output_path = base_dir / "merged_all_episodes.npz"
np.savez(
    output_path,
    obs=obs_all,
    action=action_all,
    episode_ends=episode_ends
)

print("\n合并完成！")
print(f"总 episode 数: {len(episode_ends)}")
print(f"总步数: {action_all.shape[0]}")
print(f"保存到: {output_path}")