#!/usr/bin/env python3
"""
Diffusion Policy 数据诊断脚本
运行后请把全部输出复制发给我
"""

import numpy as np
import zarr
import sys
from pathlib import Path

# 尝试导入官方 ReplayBuffer
try:
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    print("✅ 成功导入 diffusion_policy.common.replay_buffer.ReplayBuffer")
    USE_REPLAYBUFFER = True
except ImportError as e:
    print(f"⚠️ 无法直接导入 ReplayBuffer: {e}")
    print("尝试手动从 zarr 读取...")
    USE_REPLAYBUFFER = False

def main():
    zarr_path = "/home/zaq/bc_data/gello/replay_buffer.zarr"
    print("="*80)
    print("Diffusion Policy 数据诊断脚本")
    print("="*80)
    print(f"zarr 路径: {zarr_path}")
    print(f"是否存在: {Path(zarr_path).exists()}\n")

    if not Path(zarr_path).exists():
        print("❌ zarr 文件不存在！请检查路径是否正确。")
        return

    # 1. 直接用 zarr 打开查看结构
    try:
        root = zarr.open(zarr_path, mode='r')
        print("📁 zarr 根目录下的组和数组:")
        for key in root.keys():
            obj = root[key]
            if isinstance(obj, zarr.Group):
                print(f"  📂 Group: {key}")
                for subkey in obj.keys():
                    subobj = obj[subkey]
                    shape = subobj.shape if hasattr(subobj, 'shape') else 'N/A'
                    dtype = subobj.dtype if hasattr(subobj, 'dtype') else 'N/A'
                    print(f"    └─ {subkey:15} shape={shape}  dtype={dtype}")
            else:
                shape = obj.shape if hasattr(obj, 'shape') else 'N/A'
                dtype = obj.dtype if hasattr(obj, 'dtype') else 'N/A'
                print(f"  📊 Array: {key:15} shape={shape}  dtype={dtype}")
        print()
    except Exception as e:
        print(f"❌ 打开 zarr 失败: {e}")
        return

    # 2. 使用 ReplayBuffer（推荐方式）
    if USE_REPLAYBUFFER:
        try:
            rb = ReplayBuffer.copy_from_path(zarr_path)
            print(f"✅ ReplayBuffer 加载成功！")
            print(f"   总 episode 数: {rb.n_episodes}")
            print(f"   总数据长度: {len(rb)}")
            print(f"   episode_ends shape: {rb.episode_ends.shape}")
            print(f"   episode_ends: {rb.episode_ends.tolist()}\n")

            # 检查前几个 episode 的详细内容
            n_check = min(3, rb.n_episodes)
            for i in range(n_check):
                print(f"--- Episode {i} ---")
                ep = rb.get_episode(i)
                for k, v in ep.items():
                    if isinstance(v, np.ndarray):
                        print(f"  {k:20} shape={v.shape}  dtype={v.dtype}  "
                              f"range=[{v.min():.4f}, {v.max():.4f}]")
                    else:
                        print(f"  {k:20} type={type(v)}")
                print()

            # 检查 action 和 robot_joint 的维度是否一致
            if 'action' in rb.data.keys():
                action_shape = rb.data['action'].shape
                print(f"action 全局 shape: {action_shape}")
            if 'robot_joint' in rb.data.keys():
                joint_shape = rb.data['robot_joint'].shape
                print(f"robot_joint 全局 shape: {joint_shape}")
            if 'gripper_position' in rb.data.keys():
                gripper_shape = rb.data['gripper_position'].shape
                print(f"gripper_position 全局 shape: {gripper_shape}")

            # 检查图像形状
            for cam in ['camera_0', 'camera_1', 'camera_2']:
                if cam in rb.data.keys():
                    img = rb.data[cam]
                    print(f"{cam:12} shape={img.shape}  dtype={img.dtype}  "
                          f"value range=[{img.min()}, {img.max()}]")

        except Exception as e:
            print(f"❌ ReplayBuffer 加载失败: {e}")
            import traceback
            traceback.print_exc()

    # 3. 额外手动检查（保险）
    print("\n" + "="*60)
    print("额外手动检查 data 组下的关键键:")
    try:
        root = zarr.open(zarr_path, mode='r')
        data_group = root['data'] if 'data' in root else root
        
        important_keys = ['action', 'robot_joint', 'gripper_position', 
                         'camera_0', 'camera_1', 'camera_2', 'agent_pos']
        
        for key in important_keys:
            if key in data_group:
                arr = data_group[key]
                print(f"{key:18} shape={arr.shape}  dtype={arr.dtype}")
                if len(arr.shape) > 0 and arr.size > 0:
                    print(f"    min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")
            else:
                print(f"{key:18} ❌ 不存在")
    except Exception as e:
        print(f"手动检查出错: {e}")

    print("\n✅ 诊断脚本运行完毕！")
    print("请把上面**全部输出**复制发给我，我会据此给出精确修复方案。")

if __name__ == "__main__":
    main()