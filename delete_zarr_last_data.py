from diffusion_policy.common.replay_buffer import ReplayBuffer

zarr_path = "/home/zaq/bc_data/gello/replay_buffer.zarr"
rb = ReplayBuffer.create_from_path(zarr_path, mode='a')

print("before:", rb.n_episodes, "episodes,", rb.n_steps, "steps")
if rb.n_episodes > 0:
    rb.drop_episode()   # 删除最后一个 episode
print("after :", rb.n_episodes, "episodes,", rb.n_steps, "steps")
