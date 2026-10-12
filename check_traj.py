import json, numpy as np
with open('/home/i2slab4/yj/EgoX/datasets/eating1/meta.json') as f:
    meta = json.load(f)
ego = np.array(meta['test_datasets'][0]['ego_extrinsics'])
print(f'Total frames: {len(ego)}')
positions = []
for i in range(min(49, len(ego))):
    w2c = np.eye(4); w2c[:3,:] = ego[i]
    c2w = np.linalg.inv(w2c)
    positions.append(c2w[:3, 3])
positions = np.array(positions)
print(f'X: [{positions[:,0].min():.3f}, {positions[:,0].max():.3f}]')
print(f'Y: [{positions[:,1].min():.3f}, {positions[:,1].max():.3f}]')
print(f'Z: [{positions[:,2].min():.3f}, {positions[:,2].max():.3f}]')
for i in range(1, len(positions)):
    dist = np.linalg.norm(positions[i] - positions[i-1])
    if dist > 0.03:
        print(f'Jump {i-1}->{i}: {dist:.4f}m')
for i in range(min(49, len(positions))):
    print(f'  F{i}: [{positions[i,0]:.3f}, {positions[i,1]:.3f}, {positions[i,2]:.3f}]')
