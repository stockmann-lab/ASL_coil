import numpy as np
from collections import deque

def bfs_boundary(target, start_point, threshold=0.5, from_neighbor=False, dxyz_max=[-1, -1, -1], n_max=10000, from_average=False):
    
    start_point = tuple(np.asarray(start_point).tolist())
    xyz_min = np.zeros(3, dtype=int)
    xyz_max = np.array(target.shape) - 1
    for i in range(3):
        if dxyz_max[i] >=0:
            xyz_min[i] = np.max([xyz_min[i], start_point[i] - dxyz_max[i]])
            xyz_max[i] = np.min([xyz_max[i], start_point[i] + dxyz_max[i]])
    
    average = 0
    explored = np.full_like(target, False, dtype=bool)
    explored[start_point] = True
    d = deque([start_point,])

    dxyz = []
    for i in [-1, 0, 1]:
        dxyz += [(-1, 0, i), (1, 0, i), (0, -1, i), (0, 1, i)]

    n = 0
    while len(d) > 0 and n < n_max:
        n += 1
        point = d.popleft()
        if from_average:
            average = average * (n - 1)/n + 1/n * target[point]
        
        for dx, dy, dz in dxyz:
            adj_point = np.array(point) + np.array([dx, dy, dz])
            if np.all(adj_point >= xyz_min) and np.all(adj_point <= xyz_max):
                adj_point = tuple(adj_point.tolist())
                if not explored[adj_point]:
                    good = False
                    if from_neighbor:
                        if target[adj_point] > threshold * target[point]:
                            good = True
                    elif from_average:
                        if target[adj_point] > threshold * average:
                            good = True
                    elif target[adj_point] > threshold * target[start_point]:
                        good = True
                    if good: 
                        d.append(adj_point)
                        explored[adj_point] = True

    if n == n_max:
        print('Maxed out')
                    
    return explored

def thicken(explored, dx, dy, dz):
    dxyz_max = np.array([dx, dy, dz])
    xyz_min = np.zeros(3, dtype=int)
    xyz_max = np.array(explored.shape) - 1
    for point in np.argwhere(explored).tolist():
        point = tuple(point)
        for i in range(3):
            xyz_min[i] = np.max([0, point[i] - dxyz_max[i]])
            xyz_max[i] = np.min([explored.shape[i], point[i] + dxyz_max[i]])
        explored[xyz_min[0]: xyz_max[0] + 1, xyz_min[1]: xyz_max[1] + 1, xyz_min[2]: xyz_max[2] + 1] = True
    return explored
    
