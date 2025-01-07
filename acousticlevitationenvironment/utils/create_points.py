import numpy as np


def create_points(N, cube_size=0.12, max_attempts=1000):
    points = []
    attempts = 0
    
    while len(points) < N and attempts < max_attempts:
        # 在[-cube_size/2, cube_size/2]范围内随机生成一个点
        point = np.random.uniform(-cube_size/2, cube_size/2, 3)
        
        # 检查与已生成点之间的椭球体距离
        if all(np.linalg.norm((point - p) / np.array([0.015, 0.015, 0.03])) > 1 for p in points):
            points.append(point)
        
        attempts += 1
        
        if attempts == max_attempts:
            raise RuntimeError("Reached maximum attempts without generating enough points.")
    
    points = np.array(points)
    points[:, 2] += 0.12  # 修改所有z轴的值
    
    # 根据x轴的值从小到大进行排序
    points = points[points[:, 0].argsort()]
    
    return points


def create_points_multistage(N, cube_size=0.12, max_attempts=1000, distance=0.02):
    points = []
    attempts = 0

    while len(points) < N and attempts < max_attempts:
        # 在[-cube_size/2, cube_size/2]范围内随机生成一个点
        point = np.random.uniform(-cube_size/2, cube_size/2, 3)

        # 检查与已生成点之间的椭球体距离
        if all(np.linalg.norm((point - p) / np.array([0.015, 0.015, 0.03])) > 1 for p in points):
            points.append(point)

        attempts += 1

        if attempts == max_attempts:
            print(points)
            raise RuntimeError("Reached maximum attempts without generating enough start points.")

    points = np.array(points)

    # 根据x轴的值从小到大进行排序
    points = points[points[:, 0].argsort()]

    new_points = []

    for point in points:
        attempts = 0
        while attempts < max_attempts:
            # 生成 new_point，保证每个坐标值在指定范围内
            new_point = np.random.uniform(
                low=np.maximum(-cube_size / 2, point - distance),
                high=np.minimum(cube_size / 2, point + distance)
            )

            # 检查新点与已有新点之间的椭球体距离
            if all(np.linalg.norm((new_point - p) / np.array([0.015, 0.015, 0.03])) > 1 for p in new_points):
                new_points.append(new_point)
                break

            attempts += 1

            if attempts == 100:
                distance += 0.01
            elif attempts == 200:
                distance += 0.01

        if attempts == max_attempts:
            print(new_points)
            raise RuntimeError("Reached maximum attempts without generating enough target points.")
        
    new_points = np.array(new_points)

    points[:, 2] += 0.12  # 修改所有z轴的值
    new_points[:, 2] += 0.12  # 修改所有z轴的值

    return points, new_points