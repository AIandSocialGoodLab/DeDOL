import numpy as np

dy = [0,1,1,1]
dx = [-1,-1,0,1]


def generate_map(args):
    np.random.seed(args.ani_den_seed)
    if args.map_type == 'random':
        animal_density = np.random.uniform(low=0.2, high=1., size=[args.row_num, args.column_num])
        corners = [(0,0), (0, args.column_num - 1), (args.row_num - 1, 0), (args.row_num-1, args.column_num-1)]
        near_corners_1 = [(0,1),(1,0),(0, args.column_num-2), (1, args.column_num-1), (args.row_num-2, 0),
                (args.row_num-1,1), (args.row_num-1, args.column_num-2), (args.row_num-2, args.column_num-1)]
        near_corners_2 = [(0,2), (2,0),(1,1), 
                (0, args.column_num-3), (2, args.column_num-1), (1, args.column_num-2),
                (args.row_num-3, 0), (args.row_num-1,2), (args.row_num - 2, 1), 
                (args.row_num-1, args.column_num-3), (args.row_num-3, args.column_num-1), (args.row_num - 2, args.column_num -2)]
        for x in corners:
            animal_density[x] = np.clip(animal_density[x], a_min = None, a_max = 0.28)
        if args.row_num >= 5:
            for x in near_corners_1:
                animal_density[x] = np.clip(animal_density[x], a_min = None, a_max = 0.42)
        if args.row_num >= 7:
            for x in near_corners_2:
                animal_density[x] = np.clip(animal_density[x], a_min = None, a_max = 0.56)    
        return animal_density
    elif args.map_type == 'gauss':
        animal_density = Mountainmap(args.row_num, args.column_num, args.ani_den_seed)
        return animal_density

def Mountainmap(row, col, seed):
    np.random.seed(seed)

    map0 = np.zeros((row,col))
    map1 = np.zeros((row,col))
    map = np.zeros((row, col))
        
    map0[row // 2][:] = 0.5
    map0[0][:] = 0.1
    map0[row-1][:] = 0.1

    map1[:,col // 2] = 0.5
    map1[:,0] = 0.1
    map1[:,col - 1] = 0.1

    # print(map0)
    # print(map1)

    dec = row // 2
    dec = 0.4 / dec
    for i in range(1, row // 2):
        map0[i,:] = 0.1 + i * dec
        map1[:,i] = 0.1 + i * dec

    for i in range(row // 2 + 1, row - 1):
        map0[i,:] = 0.5 - (i - row // 2)* dec
        map1[:,i] = 0.5 - (i - row // 2)* dec

    # print(map0)
    # print(map1)
    map0 += np.random.rand(row, col) * 0.1
    map1 += np.random.rand(row, col) * 0.1
    map = map0 + map1
    map -= 0.08
    # print(map)
    map = map.clip(min = 0.1, max =1)
    return map



def Mountainmap_(row, col, seed = 666):

    np.random.seed(seed)

    def inmap(x,y):
        if 0 <= x and x < col and 0 <=y and y < row:
            return True
        else:
            return False

    def get_dis(i, j, mountain):
        mindis = 100
        for grid in mountain:
            dis = ((i - grid[0]) ** 2 + (j - grid[1]) ** 2) ** 0.5
            mindis = min(dis, mindis)
        return mindis

    map = np.zeros([row, col])
    num = np.random.randint(1,3)

    lowran = [0.5, 0.2]
    highran = [0.7, 0.4]

    x_ = [0, 3]
    y_ = [4, 0]
    dir_ = [3,2]
    den_ = [0.4, 0.6]


    for i in range(2):
        
        x = x_[i]
        y = y_[i]
        den = den_[i]
        #print(den)
        dir = dir_[i]
        
        mountain = []

        tx, ty = x,y
        while(inmap(tx,ty)):
            mountain.append((tx,ty))
            map[tx][ty] += den
            tx += dx[dir]
            ty += dy[dir]     

        tx, ty = x - dx[dir], y - dy[dir]
        while(inmap(tx, ty)):
            mountain.append((tx,ty))
            map[tx][ty] += den
            tx -= dx[dir]
            ty -= dy[dir]

        for i in range(row):
            for j in range(col):
                dis = get_dis(i,j, mountain)
                #print(i,j, dis)
                if dis == 0.0:
                    continue
                map[i][j] += den - dis * 0.15 * den
                
    
    
    #noise = np.random.uniform(low = -0.1, high = 0.1, size = [row, col])
    #map += noise

    map = np.maximum(map, 0.2)
    map = np.minimum(map, 0.95)


    #print(map)
    return map

