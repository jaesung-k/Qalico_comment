from os import environ
import numpy as np
import time
import itertools as it
from pyomo.environ import *
from pyrsistent import v
import matplotlib.pyplot as plt

def opt(coe_dict:dict, x_grid:int, y_grid:int, flow_shape:int, flow:list, dire_rooms_dict:dict, area_list:list, area_tolerance:float, grid_shape:np.ndarray, grid_size:float):

    t1 = time.time()    # 処理前時刻 처리 전 시각

    # construct flow matrix
    zones = flow_shape + 1
    flo = np.array(flow)
    flo = np.triu(flo.reshape([flow_shape,flow_shape]), k=1) # np.triu(flo.reshape([5,5], k=1)
    flo_self = np.identity(flow_shape,  int) # 단위행렬
    flo_sum = (flo + flo.T + 100 * flo_self) / 100 # 행렬 + 역행렬 + 100*단위행렬 ???

    print("\n--------Setting Values in main process--------")
    print("zones : {}".format(zones))
    print("flo_sum : \n{}".format(flo_sum))
    print("area_list : {}".format(area_list))
    print("grid_shape : \n{}".format(grid_shape))
    print("x_grid : {}".format(x_grid))
    print("y_grid : {}".format(y_grid))
    print("grid_size : {}".format(grid_size))
    print("n_rooms : {}".format(dire_rooms_dict["N"]))
    print("s_rooms : {}".format(dire_rooms_dict["S"]))
    print("e_rooms : {}".format(dire_rooms_dict["E"]))
    print("w_rooms : {}".format(dire_rooms_dict["W"]))

    # grid settings
    x_nlocations = int(x_grid)
    y_nlocations = int(y_grid)
    nlocations = x_nlocations * y_nlocations
    grid_area = ga = np.square(grid_size) / 1000000

    def gen_locations(x_n, y_n) -> list:
        # 좌표
        grid_list = np.empty((y_n, x_n, 2))
        for i in range(y_n):
            for j in range(x_n):
                grid_list[i][j][0] = i
                grid_list[i][j][1] = j
        grid_list = np.reshape(grid_list, [x_n*y_n, 2])
        print("grid_list : \n{}".format(grid_list))
        # 줄행렬
        all_diffs = np.expand_dims(grid_list, axis=1) - np.expand_dims(grid_list, axis=0)
        distances_ori = np.sqrt(np.sum(all_diffs ** 2, axis=-1))
        distances = distances_ori / np.amax(distances_ori)

        return grid_list, distances, distances_ori
                    
    grid_list, distances, distances_ori = gen_locations(x_nlocations, y_nlocations)
    print("grid_loc[10][0] : {}".format(grid_list.shape))
    print("distances shape : {}".format(distances.shape))

    model = ConcreteModel()

    # q = (6,100) binary matrix 이산변수 행렬
    model.zones_param = Param(default=zones-1)
    model.nlocations_param = Param(default=nlocations-1)
    model.zones_range = RangeSet(0, model.zones_param)
    model.nlocations_range = RangeSet(0, model.nlocations_param)
    model.q = Var(model.zones_range, model.nlocations_range, domain=Binary)

    # grid_shape=0 일 경우, 
    invalid_grid = np.where(np.array(grid_shape.ravel()) == 0)[0]
    print("invalid_grids : \n{}".format(invalid_grid))
    model.invalid_grid = ConstraintList()
    for i in list(invalid_grid):
        model.invalid_grid.add(model.q[zones-1,i] == 1)
    
    # binary matrix
    flo_dis = np.einsum("ik,jl->ijkl",flo_sum,distances)
    print("flo_dis shape : {}".format(flo_dis.shape))
    cost_place_matrix = flo_dis.reshape(((zones-1) * nlocations, (zones-1) * nlocations))
    cost_place_matrix = np.triu(cost_place_matrix + np.triu(cost_place_matrix.T, k=1))
    zn = ((zones-1) * nlocations)
    zn_range = range(zn)
    model.zn_param = Param(default=zn-1)
    model.zn_range = RangeSet(0, model.zn_param)
    model.binary = Var(model.zn_range, model.zn_range, domain=Binary)
    cost_place=0
    for i in zn_range:
        for j in zn_range:
            cost_place += (cost_place_matrix[(i,j)] * model.binary[i,j]) / ((nlocations * (zones-1)) ** 2)

    # sum_poly
    grid_n = np.arange(nlocations)
    grid_loc = grid_n.reshape([y_nlocations, x_nlocations])
    grid_loc_edge_x = [grid_loc[0][i] for i in range(x_nlocations-1)] + [grid_loc[-1][i] for i in range(x_nlocations-1)]
    grid_loc_edge_y = [grid_loc[i][0] for i in range(1, y_nlocations-1)] + [grid_loc[i][-1] for i in range(y_nlocations)]
    grid_loc_edge = grid_loc_edge_x + grid_loc_edge_y
    grid_n_t = np.reshape(grid_loc.T, [nlocations])
    cost_rec_row = 0
    cost_rec_column = 0
    cost_rec_edge = 0
    for i in range(nlocations-y_nlocations):
        for k in range(zones):
            cost_rec_row += (model.q[k,grid_n_t[i]] + (-model.q[k,grid_n_t[i+y_nlocations]])) ** 2
    for i in range(nlocations-x_nlocations):
        for k in range(zones):
            cost_rec_column += (model.q[k,i] + (-model.q[k,i+x_nlocations])) ** 2    
    for i in grid_loc_edge:
        for k in range(zones-1):
            cost_rec_edge += model.q[k,i]
    cost_rec = (cost_rec_row + cost_rec_column + cost_rec_edge) / (nlocations*2 + x_nlocations + y_nlocations) # 正規化

    # dire 이산변수 행렬
    cost_dire_sum = 0
    dire_room_n = 0
    for direction, dire_rooms_list in dire_rooms_dict.items():
        if dire_rooms_list is None:
            cost_dire_sum += 0
        elif direction == "N": # Face north
            for i in range(len(dire_rooms_list)):
                for j in range(nlocations):
                    cost_dire_sum += (((y_nlocations-1)-grid_list[j][0] * model.q[dire_rooms_list[i],j]) / (area_list[dire_rooms_list[i]]/ga))
            dire_room_n += len(dire_rooms_list)
        elif direction == "E": # Face east
            for i in range(len(dire_rooms_list)):
                for j in range(nlocations):
                    cost_dire_sum += (((y_nlocations-1)-grid_list[j][1] * model.q[dire_rooms_list[i],j]) / (area_list[dire_rooms_list[i]]/ga))
            dire_room_n += len(dire_rooms_list)
        elif direction == "S": # Face south
            for i in range(len(dire_rooms_list)):
                for j in range(nlocations):
                    cost_dire_sum += ((grid_list[j][0] * model.q[dire_rooms_list[i],j]) / (area_list[dire_rooms_list[i]]/ga))
            dire_room_n += len(dire_rooms_list)
        elif direction == "W": # Face west
            for i in range(len(dire_rooms_list)):
                for j in range(nlocations):
                    cost_dire_sum += ((grid_list[j][1] * model.q[dire_rooms_list[i],j]) / (area_list[dire_rooms_list[i]]/ga))
            dire_room_n += len(dire_rooms_list)
        else:
            cost_dire_sum = 0
            dire_room_n = 1
    if all([i == None for i in dire_rooms_dict.values()]):
        dire_room_n = 1
    cost_dire_sum /= dire_room_n # 正規化
    
    # area penalty
    area_constraints = 0
    for n in range(len(area_list)):
        diff_area = (area_list[n]/ga - sum([model.q[n,i] for i in range(nlocations)])) ** 2
        area_constraints += (diff_area - (area_list[n]/ga * area_tolerance/100) ** 2)
    print("area_constraints : done")
    # print("area_constraints : {}".format(area_constraints))

    # zones one_hot
    zones_constraints = 0
    for n in range(nlocations):
        zones_constraints += (sum(model.q[:,n]) - 1) / nlocations
    print("zones_constraints : done")
    # print("zones_constraints : {}".format(zones_constraints))

    # penalty coefficients scale adjustment
    coe_dict["cost_place"] *= 1
    coe_dict["cost_rec"] *= 10
    coe_dict["cost_dire_sum"] /= 10
    coe_dict["area_constraints"] /= 10
    coe_dict["zones_constraints"] /= 0.1

    # constraints containerization
    constraints = coe_dict["zones_constraints"] * zones_constraints + coe_dict["area_constraints"] * area_constraints
    print("constraints : done")
    # print("constraints : {}".format(constraints))

    # object
    obj = coe_dict["cost_place"] * cost_place \
            + coe_dict["cost_rec"] * cost_rec \
            + coe_dict["cost_dire_sum"] * cost_dire_sum \
            + constraints
    print("obj : done")
    # print("obj : {}".format(obj))

    model.obj = Objective(expr = obj, sense = minimize)
    print("model.obj : done")

    solver_manager = SolverFactory('mindtpy')
    results = solver_manager.solve(model, mip_solver='glpk', nlp_solver='ipopt', tee=True)
    print(results)
    display(model)
    
    print('OF= ',value(model.obj))
    print('cost_place : ',value(model.cost_place))
    print('cost_rec : ',value(model.cost_rec))
    print('cost_dire_sum : ',value(model.cost_dire_sum))
    print(value(model.q[1,1]))

    result_q = np.empty((zones, nlocations))
    for i in range(1,zones):
        for j in range(1,nlocations):
            result_q[i][j] = value(model.q[i,j])
    
    print(result_q)
    result_q = np.array(result_q).transpose((1,0))
    print("\n--------result--------")
    ans_grid = result_q[:, np.newaxis, :].reshape([y_nlocations, x_nlocations, zones]).transpose(2,0,1)
    print("answer grid : \n{}".format(ans_grid))
    print("answer grid shape : {}".format(ans_grid.shape))
