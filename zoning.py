"""
zoning.py : "Qalico" zoning function
    called from app.py

@version    0.1.1
@author     Naoki TAGAWA <tagawa.naoki@teragroove.com>
@copyright  2022 Naoki TAGAWA
@license    MIT(https://choosealicense.com/licenses/mit/)
@sponserdby "MITOU Target Porject" of IPA(https://www.ipa.go.jp/jinzai/target/index.html)
"""
from os import environ
import amplify as ap
import numpy as np
import time
import itertools
from pyomo.environ import *

def opt(coe_dict:dict, x_grid:int, y_grid:int, flow_shape:int, flow:list, dire_rooms_dict:dict, area_list:list, area_tolerance:float, grid_shape:np.ndarray, grid_size:float):

    t1 = time.time()    # 処理前時刻 처리 전 시각

    # construct flow matrix
    zones = flow_shape + 1
    flo = np.array(flow)
    flo = np.triu(flo.reshape([flow_shape,flow_shape]), k=1)
    flo_self = np.identity(flow_shape,  int)
    flo_sum = (flo + flo.T + 100 * flo_self) / 100

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
    print("ga : {}".format(ga))

    def gen_locations(x_n, y_n) -> list:
        # 座標 
        # 좌표
        grid_list = np.empty((y_n, x_n, 2))
        for i in range(y_n):
            for j in range(x_n):
                grid_list[i][j][0] = i
                grid_list[i][j][1] = j
        grid_list = np.reshape(grid_list, [x_n*y_n, 2])
        print("grid_list : \n{}".format(grid_list))
        # 距離行列 
        # 줄행렬
        all_diffs = np.expand_dims(grid_list, axis=1) - np.expand_dims(grid_list, axis=0)
        distances_ori = np.sqrt(np.sum(all_diffs ** 2, axis=-1))
        distances = distances_ori / np.amax(distances_ori)

        return grid_list, distances, distances_ori
                    
    grid_list, distances, distances_ori = gen_locations(x_nlocations, y_nlocations)
    print("grid_loc[10][0] : {}".format(grid_list.shape))
    print("distances shape : {}".format(distances.shape))
    print("distances : \n{}".format(distances))

    # 変数配列の生成 
    # 변수 배열 생성
    gen = ap.SymbolGenerator(ap.BinaryPoly)
    print("gen : \n{}".format(gen))
    q = gen.array(zones, nlocations) #gen.array(6, 100)
    print("q : \n{}".format(q))

    # 無効グリッドを排除 
    # 비활성 그리드 배제
    invalid_grid = np.where(np.array(grid_shape.ravel()) == 0)[0]
    print("invalid_grids : \n{}".format(invalid_grid))
    for i in list(invalid_grid):
        q[zones-1][i] = ap.BinaryPoly(1) # q[5][i] = q_0, ap.BinarayPoly(1) = 1
    print("invalid_grids : \n{}".format(ap.BinaryPoly(1)))


    # クラスター化・ゾーン配置 
    # 클러스터화·존 배치
    flo_dis = np.einsum("ik,jl->ijkl",flo_sum,distances) #np.einsum("ik,jl->ijkl",(5,5),(100,100))
    print("flo_dis shape : {}".format(flo_dis.shape)) #(5,100,5,100)
    print("flo_dis : {}".format(flo_dis))
    cost_place_matrix = flo_dis.reshape(((zones-1) * nlocations, (zones-1) * nlocations)) #flo_dis.reshape(500,500)
    cost_place_matrix = np.triu(cost_place_matrix + np.triu(cost_place_matrix.T, k=1))
    cost_place = ap.BinaryMatrix(cost_place_matrix)
    cost_place = cost_place.to_Poly() / ((nlocations * (zones-1)) ** 2) # 正規化

    # 矩形化 
    # 직사각형화
    grid_n = np.arange(nlocations)
    grid_loc = grid_n.reshape([y_nlocations, x_nlocations])
    grid_loc_edge_x = [grid_loc[0][i] for i in range(x_nlocations-1)] + [grid_loc[-1][i] for i in range(x_nlocations-1)]
    grid_loc_edge_y = [grid_loc[i][0] for i in range(1, y_nlocations-1)] + [grid_loc[i][-1] for i in range(y_nlocations)]
    grid_loc_edge = grid_loc_edge_x + grid_loc_edge_y
    grid_n_t = np.reshape(grid_loc.T, [nlocations])

    cost_rec_row = ap.sum_poly(nlocations-y_nlocations, lambda i: ap.sum_poly(zones, lambda k: ((q[k][grid_n_t[i]] + (-q[k][grid_n_t[i+y_nlocations]])) ** 2)))
    cost_rec_column = ap.sum_poly(nlocations-x_nlocations, lambda i: ap.sum_poly(zones, lambda k: ((q[k][i] + (-q[k][i+x_nlocations])) ** 2)))
    cost_rec_edge = ap.sum_poly((x_nlocations + y_nlocations -2) * 2, lambda i: ap.sum_poly(zones-1, lambda k: q[k][grid_loc_edge[i]]))
    cost_rec_edge = ap.sum_poly(grid_loc_edge, lambda i: ap.sum_poly(zones-1, lambda k: q[k][i]))
    cost_rec = (cost_rec_row + cost_rec_column + cost_rec_edge) / (nlocations*2 + x_nlocations + y_nlocations) # 正規化

    # 方角配置 (grid_list[i][0]: y-axis, gird_list[i][1]: x-axis) 
    # 방향 배치
    cost_dire_sum = ap.BinaryPoly()
    dire_room_n = 0
    for direction, dire_rooms_list in dire_rooms_dict.items():
        if dire_rooms_list is None:
            cost_dire_sum += 0
        elif direction == "N": # Face north
            cost_dire_sum += ap.sum_poly(
                len(dire_rooms_list), lambda i: ap.sum_poly(nlocations, lambda j: (y_nlocations-1)-grid_list[j][0] * q[dire_rooms_list[i]][j]
                    ) /(area_list[dire_rooms_list[i]]/ga) # 正規化
                )
            dire_room_n += len(dire_rooms_list)
        elif direction == "E": # Face east
            cost_dire_sum += ap.sum_poly(
                len(dire_rooms_list), lambda i: ap.sum_poly(nlocations, lambda j: (y_nlocations-1)-grid_list[j][1] * q[dire_rooms_list[i]][j]
                    ) /(area_list[dire_rooms_list[i]]/ga) # 正規化
                )
            dire_room_n += len(dire_rooms_list)
        elif direction == "S": # Face south
            cost_dire_sum += ap.sum_poly(
                len(dire_rooms_list), lambda i: ap.sum_poly(nlocations, lambda j: grid_list[j][0] * q[dire_rooms_list[i]][j]
                    ) /(area_list[dire_rooms_list[i]]/ga) # 正規化
                )
            dire_room_n += len(dire_rooms_list)
        elif direction == "W": # Face west
            cost_dire_sum += ap.sum_poly(
                len(dire_rooms_list), lambda i: ap.sum_poly(nlocations, lambda j: grid_list[j][1] * q[dire_rooms_list[i]][j]
                    ) /(area_list[dire_rooms_list[i]]/ga) # 正規化
                )
            dire_room_n += len(dire_rooms_list)
        else:
            cost_dire_sum = 0
            dire_room_n = 1
    if all([i == None for i in dire_rooms_dict.values()]):
        dire_room_n = 1
    print("dire_room_n : {}".format(dire_room_n))
    cost_dire_sum /= dire_room_n # 正規化

    # 面積制約 
    # 면적 제약
    area_constraints = ap.BinaryConstraints()
    for n in range(len(area_list)):
        diff_area = (area_list[n]/ga - ap.sum_poly([q[n][i] for i in range(nlocations)])) ** 2
        area_constraints += ap.constraint.penalty(diff_area, le = (area_list[n]/ga * area_tolerance/100) ** 2, label=f"penalty_area{n}")
    
    # 重複防止制約 
    # 중복 방지 제약
    zones_constraints = sum(ap.constraint.one_hot(q[:, n], label=f"one_hot_zone{n}") for n in range(nlocations)) / nlocations # 正規化


    # penalty coefficients scale adjustment
    coe_dict["cost_place"] *= 1
    coe_dict["cost_rec"] *= 10
    coe_dict["cost_dire_sum"] /= 10
    coe_dict["area_constraints"] /= 10
    coe_dict["zones_constraints"] /= 0.1

    # constraints containerization
    constraints = coe_dict["zones_constraints"] * zones_constraints + coe_dict["area_constraints"] * area_constraints

    # model
    model = coe_dict["cost_place"] * cost_place \
            + coe_dict["cost_rec"] * cost_rec \
            + coe_dict["cost_dire_sum"] * cost_dire_sum \
            + constraints
    
    # client settings
    client = ap.client.FixstarsClient()
    client.token = "1bQRiJKq0PXWeQKUICyWgRjx3GOmgNL8"
    client.parameters.timeout = 20000    # タイムアウト10秒 타임 아웃 10초

    # solver
    solver = ap.Solver(client)


    solver.filter_solution = True
    result = solver.solve(model)
    if len(result) == 0:
        raise RuntimeError("Any one of constraints is not satisfied")

    # result
    energy, values, is_feasible = result[0].energy, result[0].values, result[0].is_feasible
    q_values = ap.decode_solution(q, values, 1)
    q_values = np.array(q_values).transpose((1,0))

    print("\n--------result--------")
    ans_grid = q_values[:, np.newaxis, :].reshape([y_nlocations, x_nlocations, zones]).transpose(2,0,1)
    print("answer grid : \n{}".format(ans_grid))
    print("answer grid shape : {}".format(ans_grid.shape))   # 各ゾーンごとのグリッド表現 각 구역별 그리드 표현

    route = np.where(np.array(q_values) == 1)[1]
    routes_list = [np.where(np.array(route) == i)[0] for i in range(zones-1)]

    # show result
    for i in range(zones-1):
        print("\n")
        print("zone" + str(i))
        print(routes_list[i])
        print(len(routes_list[i]) * ga)
    print("\n")
    print(f"is_feasible : {is_feasible}")
    # print(f"check_constraints : {model.check_constraints(values)}")
    print(f"energy : {energy}")
 
    t2 = time.time()    # 処理後時刻 처리 후 시각
    elapsed_time = t2 -t1
    print(f"経過時間 : {elapsed_time}")

    ans_list = [[int(n) for n in routes_list[i]] for i in range(len(routes_list))]
    answer_list = [[str(i) + '-' + str(s) for s in ans_list[i]] for i in range(len(ans_list))]
    a = list(itertools.chain.from_iterable(answer_list))


    return a




"""
    model = AbstractModel()
    model.x = Var(within=NonNegativeReals)
    model.y = Var(bounds = (0, None))
    model.z = Var()
    model.c1 = Constraint(expr=model.b + 5*model.c <= model.a)
"""
