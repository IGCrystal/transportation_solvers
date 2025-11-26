"""
transportation_solvers

介绍：
- 平衡辅助（自动添加虚拟行/列）
- 初始可行解：西北角法、最小元素法、Vogel (VAM)
- 优化：MODI (u-v 方法) + 闭路(stepping-stone) 调整
- 命令行界面：支持 data.json 输入，--method, --optimize, --csv 输出等

用法示例:
  python main.py --input data.json --method vam --optimize --output-csv result.csv

data.json 格式示例:
{
  "costs": [[19,30,50,10],[70,30,40,60],[40,8,70,20]],
  "supply": [7,9,18],
  "demand": [5,8,7,14],
  "method": "vam"
}

注意：本实现尽力处理退化情形，但在极端或高度退化的例子中可能需要额外的数值策略。
"""

from __future__ import annotations
import argparse
import json
import math
import csv
import copy
from typing import List, Tuple, Set
import numpy as np

EPS = 1e-9

# ----------------------------- 辅助函数 -----------------------------

def balance_problem(costs: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    costs = costs.copy().astype(float)
    supply = supply.copy().astype(float)
    demand = demand.copy().astype(float)
    s_sum = float(np.sum(supply))
    d_sum = float(np.sum(demand))
    if abs(s_sum - d_sum) < EPS:
        return costs, supply, demand
    if s_sum > d_sum:
        diff = s_sum - d_sum
        costs = np.hstack([costs, np.zeros((costs.shape[0], 1), dtype=float)])
        demand = np.append(demand, diff)
    else:
        diff = d_sum - s_sum
        costs = np.vstack([costs, np.zeros((1, costs.shape[1]), dtype=float)])
        supply = np.append(supply, diff)
    return costs, supply, demand


def pretty_print_allocation(allocation: np.ndarray, costs: np.ndarray) -> None:
    rows, cols = costs.shape
    header = "      " + "".join([f"B{j+1:<7}" for j in range(cols)])
    print(header)
    print("      " + "".join(["--------" for _ in range(cols)]))
    total_cost = 0.0
    for i in range(rows):
        row_str = f"A{i+1:<4}| "
        for j in range(cols):
            val = allocation[i, j]
            row_str += f"{val:>6.2f} "
            total_cost += val * costs[i, j]
        print(row_str)
    print("-" * 8 * cols)
    print(f"Total cost: {total_cost:.2f}")


def to_csv(allocation: np.ndarray, costs: np.ndarray, filename: str) -> None:
    rows, cols = costs.shape
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ["", *[f"B{j+1}" for j in range(cols)], "Supply"]
        w.writerow(header)
        for i in range(rows):
            row = [f"A{i+1}"] + [allocation[i, j] for j in range(cols)] + [sum(allocation[i, :])]
            w.writerow(row)
        demand_row = ["Demand"] + [sum(allocation[:, j]) for j in range(rows, rows+cols)]

# -------------------------- 初始可行解算法 --------------------------

def solve_northwest_corner(costs: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> np.ndarray:
    rows, cols = costs.shape
    allocation = np.zeros((rows, cols), dtype=float)
    s_vec = supply.copy().astype(float)
    d_vec = demand.copy().astype(float)
    r, c = 0, 0
    while r < rows and c < cols:
        qty = min(s_vec[r], d_vec[c])
        allocation[r, c] = qty
        s_vec[r] -= qty
        d_vec[c] -= qty
        if abs(s_vec[r]) < EPS and abs(d_vec[c]) < EPS:
            # 双耗尽，移动到下一行与下一列
            r += 1
            c += 1
        elif abs(s_vec[r]) < EPS:
            r += 1
        elif abs(d_vec[c]) < EPS:
            c += 1
        else:
            break
    return allocation


def solve_least_cost(costs: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> np.ndarray:
    rows, cols = costs.shape
    allocation = np.zeros((rows, cols), dtype=float)
    s_vec = supply.copy().astype(float)
    d_vec = demand.copy().astype(float)
    temp = costs.copy().astype(float)
    # 将不可用设为 inf
    while np.sum(s_vec) > EPS and np.sum(d_vec) > EPS:
        idx = np.unravel_index(np.argmin(temp), temp.shape)
        r, c = idx
        if not np.isfinite(temp[r, c]):
            break
        qty = min(s_vec[r], d_vec[c])
        allocation[r, c] = qty
        s_vec[r] -= qty
        d_vec[c] -= qty
        if abs(s_vec[r]) < EPS:
            temp[r, :] = float('inf')
        if abs(d_vec[c]) < EPS:
            temp[:, c] = float('inf')
    return allocation


def solve_vogel(costs: np.ndarray, supply: np.ndarray, demand: np.ndarray) -> np.ndarray:
    cost_matrix = costs.copy().astype(float)
    rows, cols = cost_matrix.shape
    allocation = np.zeros((rows, cols), dtype=float)
    s_vec = supply.copy().astype(float)
    d_vec = demand.copy().astype(float)
    row_done = [False] * rows
    col_done = [False] * cols

    def row_penalty(r):
        vals = [cost_matrix[r, c] for c in range(cols) if not col_done[c] and np.isfinite(cost_matrix[r, c])]
        if len(vals) >= 2:
            vals.sort()
            return vals[1] - vals[0]
        elif len(vals) == 1:
            return vals[0]
        return -1

    def col_penalty(c):
        vals = [cost_matrix[r, c] for r in range(rows) if not row_done[r] and np.isfinite(cost_matrix[r, c])]
        if len(vals) >= 2:
            vals.sort()
            return vals[1] - vals[0]
        elif len(vals) == 1:
            return vals[0]
        return -1

    while np.sum(s_vec) > EPS and np.sum(d_vec) > EPS:
        r_pen = [row_penalty(r) if not row_done[r] else -1 for r in range(rows)]
        c_pen = [col_penalty(c) if not col_done[c] else -1 for c in range(cols)]
        if max(r_pen) <= -1 and max(c_pen) <= -1:
            break
        if max(r_pen) >= max(c_pen):
            target_r = int(np.argmax(r_pen))
            candidate_cs = [c for c in range(cols) if not col_done[c] and np.isfinite(cost_matrix[target_r, c])]
            target_c = min(candidate_cs, key=lambda c: cost_matrix[target_r, c])
        else:
            target_c = int(np.argmax(c_pen))
            candidate_rs = [r for r in range(rows) if not row_done[r] and np.isfinite(cost_matrix[r, target_c])]
            target_r = min(candidate_rs, key=lambda r: cost_matrix[r, target_c])

        qty = min(s_vec[target_r], d_vec[target_c])
        allocation[target_r, target_c] = qty
        s_vec[target_r] -= qty
        d_vec[target_c] -= qty
        if abs(s_vec[target_r]) < EPS:
            row_done[target_r] = True
        if abs(d_vec[target_c]) < EPS:
            col_done[target_c] = True

    return allocation

# ----------------------- MODI (u-v) + stepping-stone -----------------------

def get_basic_cells_from_allocation(allocation: np.ndarray) -> Set[Tuple[int, int]]:
    rows, cols = allocation.shape
    basics = set((i, j) for i in range(rows) for j in range(cols) if allocation[i, j] > EPS)
    return basics


def ensure_non_degenerate(basics: Set[Tuple[int, int]], costs: np.ndarray, m: int, n: int) -> Set[Tuple[int, int]]:
    # 若基础变量数 < m+n-1，则加入最低成本的非基础格子（分配 0）以破除退化
    required = m + n - 1
    basics = set(basics)
    if len(basics) >= required:
        return basics
    # 按成本增序加入
    candidates = [(costs[i, j], i, j) for i in range(m) for j in range(n) if (i, j) not in basics]
    candidates.sort()
    for _, i, j in candidates:
        basics.add((i, j))
        if len(basics) >= required:
            break
    return basics


def compute_uv(basics: Set[Tuple[int, int]], costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m, n = costs.shape
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    # pick u0 = 0
    basics_list = list(basics)
    u[0] = 0.0
    changed = True
    while changed:
        changed = False
        for i, j in basics_list:
            if math.isfinite(u[i]) and not math.isfinite(v[j]):
                v[j] = costs[i, j] - u[i]
                changed = True
            if math.isfinite(v[j]) and not math.isfinite(u[i]):
                u[i] = costs[i, j] - v[j]
                changed = True
    return u, v


def compute_opportunity_costs(allocation: np.ndarray, basics: Set[Tuple[int, int]], costs: np.ndarray) -> np.ndarray:
    m, n = costs.shape
    u, v = compute_uv(basics, costs)
    deltas = np.full((m, n), np.nan)
    for i in range(m):
        for j in range(n):
            if (i, j) in basics:
                deltas[i, j] = 0.0
            else:
                if math.isfinite(u[i]) and math.isfinite(v[j]):
                    deltas[i, j] = costs[i, j] - (u[i] + v[j])
                else:
                    # 如果 u/v 未全确定（极端退化），先计算后再行处理
                    deltas[i, j] = costs[i, j] - (0.0 + 0.0)
    return deltas


def find_cycle(start: Tuple[int, int], basics: Set[Tuple[int, int]], m: int, n: int) -> List[Tuple[int, int]]:
    # 在基础格子集合 + start 中寻找一个闭环，路径格子交替行/列移动。
    nodes = set(basics)
    nodes.add(start)

    def neighbors_in_row(cell):
        i, j = cell
        return [(i, jj) for jj in range(n) if (i, jj) in nodes and jj != j]

    def neighbors_in_col(cell):
        i, j = cell
        return [(ii, j) for ii in range(m) if (ii, j) in nodes and ii != i]

    # DFS: 路径列表，最后回到 start，长度 >=4，且交替方向
    path = [start]

    def dfs(current, expect_row_move, visited: Set[Tuple[int,int]]):
        # expect_row_move: 下一步应该沿行（True）还是沿列（False）
        if len(path) >= 4 and current == start:
            return True
        if expect_row_move:
            for nb in neighbors_in_row(current):
                if nb == start and len(path) >= 4:
                    path.append(nb)
                    return True
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    if dfs(nb, not expect_row_move, visited):
                        return True
                    path.pop()
                    visited.remove(nb)
        else:
            for nb in neighbors_in_col(current):
                if nb == start and len(path) >= 4:
                    path.append(nb)
                    return True
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    if dfs(nb, not expect_row_move, visited):
                        return True
                    path.pop()
                    visited.remove(nb)
        return False

    # 尝试从 start 的行或列开始
    # We need to start with either a row or column move; try both
    for start_with_row in (True, False):
        path = [start]
        visited = set([start])
        if dfs(start, start_with_row, visited):
            # Normalize path to remove last duplicate start at end
            if path[-1] == start:
                return path[:-1]
            return path
    return []


def apply_stepping_stone(allocation: np.ndarray, cycle: List[Tuple[int, int]]) -> None:
    # cycle 是按顺序的闭合路径（不重复首尾）
    # 正号分配在偶数位（0,2,4...），负号在奇数位
    signs = [1 if idx % 2 == 0 else -1 for idx in range(len(cycle))]
    # 找到所有减号位置的最小分配量
    min_qty = float('inf')
    for idx, (i, j) in enumerate(cycle):
        if signs[idx] == -1:
            min_qty = min(min_qty, allocation[i, j])
    if min_qty == float('inf'):
        return
    # 更新分配
    for idx, (i, j) in enumerate(cycle):
        allocation[i, j] += signs[idx] * min_qty
        if abs(allocation[i, j]) < EPS:
            allocation[i, j] = 0.0


def modi_optimize(allocation: np.ndarray, costs: np.ndarray, supply: np.ndarray, demand: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    m, n = costs.shape
    alloc = allocation.copy().astype(float)
    # 确保基础变量数量为 m+n-1
    basics = get_basic_cells_from_allocation(alloc)
    basics = ensure_non_degenerate(basics, costs, m, n)

    it = 0
    while it < max_iter:
        it += 1
        # 计算机会成本
        deltas = compute_opportunity_costs(alloc, basics, costs)
        # 找到最负的 delta
        min_val = np.nanmin(deltas)
        if np.isnan(min_val) or min_val >= -EPS:
            break  # 已最优
        # 选择最负的格子作为进入基变量
        candidates = [(deltas[i, j], i, j) for i in range(m) for j in range(n) if (i, j) not in basics]
        candidates.sort()
        delta, bi, bj = candidates[0]
        start = (bi, bj)
        # 在 basics U {start} 中找闭环
        cyc = find_cycle(start, basics, m, n)
        if not cyc:
            # 若没找到闭环（理论上不应发生），将该格子也加入基础，避免死循环
            basics.add(start)
            continue
        # cycle 应以 start 开头；如果不是，旋转
        if cyc[0] != start:
            # 旋转到以 start 为首
            try:
                idx = cyc.index(start)
                cyc = cyc[idx:] + cyc[:idx]
            except ValueError:
                pass
        # 确保 cycle 长度为偶数
        if len(cyc) % 2 == 1:
            # 无效闭环
            basics.add(start)
            continue
        # 应用 stepping stone 调整
        apply_stepping_stone(alloc, cyc)
        # 更新基础变量集合：有分配量的为基础
        basics = get_basic_cells_from_allocation(alloc)
        basics = ensure_non_degenerate(basics, costs, m, n)
    return alloc

# ---------------------------- 主程序与 CLI ----------------------------

def compute_total_cost(allocation: np.ndarray, costs: np.ndarray) -> float:
    return float(np.sum(allocation * costs))


def main():
    parser = argparse.ArgumentParser(description='Transportation problem solver (NW, LCM, VAM) + MODI optimize')
    parser.add_argument('--input', '-i', default='data.json', help='input data.json path')
    parser.add_argument('--method', '-m', default=None, choices=['nw', 'lcm', 'vam', 'northwest', 'least', 'min', 'vogel'], help='initial solution method')
    parser.add_argument('--optimize', '-o', action='store_true', help='use MODI (u-v) + stepping-stone to optimize')
    parser.add_argument('--output-csv', default=None, help='write allocation table to CSV')
    parser.add_argument('--print', dest='do_print', action='store_true', help='print allocation and cost')
    args = parser.parse_args()

    if not args.input:
        print('No input specified')
        return
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {args.input}: {e}")
        return

    costs = np.array(data['costs'], dtype=float)
    supply = np.array(data['supply'], dtype=float)
    demand = np.array(data['demand'], dtype=float)
    method = (data.get('method') or args.method or 'vam').lower()

    costs, supply, demand = balance_problem(costs, supply, demand)

    if method in ['nw', 'northwest']:
        init_alloc = solve_northwest_corner(costs, supply, demand)
        algo_name = 'Northwest Corner'
    elif method in ['lcm', 'least', 'min']:
        init_alloc = solve_least_cost(costs, supply, demand)
        algo_name = 'Least Cost Method'
    else:
        init_alloc = solve_vogel(costs, supply, demand)
        algo_name = "Vogel's Approximation Method"

    print(f"Initial method: {algo_name}")
    if args.do_print:
        print("Initial allocation:")
        pretty_print_allocation(init_alloc, costs)
        print(f"Initial total cost: {compute_total_cost(init_alloc, costs):.2f}")

    final_alloc = init_alloc
    if args.optimize:
        print("Running MODI (u-v) optimization...")
        final_alloc = modi_optimize(init_alloc, costs, supply, demand)
        print("Optimization complete.")
        if args.do_print:
            print("Final allocation:")
            pretty_print_allocation(final_alloc, costs)
            print(f"Final total cost: {compute_total_cost(final_alloc, costs):.2f}")

    # 输出 CSV
    if args.output_csv:
        rows, cols = costs.shape
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            header = ["", *[f"B{j+1}" for j in range(cols)], "Supply"]
            w.writerow(header)
            for i in range(rows):
                row = [f"A{i+1}"] + [final_alloc[i, j] for j in range(cols)] + [sum(final_alloc[i, :])]
                w.writerow(row)
            w.writerow(["Demand"] + [sum(final_alloc[:, j]) for j in range(cols)])
        print(f"Wrote allocation to {args.output_csv}")

    if not args.do_print and not args.output_csv:
        # 最基本的输出
        print("Allocation matrix:")
        print(final_alloc)
        print(f"Total cost: {compute_total_cost(final_alloc, costs):.2f}")


if __name__ == '__main__':
    main()
