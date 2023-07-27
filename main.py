from enum import IntEnum
import random
import numpy as np
from PIL import Image


class Direction(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class Maze:

    def __init__(self, n_rows: int, n_cols: int,
                 start_pos: tuple[int, int] = (0, 0),
                 goal_pos: tuple[int, int] | None = None) -> None:
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._start_pos = start_pos
        self._goal_pos = goal_pos if goal_pos is not None else (
            self._n_rows - 1, self._n_cols - 1)
        self._horizontal_edges = [False for _ in range((n_rows - 1) * n_cols)]
        self._vertical_edges = [False for _ in range((n_cols - 1) * n_rows)]
        self._visited = [False for _ in range(n_rows * n_cols)]

        self._carve_paths()

    def _get_north_edge_idx(self, row: int, col: int):
        assert row > 0 and row < self._n_rows
        assert col >= 0 and col < self._n_cols
        edge_idx = self._n_cols * (row - 1) + col
        return edge_idx

    def _get_south_edge_idx(self, row: int, col: int):
        assert row >= 0 and row < self._n_rows - 1
        assert col >= 0 and col < self._n_cols
        edge_idx = self._n_cols * row + col
        return edge_idx

    def _get_east_edge_idx(self, row: int, col: int):
        assert row >= 0 and row < self._n_cols
        assert col >= 0 and col < self._n_cols - 1
        edge_idx = self._n_cols * row - row + col
        return edge_idx

    def _get_west_edge_idx(self, row: int, col: int):
        assert row >= 0 and row < self._n_cols
        assert col > 0 and col < self._n_cols
        edge_idx = self._n_cols * row - row + (col - 1)
        return edge_idx

    def _get_edge_idx(self, row: int, col: int, direction: Direction):
        match direction:
            case Direction.NORTH:
                return self._get_north_edge_idx(row, col)
            case Direction.SOUTH:
                return self._get_south_edge_idx(row, col)
            case Direction.EAST:
                return self._get_east_edge_idx(row, col)
            case Direction.WEST:
                return self._get_west_edge_idx(row, col)
            case _:
                raise ValueError()

    def _get_edge_at(self, row: int, col: int, direction: Direction):
        edge_idx = self._get_edge_idx(row, col, direction)
        if direction == Direction.NORTH or direction == Direction.SOUTH:
            return self._horizontal_edges[edge_idx]
        elif direction == Direction.EAST or direction == Direction.WEST:
            return self._vertical_edges[edge_idx]

    def _set_edge_at(self, row: int, col: int, direction: Direction, val: bool):
        edge_idx = self._get_edge_idx(row, col, direction)
        if direction == Direction.NORTH or direction == Direction.SOUTH:
            self._horizontal_edges[edge_idx] = val
        elif direction == Direction.EAST or direction == Direction.WEST:
            self._vertical_edges[edge_idx] = val

    def _has_non_visited_neighbours(self, row: int, col: int):
        c = 0
        if row > 0 and not self._visited[self._get_cell_index(row - 1, col)]:
            c += 1
        if row < self._n_rows - 1 and not self._visited[self._get_cell_index(row + 1, col)]:
            c += 1
        if col > 0 and not self._visited[self._get_cell_index(row, col - 1)]:
            c += 1
        if col < self._n_cols - 1 and not self._visited[self._get_cell_index(row, col + 1)]:
            c += 1
        return c

    def _get_cell_index(self, row: int, col: int):
        assert row >= 0 and row < self._n_rows
        assert col >= 0 and col < self._n_cols
        return row * self._n_cols + col

    def _get_cell_at(self, row: int, col: int, d: Direction):
        assert row >= 0 and row < self._n_rows
        assert col >= 0 and col < self._n_cols
        match d:
            case Direction.NORTH:
                return (row - 1, col)
            case Direction.SOUTH:
                return (row + 1, col)
            case Direction.EAST:
                return (row, col + 1)
            case Direction.WEST:
                return (row, col - 1)
            case _:
                raise ValueError()

    def _get_cell_coords(self, cell_idx: int):
        assert cell_idx >= 0 and cell_idx < self._n_cols * self._n_rows
        c_idx = cell_idx % self._n_rows
        r_idx = cell_idx // self._n_rows
        return (r_idx, c_idx)

    def _carve_path_recur(self, node: tuple[int, int]):
        node_idx = self._get_cell_index(*node)
        self._visited[node_idx] = True
        directions = list(Direction)
        random.shuffle(directions)
        for d in directions:
            next_node = self._get_cell_at(*node, d)
            if next_node[0] < 0 or next_node[0] >= self._n_rows:
                continue
            if next_node[1] < 0 or next_node[1] >= self._n_cols:
                continue
            next_node_idx = self._get_cell_index(*next_node)
            if self._visited[next_node_idx]:
                continue
            self._set_edge_at(*node, d, True)
            self._carve_path_recur(next_node)

    def _carve_path_iter(self, node: tuple[int, int]):
        visiting_nodes = [node]
        self._visited[self._get_cell_index(*node)] = True

        while len(visiting_nodes) > 0:
            node = visiting_nodes.pop(-1)

            if self._has_non_visited_neighbours(*node):
                visiting_nodes.append(node)

            directions = list(Direction)
            random.shuffle(directions)
            for d in directions:
                next_node = self._get_cell_at(*node, d)
                if next_node[0] < 0 or next_node[0] >= self._n_rows:
                    continue
                if next_node[1] < 0 or next_node[1] >= self._n_cols:
                    continue
                next_node_idx = self._get_cell_index(*next_node)
                if self._visited[next_node_idx]:
                    continue
                self._set_edge_at(*node, d, True)
                self._visited[next_node_idx] = True
                visiting_nodes.append(next_node)

    def _carve_paths(self):

        self._carve_path_iter(self._start_pos)

    def __repr__(self) -> str:
        ret = ""

        ret += "+" + "-" * (self._n_cols * 2 - 1) + "+\n"

        for r_idx in range(self._n_rows):

            # do a line for the top edge
            if r_idx > 0:
                ret += "|"
                edge_start_idx = self._n_cols * (r_idx - 1)
                for e_idx in range(self._n_cols):
                    if self._horizontal_edges[edge_start_idx + e_idx]:
                        ret += " "
                    else:
                        ret += "-"
                    if e_idx < self._n_cols - 1:
                        ret += "+"
                ret += "|\n"

            ret += "|"

            # print cell line
            edge_start_idx = self._n_cols * r_idx - r_idx
            for c_idx in range(self._n_cols):

                # check for left edge
                if c_idx > 0:
                    if self._vertical_edges[edge_start_idx + c_idx - 1]:
                        ret += " "
                    else:
                        ret += "|"

                if self._start_pos == (r_idx, c_idx):
                    ret += "S"
                elif self._goal_pos == (r_idx, c_idx):
                    ret += "G"
                else:
                    ret += " "

            ret += "|\n"

        ret += "+" + "-" * (self._n_cols * 2 - 1) + "+"

        return ret

    def make_binary_map(self):
        rows = self._n_rows + (self._n_rows - 1)
        cols = self._n_cols + (self._n_cols - 1)
        ret = np.zeros((rows, cols), dtype=np.int8)

        for r in range(rows):
            if r % 2 == 0:
                # cell row
                r_idx = r // 2
                edge_start_idx = self._n_cols * r_idx - r_idx
                for c_idx in range(self._n_cols):
                    if c_idx > 0:
                        ret[r][2 * c_idx - 1] = 1 * \
                            self._vertical_edges[edge_start_idx + c_idx - 1]
                    ret[2 * r_idx][2 * c_idx] = 1
            else:
                r_idx = (r + 1) // 2
                if r_idx > 0:
                    for c_idx in range(self._n_cols):
                        edge_start_idx = self._n_cols * (r_idx - 1)
                        ret[r][2 * c_idx] = 1 * \
                            self._horizontal_edges[edge_start_idx + c_idx]

        return ret

    def to_image(self, size: tuple[int, int] = (1024, 1024), resample=Image.Resampling.BOX):
        img = Image.fromarray(255 * self.make_binary_map())
        img = img.resize(size, resample=resample)
        return img

    def move(self, pos: tuple[int, int], d: Direction) -> (bool, tuple[int, int]):
        if self._get_edge_at(*pos, d) is False:
            return False, pos
        next_cell = self._get_cell_at(*pos, d)
        return True, next_cell


maze = Maze(10, 10)
# maze.to_image().show()

# print(maze)
# pos = (0, 0)
# can = True
# while can:
#     (can, pos) = maze.move(pos, Direction.EAST)
#     print(can, pos)
