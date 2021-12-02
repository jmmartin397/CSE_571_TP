import os
import numpy as np
from collections.abc import Generator
import random
from layout import Layout

LAYOUT_PATH_TEMPLATE = 'layouts/{}.lay'

ELEMENT_TO_INT = {
    ' ': 0,
    '%': 1,
    '.': 2,
    'o': 3,
    'P': 4,
    'G': 5,
}
INT_TO_ELEMENT = {i:e for e,i in ELEMENT_TO_INT.items()}


class LayoutGen(Generator):
    def __init__(self, base_layout_name, num_ghosts=None, num_capsules=None,
                                         num_food=None, wall_removal_precentage=10):
        file_path = LAYOUT_PATH_TEMPLATE.format(base_layout_name)
        if not os.path.exists(file_path):
            print('Error: layout file at {} does not exist'.format(file_path))
            return
        
        base_layout = []
        with open(file_path) as file:
            for line in file.readlines():
                line = line.strip()
                row = [ELEMENT_TO_INT[c] for c in line]
                base_layout.append(row)
        base_layout = np.array(base_layout)

        if num_ghosts is None:
            self.num_ghosts = len(base_layout[base_layout == ELEMENT_TO_INT['G']])
        else:
            self.num_ghosts = num_ghosts
        
        if num_capsules is None:
            self.num_capsules = len(base_layout[base_layout == ELEMENT_TO_INT['o']])
        else:
            self.num_capsules = num_capsules
        
        if num_food is None:
            self.num_food = len(base_layout[base_layout == ELEMENT_TO_INT['.']])
        else:
            self.num_food = num_food

        self.wall_removal_precentage = wall_removal_precentage / 100

        base_layout[base_layout == ELEMENT_TO_INT['.']] = 0
        base_layout[base_layout == ELEMENT_TO_INT['o']] = 0
        base_layout[base_layout == ELEMENT_TO_INT['G']] = 0
        base_layout[base_layout == ELEMENT_TO_INT['P']] = 0
        self.base_layout = base_layout

    def send(self, ignored_val):
        layout = self.base_layout.copy()
        # remove walls
        height, width = layout.shape
        inner_layout = layout[1:height-1, 1:width-1]
        rows, cols = np.where(inner_layout == ELEMENT_TO_INT['%'])
        for point in zip(rows, cols):
            if random.random() < self.wall_removal_precentage:
                inner_layout[point] = ELEMENT_TO_INT[' ']
        layout[1:height-1, 1:width-1] = inner_layout
        # add pacman to random open position
        rows, cols = np.where(layout == ELEMENT_TO_INT[' '])
        open_positions = list(zip(rows, cols))
        point = random.choice(open_positions)
        open_positions.remove(point)
        layout[point] = ELEMENT_TO_INT['P']
        # add ghosts to random open positions
        for _ in range(self.num_ghosts):
            point = random.choice(open_positions)
            open_positions.remove(point)
            layout[point] = ELEMENT_TO_INT['G']
        # add capsules to random open positions
        for _ in range(self.num_capsules):
            point = random.choice(open_positions)
            open_positions.remove(point)
            layout[point] = ELEMENT_TO_INT['o']
        # add food to random open positions
        for _ in range(self.num_food):
            point = random.choice(open_positions)
            open_positions.remove(point)
            layout[point] = ELEMENT_TO_INT['.']
        # convert array back to string
        lines = []
        for row in layout:
            lines.append(''.join([INT_TO_ELEMENT[i] for i in row]))
        return Layout(lines)

    #need to implement abstract method
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration


class LayoutGenMultipleBaseLayouts(Generator):
    def __init__(self, base_layout_names, **kwargs):
        self.layoutGens = [LayoutGen(name, **kwargs) for name in base_layout_names]

    def send(self, ignored_val):
        layoutGen = random.choice(self.layoutGens)
        return next(layoutGen)

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration


if __name__ == '__main__':
    names = [
        'mediumClassic',
        'smallClassic',
    ]
    gen = LayoutGenMultipleBaseLayouts(names, num_food=50, num_capsules=2, wall_removal_precentage=20)
    print(next(gen))