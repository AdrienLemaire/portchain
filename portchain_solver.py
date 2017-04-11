#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Paint shop coding exercise:

Required lib: numpy

Usage:
    $ chmod +x portchain_solver.py
    $ ./portchain_solver.py -h
    $ ./portchain_solver.py < file.txt
'''


import argparse
import sys
try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    print('Please install the numpy and pandas libraries as follow:\n'
         '\n\t$ pip install numpy pandas')
    exit(0)



# Parse the data

def get_matrix(nb_colors, requests):
    '''
    From input, generate 2 1-hot encoded matrices,
    one to describe wanted matte colors, and one
    to describe wanted colors.

    Args:
        nb_colors (int): Number of colors
        requests  (str): raw input, 1 line per client

    Returns:
        tuple of numpy.ndarray: Matrices
    '''
    requests = list(filter(None, requests.split('\n')))
    matte_wanted = np.zeros((len(requests), nb_colors))
    colors_wanted = np.zeros((len(requests), nb_colors))

    for client, line in enumerate(requests):
        data = line.split()
        if len(data)%2 != 0:
            raise ValueError('No solution exists')

        for i in range(0, len(data), 2):
            try:
                color, sheen = int(data[i]), data[i+1]
            except:
                raise ValueError('No solution exists')

            matte_wanted[client][color-1] = sheen == 'M'
            colors_wanted[client][color-1] = 1

    matte_wanted = pd.DataFrame(matte_wanted, columns=range(nb_colors))
    colors_wanted = pd.DataFrame(colors_wanted, columns=range(nb_colors))
    return matte_wanted, colors_wanted


# Heuristics

def locs_for_unique_requests(colors, matte):
    colors_rows = colors[colors.sum(axis=1) == 1]
    colors_target = colors_rows.loc[:, colors_rows.sum(axis=0) > 0]
    if colors_target.size == 0:
        # No more unique rows to reduce, quit
        return False, False

    matte_target = matte.loc[colors_target.index, colors_target.columns]
    matte_sum = matte_target.sum()

    if (matte_sum[matte_sum != 0] != colors_target.sum()[matte_sum != 0]).any():
        # Unique requests have different sheen requirements, impossible to process
        raise ValueError('No solution exists')

    # Update batch, sheen_chosen and client_satisfied
    locs = np.where(colors_target == 1)
    return colors_target, locs


def locs_for_glossy(colors, matte):
    colors_target = matte[colors == 1].dropna(axis=1)
    if colors_target.size == 0:
        return False, False

    locs = np.where(colors_target == 0)
    return colors_target, locs


def locs_for_remaining(colors, matte):
    colors_target = matte[colors == 1]
    if colors_target.size == 0:
        return False, False
    locs = np.where(colors_target == 1)
    return colors_target, locs


# Main

def get_batch(nb_colors, requests):
    '''
    Calculates optimal batch of colors to satisfy all requests

    Args:
        nb_colors (int): Nb of colors to handle
        requests (str): Raw requests

    Returns:
        list of M/G values
    '''
    # Create matrices
    matte, colors = get_matrix(nb_colors, requests)

    # Create vectors
    batch = np.zeros(nb_colors)
    sheen_chosen = np.zeros(nb_colors)
    client_satisfied = np.zeros(len(matte))

    def compute_iter(f):
        # reduce data matrix to unsatisfied clients and unchosen sheen colors
        colors_reduced = colors.iloc[client_satisfied == 0, sheen_chosen == 0]
        matte_reduced = matte.iloc[client_satisfied == 0, sheen_chosen == 0]

        # Find targets and locations
        colors_target, locs = f(colors_reduced, matte_reduced)
        if colors_target is False:
            return

        for x, y in zip(*[l.tolist() for l in locs]):
            row = colors_target.index[x]     # client id
            col = colors_target.columns[y]   # color id

            # Set color sheen as chosen
            sheen_chosen[col] = 1
            # Update color's sheen
            sheen = matte.loc[row, col]
            batch[col] = sheen
            # Set all customers who requested this color/sheen as satisfied
            mask = (colors[col] == 1) & (matte[col] == sheen)
            for row in colors[mask].index:
                client_satisfied[row] = 1
        compute_iter(f)

    # Compute solution
    for f in [locs_for_unique_requests, locs_for_glossy, locs_for_remaining]:
        if client_satisfied.all():
            break  # solution already found, quit
        compute_iter(f)

    if not client_satisfied.all():
        raise ValueError('No solution exists')
    else:
        return ' '.join(['M' if m else 'G' for m in batch])


# I/O

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Paint shop coding exercise:
        $ ./portchain_solver.py < input.txt''')
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin)
    args = parser.parse_args()

    nb_colors = int(args.infile.readline())
    requests = args.infile.read()
    batch = get_batch(nb_colors, requests)
    print(batch)
