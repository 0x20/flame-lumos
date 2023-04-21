import numpy as np
import time
import argparse
import lumos_or_pygame


def generate_flame_matrix(height, width):
    flame_matrix = np.zeros((height, width), dtype=int)
    flame_matrix[-1, :] = np.random.randint(160, 256, width)
    return flame_matrix


def flame_matrix_to_color_matrix(flame_matrix):
    height, width = flame_matrix.shape
    color_matrix = np.zeros((height, width, 3), dtype=int)

    for y in range(height):
        for x in range(width):
            val = flame_matrix[y, x]
            r = min(val, 255)
            g = min(val // 2, 127)
            b = 0
            color_matrix[y, x] = (r, g, b)

    return color_matrix


def update_flame_matrix(flame_matrix):
    height, width = flame_matrix.shape
    padded_matrix = np.pad(flame_matrix, ((0, 0), (1, 1)),
                           'constant', constant_values=0)

    for y in range(height - 1, 0, -1):
        rolled_left = np.roll(padded_matrix[y - 1:y + 1, :], -1, axis=1)
        rolled_right = np.roll(padded_matrix[y - 1:y + 1, :], 1, axis=1)

        neighborhood = np.concatenate(
            (flame_matrix[y - 1:y + 1, :], rolled_left[:, :-2], rolled_right[:, 2:]), axis=1)
        total = np.sum(neighborhood, axis=0)

        flame_matrix[y - 1, :] = np.maximum(
            0, (total[0:width] // 2) - np.random.randint(0, 14, width))




def main(display_method):
    height, width, scale = 10, 120, 15
    screen = ''
    display = lumos_or_pygame.Display(display_method, width, height, scale, 'fire')

    flame_matrix = generate_flame_matrix(height, width)

    running = True
    while running:
        update_flame_matrix(flame_matrix)
        color_matrix = flame_matrix_to_color_matrix(flame_matrix)
        running = display.push(color_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', choices=['pygame', 'lumos'], default='pygame',
                        help='Choose the display method: pygame or lumos')
    args = parser.parse_args()

    main(args.display)
