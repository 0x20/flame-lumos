
from __future__ import division
import numpy as np
import time
import pygame
import argparse
import lumos


def generate_game_of_life_matrix(height, width):
    # Start with an empty matrix
    matrix = np.zeros((height, width), dtype=int)

    # Define a glider
    glider = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.float64)

    # Define a Beacon
    beacon = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1]
    ], dtype=np.float64)

    # Define a Toad
    toad = np.array([
        [0, 1, 1, 1],
        [1, 1, 1, 0]
    ], dtype=np.float64)

    # Define a Lightweight spaceship
    lw_spaceship = np.array([
        [1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 1]
    ], dtype=np.float64)

    # Place multiple gliders in the matrix
    matrix[0:3, 0:3] = glider
    matrix[0:3, 10:13] = glider

    # Place Beacon
    matrix[3:7, 20:24] = beacon

    # Place Toad
    matrix[3:5, 40:44] = toad

    # Place Lightweight spaceship
    matrix[6:10, 60:65] = lw_spaceship

    return matrix


# def generate_game_of_life_matrix(height, width):
#     # Start with an empty matrix
#     matrix = np.zeros((height, width), dtype=int)

#     # Define a glider
#     glider = np.array([
#         [0, 1, 0],
#         [0, 0, 1],
#         [1, 1, 1]
#     ], dtype=np.float64)

#     # Define a Beacon
#     beacon = np.array([
#         [1, 1, 0, 0],
#         [1, 1, 0, 0],
#         [0, 0, 1, 1],
#         [0, 0, 1, 1]
#     ], dtype=np.float64)

#     # Place multiple gliders in the matrix
#     matrix[0:3, 0:3] = glider
#     matrix[0:3, 10:13] = glider
#     matrix[0:3, 20:23] = glider
#     matrix[0:3, 30:33] = glider
#     matrix[0:3, 40:43] = glider
#     matrix[0:3, 50:53] = glider

#     # Place Beacon at the right end
#     matrix[6:10, 116:120] = beacon

#     return matrix


# def generate_game_of_life_matrix(height, width):
#     # Start with an empty matrix
#     matrix = np.zeros((height, width), dtype=int)

#     # Define a glider
#     glider = np.array([
#         [0, 1, 0],
#         [0, 0, 1],
#         [1, 1, 1]
#     ], dtype=np.float64)

#     # Place multiple gliders in the matrix
#     matrix[0:3, 0:3] = glider
#     matrix[0:3, 10:13] = glider
#     matrix[0:3, 20:23] = glider
#     matrix[0:3, 30:33] = glider
#     matrix[0:3, 40:43] = glider
#     matrix[0:3, 50:53] = glider
#     matrix[0:3, 60:63] = glider
#     matrix[0:3, 70:73] = glider
#     matrix[0:3, 80:83] = glider
#     matrix[0:3, 90:93] = glider
#     matrix[0:3, 100:103] = glider
#     matrix[0:3, 110:113] = glider

    return matrix

# def generate_game_of_life_matrix(height, width):
#     return np.random.choice([0, 1], size=(height, width), p=[0.5, 0.5])


def game_of_life_matrix_to_frame(matrix, gradient):
    frame = ''
    for y, row in enumerate(matrix):
        for x, val in enumerate(row):
            if val:
                r, g, b = tuple(int(c * 255) for c in gradient[y, x])
            else:
                r = g = b = 0
            frame += '{:02x}{:02x}{:02x}'.format(r, g, b)
    return frame


def generate_random_color_with_min_distance(reference_color, min_distance):
    new_color = np.random.uniform(size=3)
    while np.linalg.norm(new_color - reference_color) < min_distance:
        new_color = np.random.uniform(size=3)
    return new_color


# def update_game_of_life_matrix(matrix):
#     neighbors = np.zeros(matrix.shape, dtype=int)
#     for y in range(-1, 2):
#         for x in range(-1, 2):
#             if y != 0 or x != 0:
#                 neighbors += np.roll(np.roll(matrix, y, axis=0), x, axis=1)
#     matrix = (neighbors == 3) | (matrix & (neighbors == 2))
#     return matrix


def update_game_of_life_matrix(matrix):
    # Compute the sum of the neighbors using convolution
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbors = scipy.signal.convolve2d(
        matrix, kernel, mode='same', boundary='wrap')

    # Apply the rules of Game of Life
    matrix = np.where((neighbors == 3) | (
        (matrix == 1) & (neighbors == 2)), 1, 0)

    return matrix


def show_frame_pygame(frame, width, height, scale, screen):
    frame_list = [frame[i:i+6] for i in range(0, len(frame), 6)]
    pixel_matrix = [frame_list[i * width:(i + 1) * width]
                    for i in range(height)]

    for y, row in enumerate(pixel_matrix):
        for x, pixel in enumerate(row):
            color = tuple(int(pixel[i:i+2], 16) for i in (0, 2, 4))
            pygame.draw.rect(screen, color, pygame.Rect(
                x * scale, y * scale, scale, scale))


def show_frame_lumos(frame, width, height):
    frame_list = [frame[i:i+6] for i in range(0, len(frame), 6)]
    pixel_matrix = [frame_list[i * width:(i + 1) * width]
                    for i in range(height)]
    # Convert RGB hex strings back to a single string
    frame_hex = ''.join([pixel for row in pixel_matrix for pixel in row])
    lumos.push(frame_hex)


def rgb_to_hsv(r, g, b):
    max_value = float(max(r, g, b))
    min_value = float(min(r, g, b))
    difference = max_value - min_value
    # print("Max value: ", max_value)
    # print("Min value: ", min_value)
    # print("Difference: ", difference)

    if max_value == min_value:
        h = 0
    elif max_value == r:
        h = (60 * ((g - b) / difference) + 360) % 360
    elif max_value == g:
        h = (60 * ((b - r) / difference) + 120) % 360
    elif max_value == b:
        h = (60 * ((r - g) / difference) + 240) % 360
    # print("Hue: ", h)

    if max_value == 0:
        s = 0
    else:
        s = (difference / max_value)
    # print("Saturation: ", s)

    v = max_value / 255.0
    # print("Value: ", v)

    return h / 360.0, s, v


def hsv_to_rgb(h, s, v):
    # v = max(0.1,v)
    h = h * 360.0
    hi = int(h / 60.0) % 6
    f = h / 60.0 - hi
    # print("Hue (degrees): ", h)
    # print("Hue interval: ", hi)
    # print("Fractional part: ", f)

    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    # print("P: ", p)
    # print("Q: ", q)
    # print("T: ", t)

    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q

    # print("RGB: ", int(r * 255), int(g * 255), int(b * 255))

    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def main(display_method):
    height, width, scale = 20, 240, 5  # Increase the grid size
    display_height, display_width = 10, 120  # Define the size of the display
    screen = ''

    color1 = np.random.uniform(size=3)
    min_distance = 0.5
    color2 = generate_random_color_with_min_distance(color1, min_distance)

    if display_method == 'pygame':
        # Initialize pygame
        pygame.init()
        # Update this to use the display size
        screen = pygame.display.set_mode(
            (display_width * scale, display_height * scale))
        pygame.display.set_caption('Game of Life')
        clock = pygame.time.Clock()

    flame_matrix = generate_game_of_life_matrix(height, width)

    running = True
    fc = 999
    while running:
        if display_method == 'pygame':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        fc += 1

        # Shift hue for the gradient colors
        color1_hsv = rgb_to_hsv(*color1)
        # print("before ", color1_hsv[0])
        # print("increment ", color1_hsv[0], color1_hsv[0] +
        #       float(fc) / 1000.0, (color1_hsv[0] + float(fc) / 1000.0) % 1)
        # print("after ",  (color1_hsv[0] + float(fc) / 1000.0) % 1)
        color1_hsv = ((color1_hsv[0] + 1 / 10.0) %
                      1, color1_hsv[1], color1_hsv[2])
        color1 = np.array(hsv_to_rgb(*color1_hsv))
        color2_hsv = rgb_to_hsv(*color2)
        color2_hsv = ((color2_hsv[0] + float(fc) / 1000.0) %
                      1, color2_hsv[1], color2_hsv[2])
        color2 = np.array(hsv_to_rgb(*color2_hsv))

        gradient = np.array([[color1 + (color2 - color1) * (float(x + y) /
                            (width + height - 2)) for x in range(width)] for y in range(height)])
        flame_matrix = update_game_of_life_matrix(flame_matrix)
        # Update this to use only the display portion of the matrix and gradient
        frame = game_of_life_matrix_to_frame(
            flame_matrix[:display_height, :display_width], gradient[:display_height, :display_width])

        if display_method == 'pygame':
            screen.fill((0, 0, 0))
            # Update this to use the display size
            show_frame_pygame(frame, display_width,
                              display_height, scale, screen)
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
        elif display_method == 'lumos':
            # Update this to use the display size
            show_frame_lumos(frame, display_width, display_height)
            # time.sleep(1/40)  # 30 FPS

    if display_method == 'pygame':
        pygame.quit()


# def main(display_method):
#     height, width, scale=10, 120, 5
#     screen=''

#     color1=np.random.uniform(size=3)
#     min_distance=0.5
#     color2=generate_random_color_with_min_distance(color1, min_distance)

#     if display_method == 'pygame':
#         # Initialize pygame
#         pygame.init()
#         screen=pygame.display.set_mode((width * scale, height * scale))
#         pygame.display.set_caption('Flame Animation')
#         clock=pygame.time.Clock()

#     flame_matrix=generate_game_of_life_matrix(height, width)

#     running=True
#     fc=999
#     while running:
#         if display_method == 'pygame':
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running=False

#         fc += 1

#         # print fc

#         # running = False
#         # Shift hue for the gradient colors
#         color1_hsv=rgb_to_hsv(*color1)
#         color1_hsv=((color1_hsv[0] + float(fc) / 1000.0) %
#                       1, color1_hsv[1], color1_hsv[2])
#         color1=np.array(hsv_to_rgb(*color1_hsv))
#         color2_hsv=rgb_to_hsv(*color2)
#         color2_hsv=((color2_hsv[0] + float(fc) / 1000.0) %
#                       1, color2_hsv[1], color2_hsv[2])
#         color2=np.array(hsv_to_rgb(*color2_hsv))

#         gradient=np.array([[color1 + (color2 - color1) * (float(x + y) /
#                             (width + height - 2)) for x in range(width)] for y in range(height)])
#         flame_matrix=update_game_of_life_matrix(flame_matrix)
#         frame=game_of_life_matrix_to_frame(flame_matrix, gradient)

#         if display_method == 'pygame':
#             screen.fill((0, 0, 0))
#             show_frame_pygame(frame, width, height, scale, screen)
#             pygame.display.flip()
#             clock.tick(30)  # 30 FPS
#         elif display_method == 'lumos':
#             lumos.push(frame)
#             # time.sleep(1/40)  # 30 FPS

#     if display_method == 'pygame':
#         pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', choices=['pygame', 'lumos'], default='pygame',
                        help='Choose the display method: pygame or lumos')
    args = parser.parse_args()

    main(args.display)
