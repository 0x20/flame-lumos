import numpy as np
import time
import pygame
import argparse
try:
    import lumos
except:
    print("noLumos")


def generate_flame_matrix(height, width):
    flame_matrix = np.zeros((height, width), dtype=int)
    flame_matrix[-1, :] = np.random.randint(160, 256, width)
    return flame_matrix


def flame_matrix_to_frame(flame_matrix):
    frame = ''
    for row in flame_matrix:
        for val in row:
            r = min(val, 255)
            g = min(val // 2, 127)
            b = 0
            frame += f'{r:02x}{g:02x}{b:02x}'
    return frame


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
    # assuming lumos.display() accepts a 2D list of RGB tuples
    lumos.display(pixel_matrix)


def main(display_method):
    height, width, scale = 10, 120, 15
    screen = ''

    if display_method == 'pygame':
        # Initialize pygame
        pygame.init()
        screen = pygame.display.set_mode((width * scale, height * scale))
        pygame.display.set_caption('Flame Animation')
        clock = pygame.time.Clock()

    flame_matrix = generate_flame_matrix(height, width)

    running = True
    while running:
        if display_method == 'pygame':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        update_flame_matrix(flame_matrix)
        frame = flame_matrix_to_frame(flame_matrix)

        if display_method == 'pygame':
            screen.fill((0, 0, 0))
            show_frame_pygame(frame, width, height, scale, screen)
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
        elif display_method == 'lumos':
            show_frame_lumos(frame, width, height)
            time.sleep(1/30)  # 30 FPS

    if display_method == 'pygame':
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', choices=['pygame', 'lumos'], default='pygame',
                        help='Choose the display method: pygame or lumos')
    args = parser.parse_args()

    main(args.display)
