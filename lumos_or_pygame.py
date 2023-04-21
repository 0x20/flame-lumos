import pygame

try:
    import lumos
except:
    print("noLumos")
import time

pygame.init()


def matrix_to_frame(self, matrix):
    frame = ''
    for row in matrix:
        for val in row:
            r, g, b = val
            frame += f'{r:02x}{g:02x}{b:02x}'
    return frame


class Display:
    def __init__(self, destination, width, height, scale=5, caption='Set My Name', fps=30):
        self.destination = destination
        self.width = width
        self.height = height
        self.scale = scale
        self.fps = fps

        if self.destination == 'pygame':
            self.screen = pygame.display.set_mode(
                (width * scale, height * scale))
            pygame.display.set_caption(caption)
            self.clock = pygame.time.Clock()

    def push(self, color_matrix):
        if self.destination == 'pygame':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
            for y, row in enumerate(color_matrix):
                for x, val in enumerate(row):
                    pygame.draw.rect(self.screen, val, pygame.Rect(
                        x * self.scale, y * self.scale, self.scale, self.scale))

            pygame.display.flip()
            self.clock.tick(self.fps)  # 30 FPS

        elif self.destination == 'lumos':
            frame = self.matrix_to_frame(color_matrix)
            lumos.push(frame)
            time.sleep(1.0/self.fps)
        return True