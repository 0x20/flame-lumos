import pygame

try:
    import lumos
except:
    print("noLumos")
import time



class Display:
    def __init__(self, destination, width, height, scale=5, caption='Set My Name', fps=30):
        self.destination = destination
        self.width = width
        self.height = height
        self.scale = scale
        self.fps = fps
        self.hex_list = [''] * (width * height)

        if self.destination == 'pygame':
            pygame.init()
            self.screen = pygame.display.set_mode(
                (width * scale, height * scale))
            pygame.display.set_caption(caption)
            self.clock = pygame.time.Clock()

    def matrix_to_frame(self, matrix):
        hex_list = self.hex_list
        idx = 0
        for row in matrix:
            for val in row:
                r, g, b = val
                hex_list[idx] = f'{r:02x}{g:02x}{b:02x}'
                idx += 1
        return ''.join(hex_list)

    def push(self, color_matrix):
        if self.destination == 'pygame':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
            self.screen.fill((0, 0, 0))

            for y, row in enumerate(color_matrix):
                for x, val in enumerate(row):
                    r, g, b = val
                    # Clip color values to the valid range
                    r = max(0, min(r, 255))
                    g = max(0, min(g, 255))
                    b = max(0, min(b, 255))

                    pygame.draw.rect(self.screen, (r, g, b), pygame.Rect(
                        x * self.scale, y * self.scale, self.scale, self.scale))


            pygame.display.flip()
            self.clock.tick(self.fps)  # 30 FPS

        elif self.destination == 'lumos':
            frame = self.matrix_to_frame(color_matrix)
            lumos.push(frame)
            time.sleep(1.0/self.fps)
        return True
