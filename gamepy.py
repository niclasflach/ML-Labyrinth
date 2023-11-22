import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Drawing a Rectangle")

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)

# Set up the rectangle
rectangle_width, rectangle_height = 50, 30
rectangle_x, rectangle_y = 375, 285

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Draw background
    screen.fill(white)

    # Draw the rectangle
    pygame.draw.rect(screen, black, (rectangle_x, rectangle_y, rectangle_width, rectangle_height))

    # Update the display
    pygame.display.flip()

    # Control the frames per second (FPS)
    pygame.time.Clock().tick(60)