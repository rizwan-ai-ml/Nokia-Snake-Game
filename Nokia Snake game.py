import cv2
import mediapipe as mp
import pygame
import sys
import time
import numpy as np

# ---------- Settings ----------
CAM_WIDTH, CAM_HEIGHT = 320, 240
GAME_WIDTH, GAME_HEIGHT = 600, 600
CELL_SIZE = 20
FPS = 18  # faster snake

# Zones for gesture direction
LEFT_ZONE = 0.25
RIGHT_ZONE = 0.75
TOP_ZONE = 0.25
BOTTOM_ZONE = 0.75

# ---------- MediaPipe setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ---------- Pygame setup ----------
pygame.init()
game_display = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
pygame.display.set_caption("Gesture Controlled Snake")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

def draw_text(surface, text, pos, color=(255,255,255)):
    img = font.render(text, True, color)
    surface.blit(img, pos)

def get_direction_from_point(x_norm, y_norm):
    """Determine direction from normalized fingertip position"""
    if x_norm < LEFT_ZONE:
        return 'LEFT'
    if x_norm > RIGHT_ZONE:
        return 'RIGHT'
    if y_norm < TOP_ZONE:
        return 'UP'
    if y_norm > BOTTOM_ZONE:
        return 'DOWN'
    return None

# ---------- Snake Game ----------
class Snake:
    def __init__(self):
        self.positions = [(GAME_WIDTH//2, GAME_HEIGHT//2)]
        self.direction = 'RIGHT'
        self.grow_pending = 0

    def head(self):
        return self.positions[0]

    def move(self):
        x, y = self.head()
        if self.direction == 'UP':
            y -= CELL_SIZE
        elif self.direction == 'DOWN':
            y += CELL_SIZE
        elif self.direction == 'LEFT':
            x -= CELL_SIZE
        elif self.direction == 'RIGHT':
            x += CELL_SIZE
        new_head = (x % GAME_WIDTH, y % GAME_HEIGHT)
        self.positions.insert(0, new_head)
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            self.positions.pop()

    def change_direction(self, new_dir):
        opposites = {'UP':'DOWN', 'DOWN':'UP', 'LEFT':'RIGHT', 'RIGHT':'LEFT'}
        if new_dir and opposites.get(new_dir) != self.direction:
            self.direction = new_dir

    def grow(self):
        self.grow_pending += 3

    def collides_with_self(self):
        return self.head() in self.positions[1:]

class Food:
    def __init__(self):
        self.pos = self.random_pos()

    def random_pos(self):
        cols = GAME_WIDTH // CELL_SIZE
        rows = GAME_HEIGHT // CELL_SIZE
        return (np.random.randint(0, cols)*CELL_SIZE,
                np.random.randint(0, rows)*CELL_SIZE)

    def respawn(self):
        self.pos = self.random_pos()

# ---------- Main ----------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    snake = Snake()
    food = Food()
    last_direction = None
    prev_points = []

    running = True
    paused = False

    while running:
        # --- Handle pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_p:
                    paused = not paused

        # --- Read camera ---
        ret, frame = cap.read()
        if not ret:
            print("Camera not found.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        fingertip_norm = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark[8]
            fingertip_norm = (lm.x, lm.y)
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        game_direction = None
        if fingertip_norm is not None:
            fx, fy = fingertip_norm

            # Smooth fingertip with moving average (3 frames)
            prev_points.append((fx, fy))
            if len(prev_points) > 3:
                prev_points.pop(0)
            fx = np.mean([p[0] for p in prev_points])
            fy = np.mean([p[1] for p in prev_points])

            game_direction = get_direction_from_point(fx, fy)
            if game_direction:
                snake.change_direction(game_direction)
                last_direction = game_direction

        # --- Update Game ---
        if not paused:
            snake.move()
            if snake.head() == food.pos:
                snake.grow()
                food.respawn()
            if snake.collides_with_self():
                snake = Snake()
                food = Food()

        # --- Draw Game ---
        game_display.fill((0,0,0))
        for x in range(0, GAME_WIDTH, CELL_SIZE):
            pygame.draw.line(game_display, (40,40,40), (x,0), (x,GAME_HEIGHT))
        for y in range(0, GAME_HEIGHT, CELL_SIZE):
            pygame.draw.line(game_display, (40,40,40), (0,y), (GAME_WIDTH,y))

        pygame.draw.rect(game_display, (200,0,0), (*food.pos, CELL_SIZE, CELL_SIZE))
        for i, pos in enumerate(snake.positions):
            color = (0,200,0) if i==0 else (0,150,0)
            pygame.draw.rect(game_display, color, (*pos, CELL_SIZE, CELL_SIZE))

        draw_text(game_display, f"Score: {len(snake.positions)-1}", (10,10))
        draw_text(game_display, "Press P=Pause | ESC=Exit", (10,35))
        pygame.display.flip()
        clock.tick(FPS)

        # --- Camera feedback ---
        if fingertip_norm is not None:
            cx = int(fingertip_norm[0]*CAM_WIDTH)
            cy = int(fingertip_norm[1]*CAM_HEIGHT)
            cv2.circle(frame, (cx, cy), 8, (0,255,0), -1)
            cv2.putText(frame, f"{last_direction or 'None'}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        small = cv2.resize(frame, (320,240))
        cv2.imshow("Camera (press Q to close)", small)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
