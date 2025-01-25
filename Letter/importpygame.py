import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# Глобальные переменные для трехмерных преобразований
scale_factor = 1.0
translation = [0.0, 0.0, 0.0]
rotation_angle = 0.0
rotation_axis = [0.0, 0.0, 1.0]

def draw_axes():
    glBegin(GL_LINES)
    glColor3f(1.0, 0.0, 0.0)  # Красный цвет для оси X
    glVertex3f(-5.0, 0.0, 0.0)
    glVertex3f(5.0, 0.0, 0.0)

    glColor3f(0.0, 1.0, 0.0)  # Зеленый цвет для оси Y
    glVertex3f(0.0, -5.0, 0.0)
    glVertex3f(0.0, 5.0, 0.0)

    glColor3f(0.0, 0.0, 1.0)  # Синий цвет для оси Z
    glVertex3f(0.0, 0.0, -5.0)
    glVertex3f(0.0, 0.0, 5.0)
    glEnd()

def draw_letter_я():
    glBegin(GL_QUADS)
    # Передняя вертикальная линия
    glVertex3f(0, -0.8, 0)
    glVertex3f(0, 1, 0)
    glVertex3f(-0.2, 1, 0)
    glVertex3f(-0.2, -0.8, 0)

    # Задняя вертикальная линия
    glVertex3f(0, -0.8, -0.2)
    glVertex3f(0, 1, -0.2)
    glVertex3f(-0.2, 1, -0.2)
    glVertex3f(-0.2, -0.8, -0.2)

    # Верхняя грань
    glVertex3f(0, 1, 0)
    glVertex3f(-0.2, 1, 0)
    glVertex3f(-0.2, 1, -0.2)
    glVertex3f(0, 1, -0.2)

    # Нижняя грань
    glVertex3f(0, -0.8, 0)
    glVertex3f(-0.2, -0.8, 0)
    glVertex3f(-0.2, -0.8, -0.2)
    glVertex3f(0, -0.8, -0.2)

    # Левая грань
    glVertex3f(-0.2, -0.8, 0)
    glVertex3f(-0.2, 1, 0)
    glVertex3f(-0.2, 1, -0.2)
    glVertex3f(-0.2, -0.8, -0.2)

    # Правая грань
    glVertex3f(0, -0.8, 0)
    glVertex3f(0, 1, 0)
    glVertex3f(0, 1, -0.2)
    glVertex3f(0, -0.8, -0.2)
    glEnd()
    
    glBegin(GL_QUADS)
    # Диагональная линия
    # Передняя вертикальная линия
    glVertex3f(0, 0.5, 0)
    glVertex3f(0, 0.3, 0)
    glVertex3f(-0.5, -0.8, 0)
    glVertex3f(-0.8, -0.8, 0)

    # Задняя вертикальная линия
    glVertex3f(0, 0.5, -0.2)
    glVertex3f(0, 0.3, -0.2)
    glVertex3f(-0.5, -0.8, -0.2)
    glVertex3f(-0.8, -0.8, -0.2)

    # Нижняя грань
    glVertex3f(-0.5, -0.8, 0)
    glVertex3f(-0.8, -0.8, 0)
    glVertex3f(-0.8, -0.8, -0.2)
    glVertex3f(-0.5, -0.8, -0.2)

    # Левая грань
    glVertex3f(-0.5, -0.8, 0)
    glVertex3f(-0.5, -0.8, -0.2)
    glVertex3f(0, 0.3, -0.2)
    glVertex3f(0, 0.3, 0)

    # Правая грань
    glVertex3f(0, -0.8, 0)
    glVertex3f(0, 1, 0)
    glVertex3f(0, 1, -0.2)
    glVertex3f(0, -0.8, -0.2)
    glEnd()

    glBegin(GL_QUADS)
    #круг
    #диагональ верхняя
    glVertex3f(-0.2, 1, 0)
    glVertex3f(-0.2, 0.8, 0)
    glVertex3f(-0.7, 0.65, 0)
    glVertex3f(-0.7, 0.8, 0)
    
    glVertex3f(-0.2, 1, -0.2) # Задняя грань
    glVertex3f(-0.2, 0.8,  -0.2)
    glVertex3f(-0.7, 0.65, -0.2)
    glVertex3f(-0.7, 0.8, -0.2)
    
    glVertex3f(-0.2, 0.8, 0) # Нижняя грань
    glVertex3f(-0.7, 0.65, 0)
    glVertex3f(-0.7, 0.65, -0.2)
    glVertex3f(-0.2, 0.8, -0.2)
    
    glVertex3f(-0.2, 1, 0) # Верхняя грань
    glVertex3f(-0.7, 0.8, 0)
    glVertex3f(-0.7, 0.8, -0.2)
    glVertex3f(-0.2, 1, -0.2)

    
    #вертикальная линия
    glVertex3f(-0.55, 0.3, 0)
    glVertex3f(-0.55, 0.8, 0)
    glVertex3f(-0.7, 0.8, 0)
    glVertex3f(-0.7, 0.3, 0)
    
    glVertex3f(-0.55, 0.3, -0.2) # Задняя грань
    glVertex3f(-0.55, 0.8, -0.2)
    glVertex3f(-0.7, 0.8, -0.2)
    glVertex3f(-0.7, 0.3, -0.2)
    
    glVertex3f(-0.55, 0.3, 0) # нижняя грань
    glVertex3f(-0.7, 0.3, 0)
    glVertex3f(-0.7, 0.3, -0.2)
    glVertex3f(-0.55, 0.3, -0.2)
    
    glVertex3f(-0.55, 0.8, 0) # верхняя грань
    glVertex3f(-0.7, 0.8, 0)
    glVertex3f(-0.7, 0.8, -0.2)
    glVertex3f(-0.55, 0.8, -0.2)
    
    glVertex3f(-0.7, 0.8, 0) # Левая грань
    glVertex3f(-0.7, 0.3, 0)
    glVertex3f(-0.7, 0.3, -0.2)
    glVertex3f(-0.7, 0.8, -0.2)

    glVertex3f(-0.55, 0.8, 0) # Правая грань
    glVertex3f(-0.55, 0.3, 0)
    glVertex3f(-0.55, 0.3, -0.2)
    glVertex3f(-0.55, 0.8, -0.2)
    
    #диагональ нижняя
    glVertex3f(-0.2, 0.3, 0)
    glVertex3f(-0.2, 0.1, 0)
    glVertex3f(-0.7, 0.3, 0)
    glVertex3f(-0.7, 0.45, 0)
    
    glVertex3f(-0.2, 0.3, -0.2) # Задняя грань
    glVertex3f(-0.2, 0.1, -0.2)
    glVertex3f(-0.7, 0.3, -0.2)
    glVertex3f(-0.7, 0.45, -0.2)
    
    glVertex3f(-0.2, 0.3, 0) # Верхняя грань
    glVertex3f(-0.7, 0.45, 0)
    glVertex3f(-0.7, 0.45, -0.2)
    glVertex3f(-0.2, 0.3, -0.2)
    
    glVertex3f(-0.2, 0.1, 0) # Нижняя грань
    glVertex3f(-0.7, 0.3, 0)
    glVertex3f(-0.7, 0.3, -0.2)
    glVertex3f(-0.2, 0.1, -0.2)
    glEnd()
    
    
def apply_transformations():
    glLoadIdentity()  # Сбрасываем текущую матрицу моделирования

    # Перенос на текущие координаты объекта
    glTranslatef(translation[0], translation[1], translation[2])

    # Масштабирование
    glScalef(scale_factor, scale_factor, scale_factor)

    # Вращение вокруг заданной оси
    glRotatef(rotation_angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
    
def handle_key_press(key):
    global scale_factor, translation, rotation_angle, rotation_axis

    # Масштабирование
    if key == pygame.K_EQUALS:
        scale_factor += 0.1
    elif key == pygame.K_MINUS:
        scale_factor -= 0.1

    # Перенос
    if key == pygame.K_LEFT:
        translation[0] -= 0.1
    elif key == pygame.K_RIGHT:
        translation[0] += 0.1
    elif key == pygame.K_UP:
        translation[1] += 0.1
    elif key == pygame.K_DOWN:
        translation[1] -= 0.1

    # Вращение
    if key == pygame.K_a:
        rotation_axis = (0, 0, 1)
        rotation_angle -= 3.0
    elif key == pygame.K_d:
        rotation_axis = (0, 0, 1)
        rotation_angle += 3.0
    elif key == pygame.K_w:
        rotation_axis = (1, 0, 0)  # Вращение вокруг оси X
        rotation_angle -= 3.0
    elif key == pygame.K_s:
        rotation_axis = (1, 0, 0)  # Вращение вокруг оси X
        rotation_angle += 3.0
    elif key == pygame.K_q:
        rotation_axis = (0, 1, 0)  # Вращение вокруг оси Y
        rotation_angle -= 3.0
    elif key == pygame.K_e:
        rotation_axis = (0, 1, 0)  # Вращение вокруг оси Y
        rotation_angle += 3.0
    apply_transformations()  # Применить преобразования
    draw_letter_я()  # Отрисовать объект
        # Отображение проекций
    if key == pygame.K_x:
        projection_axis = "ox"
    elif key == pygame.K_y:
        projection_axis = "oy"
    elif key == pygame.K_z:
        projection_axis = "oz"

        
def draw_orthographic_projections():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    # Отрисовка проекции на плоскость Oxy
    glViewport(0, 0, 400, 300)
    glOrtho(-5, 5, -5, 5, -50, 50)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    draw_axes()
    apply_transformations()
    draw_letter_я()

    # Отрисовка проекции на плоскость Oxz
    glViewport(400, 0, 400, 300)
    glOrtho(-5, 5, -5, 5, -50, 50)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    draw_axes()
    apply_transformations()
    draw_letter_я()

    # Отрисовка проекции на плоскость Oyz
    glViewport(0, 300, 400, 300)
    glOrtho(-5, 5, -5, 5, -50, 50)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    draw_axes()
    apply_transformations()
    draw_letter_я()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                handle_key_press(event.key)

        glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_axes()  # Отрисовка системы координат

        #draw_orthographic_projections()  # Отрисовка проекций
        #apply_transformations()  # Применение преобразований
        draw_letter_я()  # Отрисовка объекта
        

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == '__main__':
    main()