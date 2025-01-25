import pygame as pg


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def calc_dist(point, x, y):
    return ((x - point.x) ** 2 + (y - point.y) ** 2) ** 0.5

def calc_near(points, x, y, k):
    dist = []
    for i, point in enumerate(points):
        dist.append((calc_dist(point, x, y), i))

    dist.sort(key=lambda x: x[0])

    res = []
    for i in range(k):
        res.append(dist[i][1])
    res.sort()
    return res


screen_width = 800
screen_height = 600
k = 5

screen = pg.display.set_mode((screen_width, screen_height))

pg.init()

run = True

cords = []
points = []
point_color = (0, 0, 255)
file = open("points.txt", "r").readlines()

for line in file:
    if line != "":
        cords.append(list(map(int, line[:-1].split())))

for cord in cords:
    points.append(Point(cord[0], cord[1]))

f = False

while run:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
            break
        if event.type == pg.MOUSEBUTTONDOWN:
            x, y = event.pos
            points.append(Point(x, y))

    keys = pg.key.get_pressed()

    if keys[pg.K_u]:
        f = False

    if not f:
        f = True
        for x in range(screen_width):
            for y in range(screen_height):
                a = calc_near(points, x, y, k)
                r = sum(a) * 17 * 13 * (a[0] + 1) % 255
                g = 1
                for num in a:
                    g *= (num + 5) ** 2
                    g %= 255
                b = 0
                for num in a:
                    b += (num * 2) ** 2
                    b %= 255
                color = (r, g, b)
                screen.set_at((x, y), color)

    for point in points:
        pg.draw.circle(screen, point_color, (point.x, point.y), 5)
    pg.display.flip()
    pg.time.delay(30)

pg.quit()