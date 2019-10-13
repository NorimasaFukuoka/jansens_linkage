import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
os.makedirs('figures', exist_ok=True)
os.makedirs('gifs', exist_ok=True)


def main():
    for rad in np.linspace(0, 2*np.pi, 60):
        plot_jansen(rad)
    files = [f'figures/{rad}rad.png' for rad in np.linspace(0, 2*np.pi, 60)]
    files = list(filter(lambda f: os.path.exists(f), files))
    imgs = list(map(lambda f: Image.open(f), files))
    imgs[0].save(f'gifs/jansen_linkage.gif', save_all=True,
                 append_images=imgs[1:], duration=20, loop=0)


def rotate(xy, rad=None, deg=None):
    if deg is not None:
        rad = np.deg2rad(deg)
    sin = np.sin(rad)
    cos = np.cos(rad)
    rot = np.array([[cos, -sin], [sin,  cos]])
    return np.dot(rot, xy)


def triangle(p1, p2, l1, l2):
    '''
    Calc the position of P3 from P1, P2, l1, l2.
    '''
    l3 = np.linalg.norm(p2 - p1)
    # Translate so that P1 overlaps the origin.
    # P2 -> P2', P3 -> P3'
    _p2 = p2 - p1
    # Rotate so that P2' is on the X axis around the origin.
    # P2' -> P2'', P3' -> P3''
    alpha = np.arctan(_p2[1]/_p2[0])
    if _p2[0] > 0:
        pass
    else:
        alpha = np.pi + alpha
    _p2_ = rotate(_p2, rad=-alpha)
    # Calc X of P3'' with cosine theorem.
    x = (np.square(l2) + np.square(l3) - np.square(l1)) / (2*l3)
    # Calc Y of P3'' with Heron's formula.
    s = (l1+l2+l3)/2
    S = np.sqrt(s*(s-l1)*(s-l2)*(s-l3))
    y = 2*S/l3
    # Rotate in reverse directiron -> translate.
    p3 = rotate(np.array([x, y]), rad=alpha) + p1
    return p3


def get_jansen_points(rad):
    O = np.zeros(2)
    A = np.array([15 * np.cos(rad), 15 * np.sin(rad)])
    B = np.array([-38., -7.8])
    C = triangle(B, A, 50., 41.5)
    D = triangle(A, B, 39.3, 61.9)
    E = triangle(B, C, 55.8, 40.1)
    F = triangle(D, E, 39.4, 36.7)
    G = triangle(D, F, 65.7, 49.)
    dic = {k: p for k, p in zip(list('OABCDEFG'),
                                (O, A, B, C, D, E, F, G))}
    return dic

def plot_jansen(rad):
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, aspect='equal')
    ax.set_xlim(-120, 120)
    ax.set_ylim(-100, 60)
    ax.grid()
    lines = ('OA', 'OB', 'AB', 'AC', 'BC', 'BD',
             'BE', 'CE', 'EF', 'DF', 'DG', 'FG')
    for r in (rad, rad + np.pi):
        left = get_jansen_points(r)
        right = {k: np.array([p[0]*-1, p[1]]) for k, p
                 in get_jansen_points(np.pi - r).items()}
        for points, c in zip((left, right), ('c', 'm')):
            for line in lines:
                v1, v2 = list(line)
                ax.plot(*zip(points[v1], points[v2]), f'{c}-')
    plt.savefig(f'figures/{rad}rad.png')
    plt.close()


if __name__ == '__main__':
    main()
