'''
 Copyright by Artem Vorontsov, Kaspersky Lab US, 2021
 email: artem7vorontsov@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

video_size = 512
fps = 18.0

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def prepare_video_writer(steps, L, filename):

    fig, axes = plt.subplots(1, 2, figsize=(2*6.4, 6.0))
    plt.subplots_adjust(wspace=0.2)

    axes[0].set_xlim([-L/4, L/4])
    axes[0].set_ylim([-L/4, L/4])

    axes[1].set_xlim([0, steps])
    axes[1].set_ylim([0.0, 1.0])

    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Performance metric')

    v_size = (int(2.2*video_size), video_size)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    wrt = cv2.VideoWriter(filename, fourcc, fps, v_size)

    cont = {'w_writer': wrt,
            'figure': fig,
            'axes': axes,
            'size': v_size,
            'L': L,
            'cnt': 0,
            'tr_cnt': 0}

    return cont
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def to_video(pos, u, loss, label, cont):

    wrt = cont['w_writer']
    fig = cont['figure']
    axes = cont['axes']
    v_size = cont['size']
    L = cont['L']
    cnt = cont['cnt']
    tr_cnt = cont['tr_cnt']
    delta = L/200

    c1 = plt.Circle((pos[0], pos[1]), 2*delta, facecolor='green', edgecolor='black')
    c2 = plt.Circle((u[0], u[1]), delta, facecolor='blue', edgecolor='blue')

    l1 = plt.Line2D((u[0] - 1.5*delta, u[0] + 1.5*delta), (u[1], u[1]), linewidth=1.0, color='blue')
    l2 = plt.Line2D((u[0], u[0]), (u[1] - 1.5*delta, u[1] + 1.5*delta), linewidth=1.0, color='blue')

    axes[0].plot(u[0], u[1], marker='.', markersize=0.5, color='blue')
    axes[0].plot(pos[0], pos[1], marker='.', markersize=0.5, color='green')

    axes[0].add_patch(c1), axes[0].add_patch(c2), axes[0].add_line(l1), axes[0].add_line(l2)

    if label == 'Inference':
        if np.mod(cnt, 6):
            axes[0].set_title(label, fontsize=15)
        else:
            axes[0].set_title('')
    else:
        axes[0].set_title(label, fontsize=15)

    if label == 'Training':
        cont['tr_cnt'] = len(loss)

    sp1 = axes[1].axvspan(0, tr_cnt, alpha=0.1, color='blue')

    if label == 'Inference':
        sp2 = axes[1].axvspan(tr_cnt, len(loss), alpha=0.1, color='green')

    loss = np.hstack(loss)
    lines = axes[1].plot(loss, linewidth=1, color='blue')

    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    frame = cv2.resize(img, dsize=v_size)

    c1.remove(), c2.remove(), l1.remove(), l2.remove()
    for line in lines:
        line.remove()
    sp1.remove()
    if label == 'Inference':
        sp2.remove()

    wrt.write(frame)

    cont['cnt'] = cont['cnt'] + 1

    return []
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def delete_video_stream(video_str):
    wrt = video_str['w_writer']

    cv2.destroyAllWindows()
    wrt.release()
    return []