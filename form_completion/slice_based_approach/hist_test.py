import sys
sys.path.append(r'C:\Users\palmerjh\Documents\AIVAS')
import numpy as np
import matplotlib.pyplot as plt
import os

#import similarity2 as sim
import taskpic as tp

pic_path = r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\gimp_tests\manual_fc'

choice_pic = r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\problems\09\choices\A\slice1.jpg'
easel_pic = r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\problems\09\box_cropped.jpg'

def main():
    #choice, easel = tuple([tp.TaskPic(os.path.join(pic_path, pic)) for pic in os.listdir(pic_path)])
    choice, easel = tuple([tp.TaskPic(os.path.join(pic_path, pic)) for pic in [choice_pic,easel_pic]])
    hist_choice, bins = np.histogram(choice.pix,bins=40,range=(0.1,1.0))
    hist_easel, _ = np.histogram(easel.pix,bins=40,range=(0.1,1.0))

    #print hist_choice
    #print hist_easel

    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    plt.bar(center, softmax(hist_choice), align='center', width=width)
    plt.title('Box Choice Distribution')
    #plt.show()
    plt.draw()
    fig = plt.gcf()
    fig.savefig('hist_choice_box.jpg')
    plt.clf()

    plt.bar(center, softmax(hist_easel), align='center', width=width)
    plt.title('Box Easel Distribution')
    #plt.show()
    plt.draw()
    fig = plt.gcf()
    fig.savefig('hist_easel_box.jpg')
    plt.clf()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == '__main__':
    main()
