#from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
from IPython import display
import os



# def plot(scores, mean_scores, return_, mean_return, loss, mean_loss, show=True):
#     if show:
#         plt.ion()
    
#     plt.figure(1)
#     display.clear_output(wait=True)
#     display.display(plt.gcf())
#     plt.clf()
#     plt.title('Training...')
#     plt.xlabel('Number of Games')
#     plt.ylabel('Score')
#     plt.plot(scores, linestyle='-', label='score')
#     plt.plot(mean_scores, linestyle='--', label='mean score')
#     plt.ylim(ymin=0)
#     plt.text(len(scores)-1, scores[-1], str(scores[-1]))
#     plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1],2)))
#     plt.legend(loc="upper left", bbox_to_anchor=(1.05,1))
#     plt.tight_layout()
#     if show:
#         plt.show(block=False)

#     plt.figure(2)
#     plt.clf()
#     plt.title('Training...')
#     plt.xlabel('Number of Games')
#     plt.ylabel('Reward')
#     plt.plot(return_, linestyle='-', label='return')
#     plt.plot(mean_return, linestyle='--', label='mean return')
#     #plt.ylim(ymin=0)
#     plt.text(len(return_)-1, return_[-1], str(return_[-1]))
#     plt.text(len(mean_return)-1, round(mean_return[-1],2), str(round(mean_return[-1],2)))
#     plt.legend(loc="upper left", bbox_to_anchor=(1.05,1))
#     plt.tight_layout()
#     if show:
#         plt.show(block=False)
        
#     plt.figure(3)
#     plt.clf()
#     plt.title('Training...')
#     plt.xlabel('Number of Games')
#     plt.ylabel('Loss')
#     plt.plot(loss, linestyle='-', label='loss')
#     plt.plot(mean_loss, linestyle='--', label='mean loss')
#     plt.ylim(ymin=0)
#     plt.text(len(loss)-1, loss[-1], str(round(loss[-1],4)))
#     plt.text(len(mean_loss)-1, round(mean_loss[-1],2), str(round(mean_loss[-1],4)))
#     plt.legend(loc="upper left", bbox_to_anchor=(1.05,1))
#     plt.tight_layout()
#     if show:
#         plt.show(block=False)
#         plt.pause(.1)

def plot(X, label, figure_num, xlabel=None, ylabel=None, show=False, clear=False, round_num=None, delay=None):
    if show:
        plt.ion()
    else:
        plt.ioff()
    
    plt.figure(figure_num)

    display.clear_output(wait=True)
    display.display(plt.gcf())

    if clear:
        plt.clf()

    plt.title('Training...')

    if xlabel is not None and ylabel is not None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    plt.plot(X, linestyle='-', label=label)
    #plt.ylim(ymin=0)

    if round_num is not None:
        text_label = str(round(X[-1], round_num))
    else:
        text_label = str(X[-1])

    plt.text(len(X)-1, X[-1], text_label)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05,1))
    plt.tight_layout()

    if show:
        plt.show(block=False)

    if delay is not None and show is True:
        plt.pause(delay)

def save_plot(figure_num, file_name):
    
    images_folder_path = './images'
    if not os.path.exists(images_folder_path):
        os.makedirs(images_folder_path)

    plt.figure(figure_num)
    file_directory = os.path.join(images_folder_path, file_name)
    plt.savefig(file_directory, bbox_inches='tight')









