from dis import dis
from tkinter import VERTICAL
from turtle import color
from myutils import convert_list_to_dict

"""I had to rename this file from plot_utils.py to Plot_Utils.py because "plot_utils"
shadows another library meaning and I was having issuse importing it in VGSales.ipynb"""
import matplotlib.pyplot as plt
def create_bar_chart(data, X_label="Type",Y_label="Frequency"): #frequency diagram
    """
    data: list[any]
    converts data into a dictionary then creates a bar graph 
    showing the frequencies of each value
    """
    
    data_dictionary = convert_list_to_dict(data)

    figure = plt.figure()
    ax = figure.add_axes([0,0,2.4,1])
    plt.xlabel(X_label)
    plt.ylabel(Y_label)

    ax.bar(data_dictionary.keys(), data_dictionary.values())
    plt.show()


def create_pie_chart(data, labels):
    """
    data: list[int] stores the values that corrispond to each label
    labels: list[str] stores all the labels
    """

    plt.pie(data, labels=labels, autopct="%1.1f%%")
    plt.show()

def create_histogram(data, num_bins=10, xticks=None):
    """creates a wide histogram"""
    #data_dictionary = convert_list_to_dict(data)
    #plt.locator_params(axis='x', nbins=10)
    #plt.xticks(xticks)
    data = sorted(data)
    figure = plt.figure()
    #figure.xticks()
    ax = figure.add_axes([0,0,2.4,1])
    #plt.xlim(11,42)
    #plt.ylim(0,50)
    ax.hist(data,bins=num_bins, color='g')
    
    plt.show()


def create_scatter_plot(x,y,x_name="X", y_name="Y"):
    """creates a simple scatter plot"""

    plt.scatter(x,y)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()

def create_scatter_plot_w_regression_line(x,y,m,c,x_name="X", y_name="Y",ticks=None):
    """creates a scatter plot then plots a line for best fit right through 
        the data"""
    #create_scatter_plot(x,y,x_name=x_name,y_name=y_name)
    plt.scatter(x,y)

    plt.plot([min(x), max(x)],[m * min(x)+c, m * max(x) + c], c='r',lw=5)
    if ticks != None:
        plt.xticks(ticks)
        plt.yticks(ticks)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()



def create_box_plot(distributions, labels, y_label = "Frequency", x_label="Type",large=False):
    """create s box plot with help from gina sprints demo
        demo-> https://github.com/GonzagaCPSC322/U3-Data-Analysis/blob/master/JupyterNotebookFunS2/MatplotlibExamples.ipynb
    """
    figure = plt.figure()

    if large:
        ax = figure.add_axes([0,0,2.6,1.3])
    plt.boxplot(distributions)
    plt.xticks(list(range(1,len(distributions)+ 1)), labels,rotation=45)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    #figure.suptitle(label,fontsize=14, fontweight='bold')


    #plt.annotate("$\mu=100$", xy=(1.5, 105), xycoords="data", horizontalalignment="center")
    #plt.annotate("$\mu=100$", xy=(0.5, 0.5), xycoords="axes fraction", 
    #             horizontalalignment="center", color="blue")


    plt.show()

    

