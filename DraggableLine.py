# helper_functions.py
import numpy as np
from matplotlib.lines import Line2D

class DraggableHorizontalLine:
    def __init__(self, ax, y, color='green', linestyle='--'):
        self.ax = ax
        self.line = Line2D(ax.get_xlim(), [y, y], color=color, linestyle=linestyle)
        self.ax.add_line(self.line)
        self.press = None
        self.connect()

    def connect(self):
        self.cidpress = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax: return
        contains, _ = self.line.contains(event)
        if not contains: return
        y0 = self.line.get_ydata()[0]
        self.press = y0, event.ydata

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.ax: return
        y0, ypress = self.press
        dy = event.ydata - ypress
        ynew = y0 + dy
        self.line.set_ydata([ynew, ynew])
        self.line.figure.canvas.draw()

    def on_release(self, event):
        self.press = None
        self.line.figure.canvas.draw()

    def disconnect(self):
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)

class DraggableVerticalLine:
    def __init__(self, ax, x, color='blue', linestyle='--'):
        self.ax = ax
        self.line = Line2D([x, x], ax.get_ylim(), color=color, linestyle=linestyle)
        self.ax.add_line(self.line)
        self.press = None
        self.connect()

    def connect(self):
        self.cidpress = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax: return
        contains, _ = self.line.contains(event)
        if not contains: return
        x0 = self.line.get_xdata()[0]
        self.press = x0, event.xdata

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.ax: return
        x0, xpress = self.press
        dx = event.xdata - xpress
        xnew = x0 + dx
        self.line.set_xdata([xnew, xnew])
        self.line.figure.canvas.draw()

    def on_release(self, event):
        self.press = None
        self.line.figure.canvas.draw()

    def disconnect(self):
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)
