class MoverMixin(object):
    on = False

    def connect(self):
        self.on = True
        connect = self.line.figure.canvas.mpl_connect
        self.cidpress = connect('button_press_event', self.on_press)
        self.cidrelease = connect('button_release_event', self.on_release)
        self.cidmotion = connect('motion_notify_event', self.on_motion)
        self.close_event = connect('close_event', self.on_close)

    def disconnect(self):
        disconnect = self.line.figure.canvas.mpl_disconnect
        disconnect(self.cidpress)
        disconnect(self.cidrelease)
        disconnect(self.cidmotion)

    def on_press(self, event):
        raise NotImplementedError

    def on_release(self, event):
        raise NotImplementedError

    def on_motion(self, event):
        raise NotImplementedError

    def on_close(self, event):
        self.on = False
        self.disconnect()

class LineMover(MoverMixin):
    def __init__(self, line, func):
        self.line = line
        self.canvas = line.figure.canvas
        self.press = None
        self.connect()
        self.func = func

    def on_press(self, event):
        if event.inaxes == self.line.axes:
            contains, attrd = self.line.contains(event)
            if contains:
                self.press = self.line.get_ydata() - event.ydata

    def on_motion(self, event):
        if self.press is not None and event.inaxes == self.line.axes:
            self.line.set_ydata(event.ydata + self.press)
            self.canvas.draw()

    def on_release(self, event):
        if self.press is not None:
            y_input = event.ydata + self.press
            self.press = None
            res = self.func(y_input[0])
            if res is not None:  # in case I need to ax.clear()
                self.line = res
            self.canvas.draw()
