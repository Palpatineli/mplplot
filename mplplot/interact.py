from matplotlib.lines import Line2D

class MoverMixin(object):
    on: bool = False
    line: Line2D

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

    def set_data(self, data: float):
        raise NotImplementedError

    def get_data(self) -> float:
        raise NotImplementedError

    @staticmethod
    def get_event(event):
        raise NotImplementedError

    def on_press(self, event):
        if event.inaxes == self.line.axes:
            self.press = self.get_data() - self.get_event(event)

    def on_motion(self, event):
        if self.press is not None and event.inaxes == self.line.axes:
            self.set_data(self.get_event(event) + self.press)
            self.canvas.draw()

    def on_release(self, event):
        if self.press is not None:
            res = self.func(self.get_event(event) + self.press[0])
            if res is not None:  # in case I need to ax.clear()
                self.line = res
            self.canvas.draw()
            self.press = None

class AxhlineMover(LineMover):
    def set_data(self, data: float):
        self.line.set_ydata(data)

    def get_data(self) -> float:
        return self.line.get_ydata()

    @staticmethod
    def get_event(event):
        return event.ydata

class AxvlineMover(LineMover):
    def set_data(self, data: float):
        self.line.set_xdata(data)

    def get_data(self) -> float:
        return self.line.get_xdata()

    @staticmethod
    def get_event(event):
        return event.xdata
