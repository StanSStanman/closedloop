import matplotlib.pyplot as plt
import xarray as xr


class online_vis:

    def __init__(self, n_sample=1e3, figsize=(10, 9), stages=True):
        self.n_sample = int(n_sample)
        self.figsize = figsize
        self.stages = stages
        self.data = None
        self.x_ax = None
        plt.ion()
        plt.pause(1e-10)
        pass

    def update_x(self):
        self.x_ax = list(range(self.data.shape[1]))
        return self.x_ax

    def plot(self):
        self.x_ax = list(range(self.data.shape[1]))

        if self.stages:
            self.fig, self.axs = plt.subplots(2, 1, 
                                              gridspec_kw={
                                                  'height_ratios': [3, 1]},
                                              figsize=self.figsize,
                                              sharex=True)
            self.axs[0].set_ylim([-2e-4, 2e-4])
            self.axs[1].set_ylim([-2, 5])
            self.ln = self.axs[0].plot(self.x_ax, self.data.T)
            self.ln_s = self.axs[1].plot(self.x_ax, self.data.stage)
        else:
            self.fig, self.axs = plt.subplots(1, 1, figsize=self.figsize)
            self.axs.set_ylim([-2e-4, 2e-4])
            self.ln = self.axs.plot(self.x_ax, self.data.T)

    def update(self, data):
        if self.x_ax == None:
            if self.data is None:
                self.data = data
            else: 
                self.data = xr.concat((self.data, data), dim='stage')

            if self.data.shape[-1] >= self.n_sample:
                self.plot()

        else:
            self.data = xr.concat((self.data, data), dim='stage')
            if self.data.shape[1] > self.n_sample:
                self.data = self.data[:, -self.n_sample:]
            self.x_ax = self.update_x()

            for i in range(len(self.ln)):
                self.ln[i].set_xdata(self.x_ax)
                self.ln[i].set_ydata(self.data[i].T)
            if self.stages:
                self.ln_s[0].set_xdata(self.x_ax)
                self.ln_s[0].set_ydata(self.data.stage)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
