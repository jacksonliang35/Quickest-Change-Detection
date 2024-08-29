import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def MCFUN(x,c0,c1,c2):
    # Assuming x > 0
    try:
        x[0] += 0.000000001
    except:
        pass
    # c[0] is y-scale; c[1] is mu; c[2] is sigma.
    return 1+10**c0 * np.exp(-(np.log(x)-c1)**2 / 2 / c2**2) / c2

t = np.linspace(0, 20, 10000)

# Define initial parameters
init_c = (2,5,1)

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
plt.plot(t,np.exp(0.02*t))
plt.ylim((1,np.exp(0.02*20)))
line, = plt.plot(t, MCFUN(t, init_c[0], init_c[1], init_c[2]), lw=2)
ax.set_xlabel('Time')

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

axc0 = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
c0_slider = Slider(
    ax=axc0,
    label='log(c[0])',
    valmin=1,
    valmax=5,
    valinit=init_c[0],
    orientation="vertical"
)

axc1 = plt.axes([0.25, 0.15, 0.65, 0.02], facecolor=axcolor)
c1_slider = Slider(
    ax=axc1,
    label="c[1]",
    valmin=3,
    valmax=10,
    valinit=init_c[1]
)

axc2 = plt.axes([0.25, 0.05, 0.65, 0.02], facecolor=axcolor)
c2_slider = Slider(
    ax=axc2,
    label="c[2]",
    valmin=0.5,
    valmax=2,
    valinit=init_c[2]
)

# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(MCFUN(t, c0_slider.val, c1_slider.val, c2_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
c0_slider.on_changed(update)
c1_slider.on_changed(update)
c2_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.0, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    c0_slider.reset()
    c1_slider.reset()
    # c2_slider.reset()
button.on_clicked(reset)

plt.show()
