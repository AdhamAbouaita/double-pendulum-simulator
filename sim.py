# ok

import sympy as sp
import numpy as np
import matplotlib as mpl
# Remove the default Matplotlib toolbar:
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
from matplotlib.collections import LineCollection

# ----------------------------------------------------------------------------------------
# Make fonts a bit larger and more uniform
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

# ----------------------------------------------------------------------------------------
# SYMBOLIC DERIVATION (same as before)
sp.init_printing(use_latex='mathjax')

m1_sym, m2_sym, g_sym, t_sym, b_sym = sp.symbols('m_1, m_2, g, t, b')
R1_sym, R2_sym = sp.symbols('R_1 R_2')
theta1 = sp.Function("theta_1")(t_sym)
theta2 = sp.Function("theta_2")(t_sym)
T1 = sp.Function("T_1")(t_sym)
T2 = sp.Function("T_2")(t_sym)

# First pendulum (top bob)
r1 = sp.Matrix([R1_sym*sp.sin(theta1), -R1_sym*sp.cos(theta1)])
a1 = sp.diff(r1, t_sym, 2)
forces1 = (sp.Matrix([0, -m1_sym*g_sym])
           + sp.Matrix([-T1*sp.sin(theta1), T1*sp.cos(theta1)])
           + sp.Matrix([T2*sp.sin(theta2), -T2*sp.cos(theta2)])
           + sp.Matrix([-b_sym*R1_sym*sp.diff(theta1, t_sym)*sp.cos(theta1),
                        -b_sym*R1_sym*sp.diff(theta1, t_sym)*sp.sin(theta1)]))
base1 = sp.Matrix([[sp.sin(theta1), -sp.cos(theta1)],
                   [sp.cos(theta1),  sp.sin(theta1)]])
eq1 = sp.simplify(base1*(m1_sym*a1 - forces1))

# Second pendulum (attached bob)
r2 = sp.Matrix([R2_sym*sp.sin(theta2), -R2_sym*sp.cos(theta2)])
a2 = sp.diff(r2, t_sym, 2)
forces2 = (sp.Matrix([0, -m2_sym*g_sym])
           + sp.Matrix([-T2*sp.sin(theta2), T2*sp.cos(theta2)])
           + sp.Matrix([-b_sym*R2_sym*sp.diff(theta2, t_sym)*sp.cos(theta2),
                        -b_sym*R2_sym*sp.diff(theta2, t_sym)*sp.sin(theta2)]))
base2 = sp.Matrix([[sp.sin(theta2), -sp.cos(theta2)],
                   [sp.cos(theta2),  sp.sin(theta2)]])
eq2 = sp.simplify(base2*(m2_sym*a2 - (forces2 - m2_sym*forces1/m1_sym)))

eqlist = [eq1[0], eq1[1], eq2[0], eq2[1]]
sol = sp.solve(eqlist, [T1, T2, sp.diff(theta1, t_sym, 2), sp.diff(theta2, t_sym, 2)])

omega1 = sp.Function("omega_1")(t_sym)
omega2 = sp.Function("omega_2")(t_sym)
alpha1 = sol[sp.diff(theta1, t_sym, 2)].subs({
    sp.diff(theta1, t_sym): omega1,
    sp.diff(theta2, t_sym): omega2
})
alpha2 = sol[sp.diff(theta2, t_sym, 2)].subs({
    sp.diff(theta1, t_sym): omega1,
    sp.diff(theta2, t_sym): omega2
})

aph1 = sp.lambdify(
    [theta1, theta2, omega1, omega2, g_sym, R1_sym, R2_sym, m1_sym, m2_sym, b_sym],
    alpha1
)
aph2 = sp.lambdify(
    [theta1, theta2, omega1, omega2, g_sym, R1_sym, R2_sym, m1_sym, m2_sym, b_sym],
    alpha2
)

# ----------------------------------------------------------------------------------------
# DEFINING THE VECTOR FIELD (for RK4 steps)
def vectorfield(var, t, cst):
    theta1_val, theta2_val, w1_val, w2_val = var  # w1 and w2 in rad/s
    g_val, L1_val, L2_val, m1_val, m2_val, b_val = cst
    return [
        w1_val,
        w2_val,
        aph1(theta1_val, theta2_val, w1_val, w2_val, g_val, L1_val, L2_val, m1_val, m2_val, b_val),
        aph2(theta1_val, theta2_val, w1_val, w2_val, g_val, L1_val, L2_val, m1_val, m2_val, b_val)
    ]

# ----------------------------------------------------------------------------------------
# GLOBAL SIMULATION PARAMETERS (inputs now in degrees for angles)
sim_params = {
    'm1': 1.0,
    'm2': 1.0,
    'l1': 1.0,
    'l2': 1.0,
    'g': 9.81,
    'b': 0.1,
    'theta1_deg': 176.3,   # top bob angle (deg)
    'theta2_deg': 176.3,   # second bob angle (deg)
    'w1': -1.0,           # angular velocity (deg/s) for top bob
    'w2': -0.5,           # angular velocity (deg/s) for second bob
    'h': 0.025,
    'time_mult': 1.0
}

current_state = None
current_time = 0.0
trail_points = []
running = True
cst = None

# ----------------------------------------------------------------------------------------
# FIGURE and WINDOW TITLE
fig = plt.figure(figsize=(12, 8))
fig.canvas.manager.set_window_title("Adham's Double Pendulum Simulator")

# ----------------------------------------------------------------------------------------
# LEFT AXIS: the Double Pendulum
ax_left = fig.add_axes([0.05, 0.45, 0.4, 0.5])
ax_left.set_aspect('equal', adjustable='box')
ax_left.set_xlim(-2.5, 2.5)
ax_left.set_ylim(-2.5, 2.5)
ax_left.set_title('Damped Double Pendulum')
ax_left.set_xlabel('x (m)')
ax_left.set_ylabel('y (m)')
ax_left.set_navigate(True)

# ----------------------------------------------------------------------------------------
# RIGHT AXIS: Time vs. Parameter
ax_right = fig.add_axes([0.55, 0.45, 0.4, 0.5])
ax_right.set_title('Pendulum Parameter vs. Time')
ax_right.set_xlabel('Time (s)')
ax_right.set_ylabel('theta1 (deg)')  # default label

time_data = []
y_data = []
line_right, = ax_right.plot([], [], 'r-')

# Define y-axis options. Angular values are converted to degrees.
# Additional options compute kinetic energies (in Joules).
plot_options = {
    "theta1 (deg)": lambda st: np.rad2deg(st[0]),
    "theta2 (deg)": lambda st: np.rad2deg(st[1]),
    "w1 (deg/s)":   lambda st: np.rad2deg(st[2]),
    "w2 (deg/s)":   lambda st: np.rad2deg(st[3]),
    "K1 (J)":       lambda st: 0.5 * sim_params['m1'] * (sim_params['l1'] * st[2])**2,
    "K2 (J)":       lambda st: 0.5 * sim_params['m2'] * ((sim_params['l1'] * st[2])**2 + (sim_params['l2'] * st[3])**2 + 2 * sim_params['l1'] * sim_params['l2'] * st[2] * st[3] * np.cos(st[0]-st[1])),
    "Total KE (J)": lambda st: (0.5 * sim_params['m1'] * (sim_params['l1'] * st[2])**2 +
                                  0.5 * sim_params['m2'] * ((sim_params['l1'] * st[2])**2 + (sim_params['l2'] * st[3])**2 +
                                  2 * sim_params['l1'] * sim_params['l2'] * st[2] * st[3] * np.cos(st[0]-st[1])))
}
current_plot_option = "theta1 (deg)"  # default

# ----------------------------------------------------------------------------------------
# Create an annotation for hover functionality on the right plot.
annot = ax_right.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                          bbox=dict(boxstyle="round", fc="w"),
                          arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    xdata, ydata_line = line_right.get_data()
    index = ind["ind"][0]
    pos = (xdata[index], ydata_line[index])
    annot.xy = pos
    text = "time = {:.2f}\nvalue = {:.2f}".format(pos[0], pos[1])
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    if event.inaxes == ax_right:
        cont, ind = line_right.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

# ----------------------------------------------------------------------------------------
# HELPER FOR CREATING LABELED TEXT BOXES
def create_labeled_textbox(pos, label_text, initial_value):
    ax_box = fig.add_axes(pos)
    tb = TextBox(ax_box, '', initial=str(initial_value))
    label_x = pos[0]
    label_y = pos[1] + pos[3] + 0.005
    fig.text(label_x, label_y, label_text, fontsize=10, va='bottom', ha='left')
    return tb

# ----------------------------------------------------------------------------------------
# CREATE TWO ROWS OF TEXT BOXES (bottom area)
# Row 1: parameters for masses, lengths, gravity, damping, and time multiplier.
text_box_m1 = create_labeled_textbox([0.05, 0.30, 0.08, 0.05], 'm1 (kg)', sim_params['m1'])
text_box_m2 = create_labeled_textbox([0.15, 0.30, 0.08, 0.05], 'm2 (kg)', sim_params['m2'])
text_box_l1 = create_labeled_textbox([0.25, 0.30, 0.08, 0.05], 'l1 (m)', sim_params['l1'])
text_box_l2 = create_labeled_textbox([0.35, 0.30, 0.08, 0.05], 'l2 (m)', sim_params['l2'])
text_box_g  = create_labeled_textbox([0.45, 0.30, 0.08, 0.05], 'g (m/s²)', sim_params['g'])
text_box_b  = create_labeled_textbox([0.55, 0.30, 0.08, 0.05], 'b', sim_params['b'])
text_box_time_mult = create_labeled_textbox([0.65, 0.30, 0.08, 0.05], 'Time mult.', sim_params['time_mult'])

# Row 2: angles and angular velocities (in degrees).
text_box_theta1 = create_labeled_textbox([0.05, 0.20, 0.08, 0.05], 'theta1 (deg)', sim_params['theta1_deg'])
text_box_theta2 = create_labeled_textbox([0.15, 0.20, 0.08, 0.05], 'theta2 (deg)', sim_params['theta2_deg'])
text_box_w1 = create_labeled_textbox([0.25, 0.20, 0.08, 0.05], 'w1 (deg/s)', sim_params['w1'])
text_box_w2 = create_labeled_textbox([0.35, 0.20, 0.08, 0.05], 'w2 (deg/s)', sim_params['w2'])

# ----------------------------------------------------------------------------------------
# BUTTONS at y=0.10
ax_button_update = fig.add_axes([0.05, 0.10, 0.2, 0.05])
button_update = Button(ax_button_update, 'Update Simulation')
ax_button_run = fig.add_axes([0.30, 0.10, 0.15, 0.05])
button_run = Button(ax_button_run, 'Pause')  # Running by default

# ----------------------------------------------------------------------------------------
# RadioButtons for selecting which parameter is plotted on the right.
ax_radio = fig.add_axes([0.50, 0.10, 0.15, 0.15])
radio = RadioButtons(ax_radio, list(plot_options.keys()))
def radio_changed(label):
    global current_plot_option, time_data, y_data
    current_plot_option = label
    ax_right.set_ylabel(label)
    # Reset the time-series data so the new plot starts fresh.
    time_data.clear()
    y_data.clear()
    line_right.set_data([], [])
    ax_right.relim()
    ax_right.autoscale_view()
    fig.canvas.draw_idle()
radio.on_clicked(radio_changed)

# ----------------------------------------------------------------------------------------
# NEW FEATURE: Query Fields for Time and Parameter.
# These two input fields (placed at the very bottom) allow you to determine the corresponding
# time from a given pendulum parameter value and vice versa.
text_box_query_time = create_labeled_textbox([0.05, 0.0, 0.20, 0.05], 'Query Time (s):', '')
text_box_query_param = create_labeled_textbox([0.30, 0.0, 0.20, 0.05], 'Query Parameter (value):', '')
updating_query = False  # flag to avoid recursive updates

def query_time_callback(text):
    global updating_query
    if updating_query:
        return
    try:
        t_input = float(text)
    except ValueError:
        return
    if len(time_data) == 0:
        return
    idx = np.argmin(np.abs(np.array(time_data) - t_input))
    param_value = y_data[idx]
    updating_query = True
    text_box_query_param.set_val("{:.2f}".format(param_value))
    updating_query = False

def query_param_callback(text):
    global updating_query
    if updating_query:
        return
    try:
        param_input = float(text)
    except ValueError:
        return
    if len(y_data) == 0:
        return
    idx = np.argmin(np.abs(np.array(y_data) - param_input))
    t_value = time_data[idx]
    updating_query = True
    text_box_query_time.set_val("{:.2f}".format(t_value))
    updating_query = False

text_box_query_time.on_submit(query_time_callback)
text_box_query_param.on_submit(query_param_callback)

# ----------------------------------------------------------------------------------------
# PENDULUM ART (ON LEFT AXIS): rods, bobs, and trail.
def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)

trail = LineCollection([], cmap=plt.get_cmap('gist_rainbow'),
                         linewidth=1, alpha=0.5, zorder=5)
ax_left.add_collection(trail)
line1, = ax_left.plot([0, 0], [0, 0], color='k', lw=2, zorder=10)
line2, = ax_left.plot([0, 0], [0, 0], color='k', lw=2, zorder=10)
circle1 = plt.Circle((0, 0), 0.08, ec="k", lw=1.5, zorder=15, fc='w')
circle2 = plt.Circle((0, 0), 0.08, ec="k", lw=1.5, zorder=15, fc='w')
ax_left.add_patch(circle1)
ax_left.add_patch(circle2)

# ----------------------------------------------------------------------------------------
# SIMULATION SETUP AND UPDATING
def run_simulation():
    global current_state, current_time, trail_points, sim_params, cst
    global time_data, y_data
    try:
        sim_params['m1'] = float(text_box_m1.text)
        sim_params['m2'] = float(text_box_m2.text)
        sim_params['l1'] = float(text_box_l1.text)
        sim_params['l2'] = float(text_box_l2.text)
        sim_params['g']  = float(text_box_g.text)
        sim_params['b']  = float(text_box_b.text)
        sim_params['time_mult'] = float(text_box_time_mult.text)
        sim_params['theta1_deg'] = float(text_box_theta1.text)
        sim_params['theta2_deg'] = float(text_box_theta2.text)
        sim_params['w1'] = float(text_box_w1.text)
        sim_params['w2'] = float(text_box_w2.text)
    except Exception as e:
        print("Error reading input fields, using defaults.", e)

    # Convert angles and angular velocities (deg -> rad).
    theta1_rad = np.deg2rad(sim_params['theta1_deg'])
    theta2_rad = np.deg2rad(sim_params['theta2_deg'])
    w1_rad = np.deg2rad(sim_params['w1'])
    w2_rad = np.deg2rad(sim_params['w2'])
    current_state = np.array([theta1_rad, theta2_rad, w1_rad, w2_rad], dtype=float)
    current_time = 0.0
    cst = [sim_params['g'], sim_params['l1'], sim_params['l2'],
           sim_params['m1'], sim_params['m2'], sim_params['b']]

    trail_points.clear()
    x1 = sim_params['l1'] * np.sin(theta1_rad)
    y1 = -sim_params['l1'] * np.cos(theta1_rad)
    x2 = x1 + sim_params['l2'] * np.sin(theta2_rad)
    y2 = y1 - sim_params['l2'] * np.cos(theta2_rad)
    trail_points.append((x2, y2))
    time_data.clear()
    y_data.clear()
    line_right.set_data([], [])
    ax_right.relim()
    ax_right.autoscale_view()
    update_plot(current_state)

def simulation_step(state, dt, cst):
    k1 = np.array(vectorfield(state, current_time, cst))
    k2 = np.array(vectorfield(state + dt/2 * k1, current_time, cst))
    k3 = np.array(vectorfield(state + dt/2 * k2, current_time, cst))
    k4 = np.array(vectorfield(state + dt * k3, current_time, cst))
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def update_plot(state):
    theta1_val, theta2_val, w1_val, w2_val = state
    L1 = sim_params['l1']
    L2 = sim_params['l2']
    x1 = L1 * np.sin(theta1_val)
    y1 = -L1 * np.cos(theta1_val)
    x2 = x1 + L2 * np.sin(theta2_val)
    y2 = y1 - L2 * np.cos(theta2_val)
    line1.set_data([0, x1], [0, y1])
    line2.set_data([x1, x2], [y1, y2])
    circle1.center = (x1, y1)
    circle2.center = (x2, y2)
    trail_points.append((x2, y2))
    xs, ys = zip(*trail_points)
    segs = make_segments(xs, ys)
    trail.set_segments(segs)
    time_data.append(current_time)
    y_val = plot_options[current_plot_option](state)
    y_data.append(y_val)
    line_right.set_data(time_data, y_data)
    ax_right.relim()
    ax_right.autoscale_view()
    fig.canvas.draw_idle()

def timer_callback():
    global current_state, current_time
    if running:
        dt = sim_params['h'] * sim_params['time_mult']
        current_state = simulation_step(current_state, dt, cst)
        current_time += dt
        update_plot(current_state)

def update_simulation(event):
    run_simulation()

def toggle_run(event):
    global running
    running = not running
    button_run.label.set_text("Pause" if running else "Start")

button_update.on_clicked(update_simulation)
button_run.on_clicked(toggle_run)

timer = fig.canvas.new_timer(interval=25)
timer.add_callback(timer_callback)
timer.start()

run_simulation()
plt.show()
