import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------
# Physical and simulation parameters
R = 10                    # Radius of the circular container
g = 100                  # Gravitational acceleration (m/s^2)
dt = 0.02                 # Main simulation time step
dt_small = 0.0005         # Time step used for collision time refinement

# Global state variables
pos = np.array([0.0, 0.0])    # Current ball position (will be set on click)
vel = np.array([0.0, 0.0])    # Current ball velocity (set on click)
path_history = []           # List to store trajectory for drawing
simulation_active = False   # Start simulation only after user click

# ----------------------
# Helper functions for physics

def projectile_update(pos, vel, dt, a):
    """
    Update position and velocity for projectile motion under constant acceleration a.
    pos_new = pos + vel*dt + 0.5*a*dt^2
    vel_new = vel + a*dt
    """
    pos_new = pos + vel * dt + 0.5 * a * dt**2
    vel_new = vel + a * dt
    return pos_new, vel_new

def find_collision_time(pos, vel, dt, R, a):
    """
    Uses bisection to find the collision time t_coll in [0, dt]
    at which the ball's distance equals R.
    
    The ball follows:
      pos(t) = pos + vel*t + 0.5*a*t^2,
    and we want the smallest t in (0, dt) such that ||pos(t)|| = R.
    Assumes that at t=0 the ball is inside (||pos|| < R)
    and at t=dt it is outside (||pos(dt)|| > R).
    """
    t_low = 0.0
    t_high = dt
    for _ in range(20):  # iterate to refine t_coll
        t_mid = (t_low + t_high) / 2.0
        pos_mid = pos + vel * t_mid + 0.5 * a * t_mid**2
        if np.linalg.norm(pos_mid) < R:
            t_low = t_mid
        else:
            t_high = t_mid
    return t_high

def reflect_velocity(v, collision_point):
    """
    Reflects the velocity vector v with respect to the circular boundary at collision_point.
    The outward normal is n = collision_point/||collision_point||.
    The reflected velocity: v' = v - 2*(v·n)*n.
    """
    n = collision_point / np.linalg.norm(collision_point)
    return v - 2 * np.dot(v, n) * n

# ----------------------
# Main simulation update for each animation frame

def update_simulation():
    """
    Advances the simulation by one dt.
    Uses projectile motion until a collision is detected.
    If a collision is found within dt, the collision time is found by bisection,
    the collision is processed (with velocity reflection) and the remaining time
    in dt is simulated.
    """
    global pos, vel, path_history
    a = np.array([0, -g])
    t_remaining = dt

    # Process the time step, handling at most one collision per dt.
    while t_remaining > 1e-8:
        # Predict next position using current dt segment
        pos_next, vel_next = projectile_update(pos, vel, t_remaining, a)
        if np.linalg.norm(pos_next) <= R:
            # No collision in this time interval; update normally.
            pos, vel = pos_next, vel_next
            path_history.append(pos.copy())
            t_remaining = 0
        else:
            # A collision occurs within t_remaining.
            # Find the collision time t_coll using bisection.
            t_coll = find_collision_time(pos, vel, t_remaining, R, a)
            # Compute state at collision.
            collision_pos, vel_at_collision = projectile_update(pos, vel, t_coll, a)
            path_history.append(collision_pos.copy())
            # Reflect the velocity instantly.
            vel_reflected = reflect_velocity(vel_at_collision, collision_pos)
            # Update state after collision.
            pos = collision_pos
            vel = vel_reflected
            # Reduce remaining time and continue the simulation for the rest of the dt.
            t_remaining -= t_coll

def simulate_prediction(num_bounces=5):
    """
    Numerically simulate the future collision points starting from the current state,
    using a small dt for integration. This function returns an array of predicted
    collision positions.
    """
    a = np.array([0, -g])
    pos_pred = pos.copy()
    vel_pred = vel.copy()
    predictions = [pos_pred.copy()]
    time_to_simulate = 5  # simulate up to 5 seconds into the future
    t = 0
    bounce_count = 0
    while t < time_to_simulate and bounce_count < num_bounces:
        # Use a small integration step.
        pos_next = pos_pred + vel_pred * dt_small + 0.5 * a * dt_small**2
        if np.linalg.norm(pos_next) <= R:
            pos_pred, vel_pred = pos_next, vel_pred + a * dt_small
            t += dt_small
        else:
            # Collision detected: refine collision time in [0, dt_small].
            t_coll = find_collision_time(pos_pred, vel_pred, dt_small, R, a)
            collision_pos, vel_at_collision = projectile_update(pos_pred, vel_pred, t_coll, a)
            predictions.append(collision_pos.copy())
            # Reflect velocity
            vel_pred = reflect_velocity(vel_at_collision, collision_pos)
            pos_pred = collision_pos.copy()
            t += t_coll
            bounce_count += 1
    return np.array(predictions)

# ----------------------
# Set up the plot and interaction

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-R-1, R+1)
ax.set_ylim(-R-1, R+1)
ax.set_aspect('equal')
ax.set_title("Interactive Bouncing Ball with Gravity and Predicted Path")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)

# Draw the circular boundary.
circle = plt.Circle((0, 0), R, color='black', fill=False, linewidth=2)
ax.add_artist(circle)
instruction_text = ax.text(-R+0.5, R-1, "Click inside the circle to launch the ball", fontsize=10, color='gray')

# Plot elements: ball marker, full trajectory, and predicted collision points.
ball_plot, = ax.plot([], [], 'o', color='blue', markersize=8)
trace_plot, = ax.plot([], [], '-', color='blue', linewidth=1)
pred_plot, = ax.plot([], [], '--', color='red', linewidth=1)

def update(frame):
    """
    Animation update function.
    If simulation is active, update the physics and then update the plot elements.
    """
    if simulation_active:
        update_simulation()
        # Update ball marker (wrap coordinates in lists)
        ball_plot.set_data([pos[0]], [pos[1]])
        # Update full trajectory trace.
        path_arr = np.array(path_history)
        trace_plot.set_data(path_arr[:, 0], path_arr[:, 1])
        # Compute predicted collision points.
        predictions = simulate_prediction(num_bounces=5)
        if predictions.size:
            pred_plot.set_data(predictions[:, 0], predictions[:, 1])
    return ball_plot, trace_plot, pred_plot

def on_click(event):
    """
    When the user clicks inside the circle, the simulation is activated.
    The ball starts at the clicked position with a random initial velocity.
    """
    global pos, vel, simulation_active, path_history
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    if x**2 + y**2 > R**2:
        return
    pos = np.array([x, y], dtype=float)
    # Random initial velocity: choose a random direction and a speed in a realistic range.
    angle = np.random.uniform(0, 2*np.pi)
    speed = np.random.uniform(5, 15)
    vel = np.array([speed * np.cos(angle), speed * np.sin(angle)], dtype=float)
    path_history = [pos.copy()]
    simulation_active = True
    instruction_text.set_visible(False)

fig.canvas.mpl_connect('button_press_event', on_click)

# Create the animation; if backend issues arise with blitting, remove "blit=True".
anim = FuncAnimation(fig, update, frames=600, interval=20, blit=True)

plt.show()
