# main.py
import numpy as np
from config import N, M, K, U, sat_height
from hexgrid import generate_hex_grid
from users import create_users
from channel import free_space_path_loss, shadowed_rician_fading
from antenna import antenna_gain
from ssb import assign_ssb_periodicity, ue_random_access_delay
from utils import compute_boresight_angle

# Setup region and satellite
cell_centers = generate_hex_grid()
region_side = cell_centers.max(axis=0)
sat_pos = np.array([region_side[0] / 2, region_side[1] / 2, sat_height])

# Assign users
users_xy, users_cell_idx = create_users()

# Assign SSB periodicity
ssb_periodicity = assign_ssb_periodicity(K)

# Path loss per cell
pl_cells = free_space_path_loss(sat_pos, cell_centers)

# Shadowed-Rician channel per cell
fading_cells = shadowed_rician_fading(K)

# Simulate for one example user
example_user_idx = 0
user_xy = users_xy[example_user_idx]
cell_idx = users_cell_idx[example_user_idx]
cell_center = cell_centers[cell_idx]
pl = pl_cells[cell_idx]
fading = fading_cells[cell_idx]

# Boresight angle between user and beam (assume beam aiming at cell center)
theta = compute_boresight_angle(sat_pos, user_xy, cell_center)
gain = antenna_gain(theta)

# Store/print example output values
print(f"User {example_user_idx} in cell {cell_idx}:")
print(f"  Cell center {cell_center}")
print(f"  Path Loss {pl:.2e} | Fading {fading:.2e} | Antenna Gain {gain:.2e}")

# UE random access delay distribution
random_access_delays = ue_random_access_delay(users_cell_idx, ssb_periodicity, U)
print(f"Random access delay (mean): {random_access_delays.mean():.2f} ms")
