import csv
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np

from kalman import KalmanFilter


def read_data(case):
	# Path for input file
	base_path = os.path.dirname(os.path.abspath(__file__))
	file_path = os.path.join(base_path, f'data/Data{case}.csv')

	# Read in data to a numpy array
	# Using csv.reader instead of numpy's genfromtxt because it's much faster
	with open(file_path) as csv_file:
		data = np.array(list(csv.reader(csv_file)), dtype=np.float64)

	return data


def get_output_paths(case):
	# Create output directory if not exist
	out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
	if not os.path.isdir(out_path):
		os.mkdir(out_path)

	# Create directory for plots of this case if not exist
	case_path = os.path.join(out_path, f'case_{case}')
	if not os.path.isdir(case_path):
		os.mkdir(case_path)

	return out_path, case_path


def save_data(data, filename, out_path):
	# Write data to file
	path = os.path.join(out_path, f'{filename}.csv')
	np.savetxt(path, data, delimiter=', ')


def plot_trajectory(data, title, index, case_path):
	# Plot a single trajectory and save image
	figure = plt.figure(index)
	ax = plt.axes(projection='3d')
	plt.title(title)

	# The range specifies the color of each point,
	# which increases linearly with the time of that position
	x = data[:, 0].size
	ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=range(x), cmap='Greens')

	path = os.path.join(case_path, f'{title}.png')
	figure.savefig(path)


def plot_combined(data1, data2, title, index, case_path):
	# Plot 2 trajectories in a single graph and save image
	figure = plt.figure(index)
	ax = plt.axes(projection='3d')
	plt.title(title)

	# The noisy positions are radially further from the filtered positions.
	# We use a very low opacity to prevent them from entirely covering the
	# updated positions in the combined plot.
	x = data1.shape[0]
	ax.scatter3D(
		data1[:, 0], data1[:, 1], data1[:, 2],
		c=range(x), cmap='Reds', edgecolors='none', alpha=0.1
	)
	# We reverse the color gradient direction to prevent ambiguity in the plot
	ax.scatter3D(
		data2[:, 0], data2[:, 1], data2[:, 2],
		c=range(x, 0, -1), cmap='Greens'
	)

	path = os.path.join(case_path, f'{title}.png')
	figure.savefig(path)


def main():
	# Time interval between measurements
	dt = 1
	dt2 = dt * dt

	# Initial process covariance matrix
	# We'll set the position variance to 1 as the absolute value doesn't matter
	# all other values will be relative to this
	# We're blind guessing initial velocity and acceleration (as 0)
	# so we give them a very high initial variance value
	initial_pos_var = 1
	initial_vel_var = 1e-9
	initial_acc_var = 1e-9
	initial_cov = np.array([
		[initial_pos_var, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, initial_pos_var, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, initial_pos_var, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, initial_vel_var, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, initial_vel_var, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, initial_vel_var, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, initial_acc_var, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, initial_acc_var, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, initial_acc_var],
	])

	# Measurement covariance matrix
	# We use the same position variance as our initial as the source of data
	# for both is the same.
	# Further, we're combining the values from 5 stations for the velocity
	# so the variance is one 5th.
	pos_var = initial_pos_var
	vel_var = pos_var / 5
	measurement_cov = np.array([
		[pos_var, 0, 0, 0, 0, 0],
		[0, pos_var, 0, 0, 0, 0],
		[0, 0, pos_var, 0, 0, 0],
		[0, 0, 0, vel_var, 0, 0],
		[0, 0, 0, 0, vel_var, 0],
		[0, 0, 0, 0, 0, vel_var],
	])

	# State Transition Matrix
	# We assume a constant acceleration model without any process information
	a = np.array([
		[1, 0, 0, dt, 0, 0, dt2 / 2, 0, 0],
		[0, 1, 0, 0, dt, 0, 0, dt2 / 2, 0],
		[0, 0, 1, 0, 0, dt, 0, 0, dt2 / 2],
		[0, 0, 0, 1, 0, 0, dt, 0, 0],
		[0, 0, 0, 0, 1, 0, 0, dt, 0],
		[0, 0, 0, 0, 0, 1, 0, 0, dt],
		[0, 0, 0, 0, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 1],
	])

	# Control Matrix
	# As we have no information about the process, we don't perform any updates
	b = np.zeros((9, 1))

	# Estimate Error Transform Matrix
	# We're measuring the first 6 of our 9 state variables
	h = np.array([
		[1, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 1, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 1, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 1, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 1, 0, 0, 0],
	])

	# Process noise matrix
	# Noise introduced in each step of the process
	# Standard matrix for constant acceleration model with dt = 1
	# Noise coefficient computed while tuning, generalizes well to all CSVs.
	noise_coeff = 1e-20
	q = noise_coeff * np.array([
		[1, 0, 0, 2, 0, 0, 2, 0, 0],
		[0, 1, 0, 0, 2, 0, 0, 2, 0],
		[0, 0, 1, 0, 0, 2, 0, 0, 2],
		[2, 0, 0, 4, 0, 0, 4, 0, 0],
		[0, 2, 0, 0, 4, 0, 0, 4, 0],
		[0, 0, 2, 0, 0, 4, 0, 0, 4],
		[2, 0, 0, 4, 0, 0, 4, 0, 0],
		[0, 2, 0, 0, 4, 0, 0, 4, 0],
		[0, 0, 2, 0, 0, 4, 0, 0, 4],
	])

	# We use this counter to generate unique indices for the figures
	c = itertools.count()

	# Titles and filenames for the figures
	label_m = 'trajectory_measured'
	label_u = 'trajectory_updated'
	label_p_m = 'partial_trajectory_measured'
	label_p_u = 'partial_trajectory_updated'
	label_c = 'trajectory_combined'

	# Iterate over all 4 cases
	for case in range(1, 5):
		print(f'Case {case}: Initializing')
		# Read in data from the current csv
		positions = read_data(case)

		# Store previous positions wrt the last 5 stations to compute velocity
		prev_pos = positions[0, 3:].reshape((5, 3))

		# Store updated drone positions for plotting
		updated_positions = np.zeros(positions[:, :3].shape)
		updated_positions[0] = positions[0, :3]

		# Each iteration compute the station position
		station_positions = np.zeros((5, 3))

		# Initial State Vector
		# [px, py, pz, vx, vy, vz, ax, ay, az]'
		# Has the 3 positions wrt station 1 and then 6 0s
		initial_state = np.atleast_2d(np.hstack((positions[0, :3], [0] * 6))).T

		# Create KalmanFilter object
		kalman = KalmanFilter(
			initial_state, initial_cov, measurement_cov, a, b, h, q
		)

		print(f'Case {case}: Starting Kalman iterations')
		# Iterate over positions starting with the second
		for i, pos in enumerate(positions[1:]):
			# Get the velocities wrt the 5 stations and average
			cur_pos = pos[3:].reshape((5, 3))
			vel = np.mean(cur_pos - prev_pos, axis=0) / dt

			# Compute process and measurement vectors
			# Process is simply 0 as we have no control information
			# Measurement is the 3 positions and 3 averaged velocities
			process = np.zeros(1)
			measurement = np.atleast_2d(np.hstack((pos[:3], vel))).T

			# Perform an iteration of the Filter, store updated position
			kalman.iterate(process, measurement)
			updated_positions[i + 1] = kalman.state[:, 0].T[:3]

			# The position of the station is simply a running average
			# of the difference in position of the drone wrt station 1
			# and wrt the target station
			station_positions *= i / (i + 1)
			station_positions += (pos[:3] - cur_pos) / (i + 1)

			# Update variables for next iteration
			prev_pos = cur_pos

		print(f'Case {case}: Kalman iterations done')

		# Generate output folders if not exist and get path
		out_path, case_path = get_output_paths(case)

		# Save drone and station position data to output file
		save_data(updated_positions, f'drone_positions_{case}', out_path)
		save_data(station_positions, f'station_positions_{case}', out_path)

		# Plotting
		plot_trajectory(positions[:, :3], label_m, next(c), case_path)
		plot_trajectory(updated_positions, label_u, next(c), case_path)
		plot_trajectory(positions[-200:, :3], label_p_m, next(c), case_path)
		plot_trajectory(updated_positions[-200:], label_p_u, next(c), case_path)
		plot_combined(
			positions[:, :3], updated_positions,
			label_c, next(c), case_path
		)

		print(f'Case {case}: Data saved to output folder')


if __name__ == '__main__':
	main()
