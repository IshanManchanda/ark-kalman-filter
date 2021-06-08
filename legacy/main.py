import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

from kalman import KalmanFilter


def read_data():
	# Path for input file
	base_path = os.path.dirname(os.path.abspath(__file__))
	file_path = os.path.join(base_path, 'data/Data1.csv')

	# Read in data to a numpy array
	with open(file_path) as csv_file:
		data = np.array(list(csv.reader(csv_file)), dtype=np.float64)

	# data = data.reshape((-1, 4))
	idx = [0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 16, 2, 5, 8, 11, 14, 17]
	return data[:, idx]


def save_data(data, filename):
	# Create output directory if not exist
	dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
	if not os.path.isdir(dir_path):
		os.mkdir(dir_path)

	# Write data to file
	path = os.path.join(dir_path, f'{filename}.txt')
	np.savetxt(path, data, delimiter=' , ')


def plot_trajectory(data, title, index):
	# Plot a single trajectory and save as image
	# TODO: Change to 3D plot
	figure = plt.figure(index)
	ax = plt.axes(projection='3d')
	plt.title(title)

	x = data[:, 0].size
	ax.scatter3D(data[:, 0], data[:, 6], data[:, 12], c=range(x), cmap='Greens')

	base_path = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(base_path, f'out/{title}.png')
	figure.savefig(path)


def plot_combined(data1, data2, label1, label2, title, index):
	# Plot 2 trajectories in a single graph and save as imag
	figure = plt.figure(index)
	ax = plt.axes(projection='3d')
	plt.title(title)
	plt.legend([label1, label2])

	x = data1.shape[0]
	ax.scatter3D(
		data1[:, 0], data1[:, 6], data1[:, 12], c=range(x),
		cmap='Reds', edgecolors='none', alpha=0.25
	)
	ax.scatter3D(data2[:, 0], data2[:, 6], data2[:, 12], c=range(x), cmap='viridis')

	base_path = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(base_path, f'out/{title}.png')
	figure.savefig(path)


def main():
	# dt computed while exploring data by averaging time periods
	# over the given dataset after removing outliers
	# dt = 1

	# Initial process covariance matrix
	pos_var = 1
	vel_var = 0.001
	# initial_cov = np.array([
	# 	[pos_var, 0, 0, 0],
	# 	[0, pos_var, 0, 0],
	# 	[0, 0, vel_var, 0],
	# 	[0, 0, 0, vel_var]
	# ])
	initial_cov = pos_var * np.eye(18)
	# Initially computed values for the matrix were:
	# 	[2.5, -0.4, 7, 0],
	# 	[-0.4, 2.5, 0, 7],
	# 	[7, 0, 0.5, 0],
	# 	[0, 7, 0, 0.5]
	# But these were changed to the current values when tuning.
	# The off-diagonal elements showed no significant contribution
	# and reducing the vel var wrt pos var increased performance.

	# Measurement covariance matrix (or measurement error)
	measurement_cov = 5 * pos_var * np.eye(18)

	# State Transition Matrix
	# Constant velocity model
	a = np.eye(18)

	# Control Matrix
	# Apply constant velocity model
	b = np.array([1, 0, 0] * 6 + [0, 1, 0] * 6 + [0, 0, 1] * 6).reshape((18, 3))

	# Estimate Error Transform Matrix
	h = np.eye(18)

	# Read in data
	positions = read_data()

	# Store previous velocity for computing acceleration
	# prev_vel = updated_prev_vel = velocities[0]
	prev_pos = positions[0].reshape((6, 3))
	vel = np.array([0, 0, 0])
	# Store updated positions for plotting
	updated_positions = np.zeros(positions.shape)
	updated_positions[0, :] = positions[0, :]

	# Initial State Vector
	initial_state = np.atleast_2d(positions[0]).T

	# Create KalmanFilter object
	kalman = KalmanFilter(initial_state, initial_cov, measurement_cov, a, b, h)

	for i, pos in enumerate(positions[1:]):
		# Compute process and measurement vectors
		process = np.atleast_2d(vel).T
		measurement = np.atleast_2d(pos).T

		# Perform an iteration of the Filter, store updated position
		kalman.iterate(process, measurement)
		updated_positions[i + 1] = kalman.state[:, 0].T

		# Print updated position
		# print(f"---Iteration {i + 1}---")
		# print(f"Pos x: {kalman.state[0, 0]}, var: {kalman.process_cov[0, 0]}")
		# print(f"Pos y: {kalman.state[1, 0]}, var: {kalman.process_cov[1, 1]}")

		# Compute updated average velocity and print
		# updated_cur_vel = kalman.state[2:, 0].T
		# updated_avg_vel = (updated_cur_vel + updated_prev_vel) / 2
		# print(f"Vel x: {updated_avg_vel[0]}; var: {kalman.process_cov[2, 2]}")
		# print(f"Vel y: {updated_avg_vel[1]}; var: {kalman.process_cov[3, 3]}")

		# Reshape position to have x, y, and z cols
		cur_pos = pos.reshape((6, 3))
		# Compute velocity
		vel = np.mean(cur_pos - prev_pos, axis=0)

		# Update variables for next iteration
		prev_pos = cur_pos
	# prev_vel = cur_vel
	# updated_prev_vel = updated_cur_vel

	# Save position data to output file
	save_data(updated_positions, 'positions')
	# Plotting
	c = itertools.count()  # Generates unique numbers for the figure indices
	label_m = "Measured Trajectory"
	label_f = "Updated Trajectory"
	combined_title = "Combined Plot"
	plot_trajectory(positions[-1000:, :], label_m + " Partial", next(c))
	plot_trajectory(updated_positions[-1000:, :], label_f + " Partial", next(c))

	plot_trajectory(positions, label_m, next(c))
	plot_trajectory(updated_positions, label_f, next(c))
	plot_combined(
		positions[-1000:, :], updated_positions[-1000:, :], label_m, label_f, combined_title, next(c)
	)
	print(positions.shape, updated_positions.shape)

	print("Positions and trajectory plots saved to output folder.")


if __name__ == '__main__':
	main()
