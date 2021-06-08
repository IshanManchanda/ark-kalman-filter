import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

from kalman import KalmanFilter


def read_data():
	# Path for input file
	base_path = os.path.dirname(os.path.abspath(__file__))
	file_path = os.path.join(base_path, 'data/Data3.csv')

	# Read in data to a numpy array
	with open(file_path) as csv_file:
		data = np.array(list(csv.reader(csv_file)), dtype=np.float64)

	# TODO: Change data shape to new format
	# data = data.reshape((-1, 4))
	# idx = [0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 16, 2, 5, 8, 11, 14, 17]
	return data


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
	ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=range(x), cmap='Greens')

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
		data1[:, 0], data1[:, 1], data1[:, 2], c=range(x),
		cmap='Reds', edgecolors='none', alpha=0.05
	)
	ax.scatter3D(data2[:, 0], data2[:, 1], data2[:, 2], c=range(x), cmap='viridis')

	base_path = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(base_path, f'out/{title}.png')
	figure.savefig(path)


def main():
	# dt computed while exploring data by averaging time periods
	# over the given dataset after removing outliers
	dt = 1
	dt2 = dt * dt

	# Initial process covariance matrix
	pos_var = 1
	initial_vel_var = 1e-9
	initial_acc_var = 1e-9
	initial_cov = np.array([
		[pos_var, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, pos_var, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, pos_var, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, initial_vel_var, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, initial_vel_var, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, initial_vel_var, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, initial_acc_var, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, initial_acc_var, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, initial_acc_var],
	])

	# Measurement covariance matrix (or measurement error)
	# measurement_cov = 5 * pos_var * np.eye(18)
	vel_var = 0.005
	measurement_cov = np.array([
		[pos_var, 0, 0, 0, 0, 0],
		[0, pos_var, 0, 0, 0, 0],
		[0, 0, pos_var, 0, 0, 0],
		[0, 0, 0, vel_var, 0, 0],
		[0, 0, 0, 0, vel_var, 0],
		[0, 0, 0, 0, 0, vel_var],
	])
	# measurement_cov = np.array([
	# 	[pos_var, 0, 0],
	# 	[0, pos_var, 0],
	# 	[0, 0, pos_var],
	# ]) / 10

	# State Transition Matrix
	# Constant acceleration model
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
	# Constant acceleration model
	b = np.zeros((9, 1))

	# Estimate Error Transform Matrix
	# h = np.array([
	# 	[1, 0, 0, 0, 0, 0],
	# 	[0, 1, 0, 0, 0, 0],
	# 	[0, 0, 1, 0, 0, 0],
	# 	[0, 0, 0, 1, 0, 0],
	# 	[0, 0, 0, 0, 1, 0],
	# 	[0, 0, 0, 0, 0, 1],
	# 	[0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0],
	# 	[0, 0, 0, 0, 0, 0],
	# ])
	h = np.array([
		[1, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 1, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 1, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 1, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 1, 0, 0, 0],
	])
	# h = np.array([
	# 	[1, 0, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 1, 0, 0, 0, 0, 0, 0, 0],
	# 	[0, 0, 1, 0, 0, 0, 0, 0, 0],
	# ])

	# Process noise matrix
	q = np.array([
		[1, 0, 0, 2, 0, 0, 2, 0, 0],
		[0, 1, 0, 0, 2, 0, 0, 2, 0],
		[0, 0, 1, 0, 0, 2, 0, 0, 2],
		[2, 0, 0, 4, 0, 0, 4, 0, 0],
		[0, 2, 0, 0, 4, 0, 0, 4, 0],
		[0, 0, 2, 0, 0, 4, 0, 0, 4],
		[2, 0, 0, 4, 0, 0, 4, 0, 0],
		[0, 2, 0, 0, 4, 0, 0, 4, 0],
		[0, 0, 2, 0, 0, 4, 0, 0, 4],
	]) * 0.0000000001 ** 2

	# Read in data
	positions = read_data()
	# data = read_data()
	# drone_positions = data[:, :3]
	# tower_positions = data[:, 3:]

	# Store previous velocity for computing acceleration
	# prev_vel = updated_prev_vel = velocities[0]
	# prev_pos = positions[0].reshape((6, 3))
	prev_pos = positions[0, 3:].reshape((5, 3))
	# print(positions[0, 3:])
	# print(prev_pos)
	# return
	vel = np.array([0, 0, 0])
	# Store updated positions for plotting
	# updated_positions = np.zeros(positions.shape)
	# updated_positions[0, :] = positions[0, :]
	updated_positions = np.zeros(positions[:, :3].shape)
	updated_positions[0] = positions[0, :3]

	# Initial State Vector
	# initial_state = np.atleast_2d(positions[0]).T
	# initial_state = np.vstack((np.atleast_2d(positions[0, :3]).T, [[0], [0], [0]]))
	initial_state = np.atleast_2d(np.hstack((positions[0, :3], [0] * 6))).T

	# Create KalmanFilter object
	kalman = KalmanFilter(
		initial_state, initial_cov, measurement_cov, a, b, h, q
	)

	for i, pos in enumerate(positions[1:]):
		cur_pos = pos[3:].reshape((5, 3))
		vel = np.mean(cur_pos - prev_pos, axis=0) / dt

		# Compute process and measurement vectors
		# process = np.atleast_2d(vel).T
		process = np.zeros(1)
		measurement = np.atleast_2d(np.hstack((pos[:3], vel))).T
		# measurement = np.atleast_2d(pos[:3]).T

		# Perform an iteration of the Filter, store updated position
		kalman.iterate(process, measurement)
		updated_positions[i + 1] = kalman.state[:, 0].T[:3]

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
		# cur_pos = pos.reshape((6, 3))
		# Compute velocity
		# vel = np.mean(cur_pos - prev_pos, axis=0)

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
	plot_trajectory(positions[-200:, :3], label_m + " Partial", next(c))
	plot_trajectory(updated_positions[-200:, :3], label_f + " Partial", next(c))

	plot_trajectory(positions, label_m, next(c))
	plot_trajectory(updated_positions, label_f, next(c))
	plot_combined(
		positions[:, :3], updated_positions[:, :3], label_m, label_f, combined_title, next(c)
	)
	print(positions.shape, updated_positions.shape)

	print("Positions and trajectory plots saved to output folder.")


if __name__ == '__main__':
	main()
