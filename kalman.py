import numpy as np


class KalmanFilter:
	def __init__(self, state, process_cov, measurement_cov, a, b, h, q):
		self.state = state  # Initial State

		# Covariance matrices
		self.process_cov = process_cov
		self.measurement_error = measurement_cov
		self.q = q  # Process Noise Matrix

		# Transform Matrices
		self.a = a  # State Transition Matrix
		self.b = b  # Control Matrix
		self.h = h  # Estimate Error Transform Matrix

		# Declare additional member variables
		self.predicted_state = self.predicted_process_cov = self.gain = None

	def iterate(self, process, measurement):
		# Perform the 3-step process, return state and covariance matrix
		self.predict_state(process)
		self.compute_gain()
		self.update_state(measurement)
		return self.state, self.process_cov

	def predict_state(self, process):
		# Predict next state using previous state data
		self.predicted_state = (self.a @ self.state) + (self.b @ process)
		self.predicted_process_cov = self.a @ self.process_cov @ self.a.T

		# Add process noise
		self.predicted_process_cov += self.q

	def compute_gain(self):
		# Compute Kalman Gain using the errors
		estimate_error = self.predicted_process_cov @ self.h.T
		total_error = self.h @ estimate_error + self.measurement_error
		self.gain = estimate_error @ np.linalg.inv(total_error)

	def update_state(self, measurement):
		# We use the Joseph Form of the Process Covariance Update Equation
		# as the simplified version is numerically unstable.
		# Intuition: Floating point errors in the subtraction step could lead
		# to negative values for variables which should be non-negative,
		# which can be deadly for the filter's accuracy.
		factor = np.eye(self.h.shape[1]) - (self.gain @ self.h)
		self.process_cov = factor @ self.predicted_process_cov @ factor.T
		self.process_cov += self.gain @ self.measurement_error @ self.gain.T

		# Testing shows differences of magnitude e-22 (max ~4.2e-22)
		# between the simplified equation and the Joseph form,
		# and no losses in accuracy were observed.
		# However, we continue to use the Joseph form as it generalizes better.
		# error = self.process_cov - factor @ self.predicted_process_cov
		# print(np.abs(error))

		# Compute residual and update state
		residual = measurement - self.h @ self.predicted_state
		self.state = self.predicted_state + (self.gain @ residual)
