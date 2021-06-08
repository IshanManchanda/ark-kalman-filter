import numpy as np


class KalmanFilter:
	def __init__(self, state, process_cov, measurement_cov, a, b, h, q):
		self.state = state  # Initial State

		# Initial Process Covariance Matrix and Measurement Covariance Matrix
		self.process_cov = process_cov
		self.measurement_error = measurement_cov

		# Transform Matrices
		self.a = a  # State Transition Matrix
		self.b = b  # Control Matrix
		self.h = h  # Estimate Error Transform Matrix
		self.q = q  # Process Noise Matrix

		# print("State", state.shape, state.dtype)
		# print("Pcov", process_cov.shape)
		# print("mcov", measurement_cov.shape)
		# print("a", a.shape)
		# print("b", b.shape)
		# print("h", h.shape)

		# Declare additional members
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

		# self.predicted_process_cov += 0.01 * np.eye(18)
		# self.predicted_process_cov *= 1.2
		# print("predstate", self.predicted_state.shape)
		# print("predpcov", self.predicted_process_cov.shape)

	def compute_gain(self):
		# Compute Kalman Gain using the errors
		estimate_error = self.predicted_process_cov @ self.h.T
		total_error = self.h @ estimate_error + self.measurement_error
		self.gain = estimate_error @ np.linalg.inv(total_error)
		# print("esterr", estimate_error.shape)
		# print("terr", total_error.shape)
		# print("gain", self.gain.shape)

	def update_state(self, measurement):
		# We use the Joseph Form Update Equation as the simplified equation is
		# numerically unstable.
		# Intuition: Floating point errors in the subtraction step could lead to
		# negative values for variables which should be non-negative,
		# which can be deadly for the filter's accuracy.
		factor = np.eye(self.h.shape[1]) - (self.gain @ self.h)
		self.process_cov = factor @ self.predicted_process_cov @ factor.T
		self.process_cov += self.gain @ self.measurement_error @ self.gain.T

		# Testing shows differences of magnitude e-16 (max ~4.5e-16) between the
		# simplified equation and the Joseph form.
		# print(self.process_cov - factor @ self.predicted_process_cov)

		# Compute residual and update state
		residual = measurement - self.h @ self.predicted_state
		self.state = self.predicted_state + (self.gain @ residual)
