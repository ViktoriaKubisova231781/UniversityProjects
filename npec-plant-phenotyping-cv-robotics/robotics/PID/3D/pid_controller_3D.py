import numpy as np

class PIDController3D:
    def __init__(self, kp, ki, kd):
        """
        Initializes a PID controller for 3D vector control.

        Parameters:
            kp (float or list of 3): Proportional gain(s)
            ki (float or list of 3): Integral gain(s)
            kd (float or list of 3): Derivative gain(s)
        """
        # Store PID gains as NumPy arrays for per-axis control
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)

        # Internal state for integral and derivative terms
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

    def compute(self, error, dt):
        """
        Compute the control signal based on the current error.

        Parameters:
            error (array-like): Current positional error [x, y, z]
            dt (float): Time delta between steps

        Returns:
            np.array: Control signal for [x, y, z] axis
        """
        error = np.array(error)

        # Update integral and derivative terms
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros(3)

        # PID output: sum of proportional, integral, and derivative responses
        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        # Store current error for next derivative calculation
        self.prev_error = error
        return output

    def reset(self):
        """
        Resets the integral and previous error values to zero.
        Useful when starting a new trial.
        """
        # Clear controller state
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
