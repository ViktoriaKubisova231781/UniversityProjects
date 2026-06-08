import numpy as np  

class PIDController:
    """
    Standard 1D PID Controller.

    Attributes:
        kp (float): Proportional gain
        ki (float): Integral gain
        kd (float): Derivative gain
    """

    def __init__(self, kp, ki, kd):
        # Initialize PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # Initialize integral and previous error
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        # Update integral term
        self.integral += error * dt
        # Calculate derivative term, safeguard against division by zero
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        # Compute PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        # Update previous error
        self.prev_error = error
        return output

    def reset(self):
        # Reset the integral and derivative history
        self.integral = 0
        self.prev_error = 0