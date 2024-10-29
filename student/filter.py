# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 
#From misc/params.py, load the following parameters: dt, q, dim_state.
from misc.params import dt, q, dim_state

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############

        return np.matrix([[1, 0, 0, dt,  0,  0],
                          [0, 1, 0,  0, dt,  0],
                          [0, 0, 1,  0,  0, dt],
                          [0, 0, 0,  1,  0,  0],
                          [0, 0, 0,  0,  1,  0],
                          [0, 0, 0,  0,  0,  1]])
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        
        # Process noise intensity parameter  q = params.q  
        # Time interval between predictions dt = params.dt 

        # Calculate elements of the Q matrix using dt
        q3 = (q / 3) * dt**3  # Position noise (third order)
        q2 = (q / 2) * dt**2  # Mixed position-velocity noise (second order)
        q1 = q * dt           # Velocity noise (first order)

        # Process noise covariance matrix Q
        return np.matrix([[q3, 0,  0,  q2, 0,  0],      # Noise in x and vx
                          [0,  q3, 0,  0,  q2, 0],      # Noise in y and vy
                          [0,  0,  q3, 0,  0,  q2],     # Noise in z and vz
                          [q2, 0,  0,  q1, 0,  0],      # Mixed noise x-vx
                          [0,  q2, 0,  0,  q1, 0],      # Mixed noise y-vy
                          [0,  0,  q2, 0,  0,  q1]])    # Mixed noise z-vz

        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        # Get current state
        x = track.x
        P = track.P

        # Build F and Q
        F = self.F() # State transition matrix
        Q = self.Q() # Process noise covariance matrix

        # Predict state x and estimation error covariance P
        x = F @ x 
        P = F @ P @ F.T + Q

        # Save x and P in track
        track.set_x(x)
        track.set_P(P)
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        # Get the measurement matrix H and residual (gamma)
        H = meas.sensor.get_H(track.x)
        gamma_val = self.gamma(track, meas)
        S_val = self.S(track, meas, H)

        # Calucalte Kalman gain
        K = track.P @ H.T @ np.linalg.inv(S_val)

        # Update state and cocariance P
        x = track.x + K @ gamma_val
        I = np.identity(dim_state) # Create an identity matrix with size dim_state
        P = (I - K @ H) @ track.P

        # Save state x and covariance P 
        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # Compute the residual: gamma = z - H * x
        ############

        hx = meas.sensor.get_hx(track.x)
        gamma = meas.z - hx
        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # Compute covariance of residual: S = H * P * H.T + R
        ############

        H = meas.sensor.get_H(track.x)
        S = H @ track.P @ H.T + meas.R
        return S
        
        ############
        # END student code
        ############ 