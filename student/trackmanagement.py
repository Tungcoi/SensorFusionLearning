# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import collections

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 
from pprint import pprint

class Track:
    '''Track class with state, covariance, id, score'''
    INITIALIZED = 'initialized'
    TENTATIVE = 'tentative'
    CONFIRMED = 'confirmed'
    def __init__(self, meas, id):
        print('creating track no.', id)
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3] # rotation matrix from sensor to vehicle coordinates
        
        ############
        # Step 2: initialization:
        # - replace fixed track initialization values by initialization of x and P based on 
        # unassigned measurement transformed from sensor to vehicle coordinates
        # - initialize track state and track score with appropriate values
        ############

        # Transform measurement coordinates from sensor frame to vehicle frame
        pos_vehicle = M_rot @ meas.z

        # Initialize the state vector x (position + velocity)
        self.x = np.zeros((params.dim_state, 1))  # Vector trạng thái (position + velocity)
        self.x[:3] = pos_vehicle   # Set the measured position in the state vector

        # Initialize the state covariance matrix P with appropriate uncertainties
        self.P = np.eye(params.dim_state)
        self.P[3, 3] = params.sigma_p44 ** 2 # Variance for velocity in x-direction
        self.P[4, 4] = params.sigma_p55 ** 2  # Variance for velocity in y-direction
        self.P[5, 5] = params.sigma_p66 ** 2  # Variance for velocity in z-direction

         # Initialize track state and score
        self.state = Track.INITIALIZED
        self.score = 1. / params.window # Initial score based on window size
        
        ############
        # END student code
        ############ 
               
        # other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw =  np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        self.t = meas.t

    def set_x(self, x):
        self.x = x
        
    def set_P(self, P):
        self.P = P  
        
    def set_t(self, t):
        self.t = t  
        
    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            M_rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        
        
###################        

class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''
    def __init__(self):
        self.N = 0 # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []
        
    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):  
        ############
        # TODO Step 2: implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############
        
        
        # Loop through all unassigned tracks
        for i in unassigned_tracks:
            if not(0 <= i <len(self.track_list)):
                print(f'item {i} not in track_list {self.track_list}, \n unassigned_tracks = {unassigned_tracks}')
                continue
            track = self.track_list[i]
            print(f'unassigned_tracks begin id = {track.id}, score = {track.score}, state = {track.state}')
            if meas_list:
                if meas_list[0].sensor.in_fov(track.x):
                    print(f"Track {track.id} in FOV. -> reduce")
                    track.score -= 1. / params.window
                else:
                    print(f"Track {track.id} is not in FOV.")
            else:
                print(f"meas_list is empty!")
                    
                    
            track.score = max(track.score, 0.0)
            print(f'unassigned_tracks end id = {track.id}, score = {track.score}, state = {track.state}')
            
            # Delete the track if certain conditions are met
            if track.state == Track.CONFIRMED and track.score < params.delete_threshold:
                self.delete_track(track)
            elif track.state == Track.TENTATIVE and track.score < 0.3:
                self.delete_track(track)
            elif track.P[0, 0] > params.max_P or track.P[1, 1] > params.max_P:
                self.delete_track(track)

        ############
        # END student code
        ############ 
            
        # initialize new track with unassigned measurement
        for j in unassigned_meas: 
            if meas_list[j].sensor.name == 'lidar': # only initialize with lidar measurements
                self.init_track(meas_list[j])
            
    def addTrackToList(self, track):
        print(f'addTrackToList track no. {track.id} with score {track.score}')
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        print(f'Initializing new track with measurement {meas.z}')
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print(f'Deleting track no. {track.id} with score {track.score}')
        self.track_list.remove(track)
        
    def handle_updated_track(self, track):      
        ############
        # Step 2: implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############
        print(f'handle_updated_track begin id = {track.id}, score = {track.score}')
        # Increase the track score when updated
        track.score += 1. / params.window
        track.score = min(track.score, 1.0) # Ensure score does not exceed 1.0

        # Update the track state based on the new score
        if track.state == Track.TENTATIVE and track.score >= params.confirmed_threshold:
            track.state = Track.CONFIRMED # Set to 'confirmed' if score meets threshold
        elif track.state == Track.INITIALIZED and track.score >= 0.3:
            track.state = Track.TENTATIVE  # Move to 'tentative' state
        
        print(f'handle_updated_track end id = {track.id}, score = {track.score}')

        ############
        # END student code
        ############ 