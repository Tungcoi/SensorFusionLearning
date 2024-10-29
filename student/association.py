# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []

    def associate(self, track_list, meas_list, KF):
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############

        # Reset the lists for unassigned tracks and measurements
        self.unassigned_tracks = list(range(len(track_list)))
        self.unassigned_meas = list(range(len(meas_list)))

        # Initialize the association matrix with infinity values
        self.association_matrix = np.full((len(track_list), len(meas_list)), np.inf)

        # Compute the association matrix using Mahalanobis distance
        for i, track in enumerate(track_list):
            for j, measurement in enumerate(meas_list):
                # Calculate the Mahalanobis distance between the track and the measurement
                dist = self.MHD(track, measurement, KF)
                #print(f' i = {i}, j = {j}, dist = {dist}')
                # Check if the distance is within the gating threshold
                if self.gating(dist, measurement.sensor):
                    self.association_matrix[i, j] = dist
        #print(f'associate, matrix = {self.association_matrix}')
        ############
        # END student code
        ############ 

    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # Find the indices of the minimum value in the association matrix
        #print(f'get_closest_track_and_meas, matrix = {self.association_matrix}')
        if np.min(self.association_matrix) == np.inf:
            print("association_matrix min is infinity!!!")
            return np.nan, np.nan  # No valid associations left

        # Get the indices of the closest track-measurement pair
        ind_track, ind_meas = np.unravel_index(np.argmin(self.association_matrix), self.association_matrix.shape)

        # Map the indices to the original track and measurement
        update_track = self.unassigned_tracks[ind_track]
        update_meas = self.unassigned_meas[ind_meas]

        # Remove the track and measurement from the unassigned lists
        self.unassigned_tracks.pop(ind_track)
        self.unassigned_meas.pop(ind_meas)

        # Delete the corresponding row and column from the association matrix
        self.association_matrix = np.delete(self.association_matrix, ind_track, axis=0)
        self.association_matrix = np.delete(self.association_matrix, ind_meas, axis=1)
            
        ############
        # END student code
        ############ 
        print(f'update track = {update_track}, update_meas = {update_meas}')
        return update_track, update_meas    
        

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        
        # Compute the gating threshold using chi-squared distribution for the given degrees of freedom
        threshold = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
        return MHD < threshold
        
        ############
        # END student code
        ############ 
        

    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
       
        # Compute the residual between the measurement and the predicted state
        H = meas.sensor.get_H(track.x)
        gamma = meas.z - meas.sensor.get_hx(track.x)
       
        # Compute the residual covariance matrix S
        S = KF.S(track, meas, H)

        # Calculate the Mahalanobis distance using the formula
        MHD_value = gamma.T @ np.linalg.inv(S) @ gamma
        
        return float(MHD_value)
        
        ############
        # END student code
        ############ 

    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)