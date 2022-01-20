# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *
import numpy as np
import scipy
from util import *
from scipy import signal

import numpy as np
import scipy
from util import *
from scipy import signal
# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        self.Fmax = 15736

        self.prev_e1 =0
        self.prev_e2 = 0

        self.delta = 0
        self.deltamax = np.pi / 6


   
        self.integral_error_force = 0

        self.diff_error_force = 0
        self.prev_err_psi = 0
        self.prev_err_force = 0

        # Add additional member variables according to your need here.

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 
        timestep = 50

        _, min_idx = closestNode(X, Y, trajectory)

        # print(trajectory.shape)
        if timestep + min_idx >= trajectory.shape[0]:
            timestep = 0

        # print("Nearest way point :", min_idx)
        Xreq = trajectory[min_idx + timestep, 0]
        Yreq = trajectory[min_idx + timestep, 1]

        # print(Xreq-X);
        # print(Yreq-Y)
        psireq = np.arctan2(Yreq - Y, Xreq - X)
        # ---------------|Lateral Controller|-------------------------

        v =xdot
        A = np.array([[0, 1, 0, 0],
                      [0, -42.359 / v, 42.359, -3.388 / v],
                      [0, 0, 0, 1],
                      [0, -0.2475 / v, 0.2475, -6.7 / v]])

        
        B = np.expand_dims(np.array([0, 21.1795, 0, 2.398]).T, axis=1)
        P = np.array([-10, -6, -2, 0])
        #print(P)
        #print(A.shape)
        #print(B.shape)
        #print(P.shape)
        z = signal.place_poles(A, B, P, method="YT")
        K = z.gain_matrix
        #print(K.shape)
        e1 = np.sqrt((X - Xreq)**2 + (Y - Yreq)**2)
        e2 = wrapToPi(psi - psireq)

        # e1dot = (e1 - self.prev_e1)/delT
        # e2dot = (e2 - self.prev_e2)/delT

        e1dot = ydot +  v*e2
        e2dot = psidot


        self.prev_e1 = e1
        self.prev_e2 = e2

        e = np.array([[e1], [e1dot], [e2], [e2dot]])

        delta = -np.matmul(K, e)
        delta = np.asscalar(delta)
        #self.delta= delta
        # ---------------|Longitudinal Controller|-------------------------

        force_err = np.sqrt((Yreq - Y) ** 2 + (Xreq - X) ** 2) / (delT)
        self.integral_error_force += force_err * delT
        self.diff_error_force += (force_err - self.prev_err_force) / delT
        self.prev_err_force = force_err
        # print("The force error is :", force_err)

        kp_for = 3;
        ki_for = 0;
        kd_for = 0
        F = kp_for * force_err + ki_for * self.integral_error_force + kd_for * self.diff_error_force

        if (xdot < 0.5):
            xdot = 0.5
        if(F > self.Fmax):
            F =self.Fmax
        if(delta > self.deltamax):
            delta = self.deltamax
        if (delta < -self.deltamax):
            delta = -self.deltamax

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
