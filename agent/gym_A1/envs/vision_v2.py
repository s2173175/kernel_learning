import numpy as np
import matplotlib.pyplot as plt


class Vision:
    """
        V1.0.0 - 01/Aug/2021 by Zhaocheng Liu
        This model shines a sparse set of rays onto the floor with a maximum angle of 85 degrees. </br>
        The ray angles are regulated to form a uniform spaces matrix in the X-Y plane. </br>
        The rays have the same length which expands a sphere are space from the LiDAR sensor. </br>
        By default, the FoV is centered to (0, 0, -1) (xyz), which is pointing vertically downwards. </br>
        1. FoV_V_Max        : the maximum vertical Field-of-View angle (angle of the middle column of rays) </br>
        1. FoV_V_offset     : the vertical Field-of-View angle offset </br>
        1. FoV_H_Max        : the maximum horizontal Field-of-View angle (angle of the bottom row of rays) </br>
        1. FoV_H_offset     : the horizontal Field-of-View angle offset </br>
        1. rayLength        : maximum ray length </br>
        1. numRays_V        : number of rows </br>
        1. numRays_H        : number of columns
    """

    def __init__(self, FoV_V_Max, FoV_V_offset, FoV_H_Max, FoV_H_offset, rayLength, numRays_V, numRays_H):
        assert (numRays_V >= 0 and numRays_H >= 0 and not(numRays_V == 0 and numRays_H == 0)), \
            "Invalid number of rays, must have at least one ray"
        self.numRays_V = numRays_V
        self.numRays_H = numRays_H
        assert rayLength > 0, "The rayLength must be greater than zero!"
        self.rayLength = rayLength

        self.FoV_V_Max = FoV_V_Max
        self.FoV_V_offset = FoV_V_offset
        self.FoV_H_Max = FoV_H_Max
        self.FoV_H_offset = FoV_H_offset

        # Calculate the FoV according to the FoV angles
        self.FoV = np.clip(np.array([  # Maximum FoV
            -self.FoV_V_Max/2+self.FoV_V_offset,  # V_from  -> x min
            +self.FoV_V_Max/2+self.FoV_V_offset,  # V_to    -> x max
            -self.FoV_H_Max/2+self.FoV_H_offset,  # H_from  -> y min
            +self.FoV_H_Max/2+self.FoV_H_offset,  # H_to    -> y max
        ]), -np.pi/2+np.deg2rad(5), np.pi/2-np.deg2rad(5))

        # Calculate the polar angles of the boundaries
        self.polarAnglesBoundary = np.array([  # Polar angels of the corners of the FoV rectangle
            [self.FoV[1], self.FoV[3]],  # upper left point
            [self.FoV[1], self.FoV[2]],  # upper right point
            [self.FoV[0], self.FoV[3]],  # lower left point
            [self.FoV[0], self.FoV[2]],  # lower right point
        ])

        # cartesian coordinates of sparse points within the FoV plane (rectangle)
        imaginaryPlaneHeight = 1  # 1 metre below the LiDAR
        self.boundaryCoords = (np.array([imaginaryPlaneHeight] * 4) * np.tan(self.polarAnglesBoundary).T).T
        self.boundaryCoords = np.vstack([self.boundaryCoords.T, [-imaginaryPlaneHeight]*self.boundaryCoords.shape[0]]).T
        self.cartesianPlane = self.__getRayPlaneCoordinates(self.boundaryCoords)
        self.rayEndPoints = self.__normaliseRays(self.cartesianPlane, rayLength)

        if 0:  # Debug outputs
            print(np.rad2deg(self.FoV))
            print(np.rad2deg(self.polarAnglesBoundary))
            print(self.boundaryCoords)

    def __getRayPlaneCoordinates(self, boundaryCoords):
        """ Calculates the sparse coordinates in the X-Y plane, given
            the boundary coordinates of the detection plane range. (in local frame)
            Input: (4, 3) coordinates - plane
            Return: (N, 3) coordinates - plane
        """
        V_max, V_min = boundaryCoords[0][0], boundaryCoords[2][0]
        # V_max, V_min = self.cartesian[0][0], self.cartesian[2][0]
        x_values = []
        if self.numRays_V == 1:
            x_values = [(V_max + V_min)/2]
        elif self.numRays_V == 2:
            x_values = [V_max, V_min]
        elif self.numRays_V >= 3:
            y_diff = (V_max - V_min)/(self.numRays_V-1)
            x_values = [V_max - y_diff*i for i in range(self.numRays_V)]
        else:
            raise Exception(f"Invalid number of vertical rays: {self.numRays_V}")

        H_min, H_max = boundaryCoords[1][1], boundaryCoords[0][1]
        # H_min, H_max = self.cartesian[1][1], self.cartesian[0][1]
        y_values = [(H_max + H_min)/2] * len(x_values)
        if self.numRays_H == 1:
            y_values = [(H_max + H_min)/2]
        elif self.numRays_H == 2:
            y_values = [H_max, H_min]
        elif self.numRays_H >= 3:
            y_diff = (H_max - H_min)/(self.numRays_H-1)
            y_values = [H_max - y_diff*i for i in range(self.numRays_H)]
        else:
            raise Exception(f"Invalid number of horizontal rays: {self.numRays_H}")

        res = np.array([[[x, y, boundaryCoords[0][2]] for y in y_values] for x in x_values])
        res = res.reshape(res.shape[0] * res.shape[1], res.shape[2])
        return res

    def __normaliseRays(self, cartesianPlaneRays, rayLength):
        """ Normalise the rays to have the same maximum length. 
            Plane -> sphere, but the plane projection is still a uniform matrix. 
            Input: (N, 3) coordinates - plane
            Return: (N, 3) coordinates - sphere
        """
        old_rayLengths = np.sqrt(np.diag(np.dot(cartesianPlaneRays, cartesianPlaneRays.T)))
        cartesian_sphere = (cartesianPlaneRays.T / old_rayLengths * rayLength).T
        return cartesian_sphere

    @property
    def getRayLengths(self):
        return self.rayLength

    @property
    def getNumberRays(self):
        return self.rayEndPoints.shape[0]

    def getRayBatchLocal(self):
        return self.rayEndPoints

    def getRayBatchWorld(self, translation, rotMat):
        """Return (origin, transformed ray end points)."""
        origin = np.array([translation]*self.getNumberRays)
        return origin, (rotMat @ self.getRayBatchLocal().T).T + origin

