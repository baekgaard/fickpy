''' 

    Functions that can be useful in pre-processing eye tracking data converting between 
    Fick angles, 3D vectors and visual angles

    Written by Per Baekgaard <pgba@dtu.dk> 2022

'''

import numpy as np
import collections

# Ficks Angle Conversions
def vec2ficks(x, y, z):
    '''
    Calculate ficks angles, corresponding to the given vector, conveniently labelled azimuth and polar angle
    
    Ficks angles are based on nested gimbals, where the outer gimbal is rotated first along the 
    vertical axis (az) and then along the nested and now rotated horizontal (shoulder) axis (pol). Ficks is normally
    (as here) descibed using passive rotations, where the second rotation is around an axis rotated 
    by the first rotation. It can, however, also be described using active rotations, where the rotation axes are
    fixed in world coordinates; to do so the two rotations are applied in the opposite direction.
    
    Thus, in terms of world coordinates, calculating the resulting gaze vector using the two Ficks
    angles can be calculated as follows:
    
        (x, y, z) = np.dot(rotmat.R('y', az), np.dot(rotmat.R('x', pol), np.array([0, 0, 1])))
        
    or equivalently
    
        (x, y, z) = np.dot(np.matmul(rotmat.R('y', az), rotmat.R('x', pol)), np.array([0, 0, 1])))
        
    where x is the shoulder axis, y the vertical axis and z represents the line of sight.
    
    (Haslwanter, T. (1995). Mathematics of three-dimensional eye rotations. Vision research, 35(12), 1727-1739.)

    Note that a vector cannot represent a torsion component, so only 2 Ficks angles are returned here

    Note that the calculation here is identical to converting to "adapted" spherical coordinates, for 
    which the formula used is based on e.g. https://web.physics.ucsb.edu/~fratus/phys103/Disc/disc_notes_3_pdf.pdf, 
    just adapted to conventional gaze concepts where the polar angle is 0 at the center coordinate
    and azimuth increases in the direction of x, counting from z
    
    IMPORTANT NOTE: If used with a RHS where X is positive towards the right and Y is positive downwards and Z is line
    of sight positive forward, then a rotation towards the right is positive but a rotation downwards is also positive;
    this is not "usual" but has to do with how it's currently being used for eye tracking where the apparent X 
    and Y coordinates gets translated into apparent angles in the same direction.

    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(x , z) # Note: If we had done az = 90 - np.arccos(x / r) we would return visual angles
    pol = np.arccos(y / r)

    return np.rad2deg(az), 90 - np.rad2deg(pol)

def ficks2vec(az, pol):
    '''
    Calculate a unit vector from Ficks Angle coordinates, see vec2ficks for more comments
    '''
    
    z = np.sin(np.deg2rad(90-pol)) * np.cos(np.deg2rad(az))
    x = np.sin(np.deg2rad(90-pol)) * np.sin(np.deg2rad(az))
    y = np.cos(np.deg2rad(90-pol))

    return x, y, z


def visang2vec(h, v, z=1, norm=True):
    '''
    Calculate a directional vector x,y,z from visual angles Ah and Av
    
    The vector returned is the intercept with the direction and a plane at z=z
    
    The formula below has been derived from the vec2visang formula and also 
    deals correctly with negative angles, but breaks down when z~=0(!)
    
    Also, even though h is allowed to rotate fully, v is capped to +/- 90 degrees
    '''

    th2 = np.tan(np.deg2rad(h))**2
    tv2 = np.tan(np.deg2rad(v))**2
    th2tv2 = th2*tv2

    x2 = (th2 + th2tv2) / (1 - th2tv2) * z**2
    y2 = (tv2 + th2tv2) / (1 - th2tv2) * z**2
    
    x = np.sign((h+180)%360-180)*np.sqrt(np.abs(x2))
    y = np.sign((v+180)%360-180)*np.sqrt(np.abs(y2))
    
    if isinstance(h, (collections.abc.Sequence, np.ndarray)):
        z = np.ones(x.shape)
        z[np.where((h+90)%360>180)] = -1
    else:    
        z = z * (-1 if (h+90)%360>180 else 1)
    
    if (norm):
        r = np.sqrt(np.abs(x2) + np.abs(y2) + z**2)
    else:
        r = 1
        
    return x/r, y/r, z/r
