from geometry_msgs.msg import Quaternion

import numpy as np
from numpy.typing import NDArray


def rotmat2q(T: NDArray) -> Quaternion:
    # Function that transforms a 3x3 rotation matrix to a ros quaternion representation
    

    if T.shape != (3, 3):
        raise ValueError

    m00, m01, m02 = T[0, 0], T[0, 1], T[0, 2]
    m10, m11, m12 = T[1, 0], T[1, 1], T[1, 2]
    m20, m21, m22 = T[2, 0], T[2, 1], T[2, 2]

    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    # Normalize quaternion
    norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    q = Quaternion()
    q.x, q.y, q.z, q.w = x, y, z, w

    return q

    # TODO: implement this



# T = np.array(
# [[ 0.284, -0.958,  0.042],
#  [ 0.959,  0.282, -0.015],
#  [ 0.002,  0.045,  0.999]]
# )

# print(rotmat2q(T)) # (0.0187, 0.0125, 0.5985, 0.8008)

