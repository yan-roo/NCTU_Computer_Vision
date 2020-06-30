import numpy as np
import cv2
import glob

def calibrateCamera(path):
    # Camera calibration
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # (8,6) is for the given testing images.
    # If you use the another data (e.g. pictures you take by your smartphone),
    # you need to set the corresponding numbers.
    corner_x = 7
    corner_y = 7
    objp = np.zeros((corner_x*corner_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = sorted(glob.glob(f'./{path}/*.jpg'))

    # Step through the list and search for chessboard corners
    print('Start finding chessboard corners...')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Find the chessboard corners
        print('find the chessboard corners of',fname)
        ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    print('Camera calibration...')
    img_size = (img.shape[1], img.shape[0])
    # You need to comment these functions and write your calibration function from scratch.
    # Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
    # In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
    H_set = []
    for i in range(len(objpoints)):
        H_coef = np.zeros((corner_x*corner_y*2, 9))
        idx = 0
        img_points = imgpoints[i].reshape(-1,2)
        for obj, img in zip(objpoints[i], img_points):
            H_coef[idx*2, :] = [-obj[0], -obj[1], -1, 0, 0, 0, obj[0]*img[0], obj[1]*img[0], img[0]]
            H_coef[idx*2+1, :] = [0, 0, 0, -obj[0], -obj[1], -1, obj[0]*img[1], obj[1]*img[1], img[1]]
            idx += 1

        u, s, vh = np.linalg.svd(H_coef, full_matrices=False)
        H = vh.T[:,-1]
        if H[-1] < 0:
            H *= -1
        H /= H[-1]
        H_set.append(H.reshape(3,3))

    v = np.zeros((2 * len(H_set), 6))
    for i, h in enumerate(H_set):
        v[2*i, :] = [
            h[0,1] * h[0,0],
            h[0,1] * h[1,0] + h[1,1] * h[0,0],
            h[0,1] * h[2,0] + h[2,1] * h[0,0],
            h[1,1] * h[1,0],
            h[1,1] * h[2,0] + h[2,1] * h[1,0],
            h[2,1] * h[2,0]
            ]
        v[2*i+1, :] = [
            h[0,0]**2 - h[0,1]**2,
            2 * (h[0,0] * h[1,0] - h[0,1] * h[1,1]),
            2 * (h[0,0] * h[2,0] - h[0,1] * h[2,1]),
            h[1,0]**2 - h[1,1]**2,
            2 * (h[1,0] * h[2,0] - h[1,1] * h[2,1]),
            h[2,0]**2 - h[2,1]**2
        ]

    u, s, vh = np.linalg.svd(v, full_matrices=False)

    b = vh.T[:, -1]

    # Negative Diagonal
    if (b[0] < 0 or b[3] < 0 or b[5] < 0):
        b = b * (-1)

    B = np.array([
        [b[0], b[1], b[2]],
        [b[1], b[3], b[4]],
        [b[2], b[4], b[5]]
        ])

    # B = K^(-T) * K^(-1)
    # Cholesky decomposition to solve K
    l = np.linalg.cholesky( B )

    intrinsic = np.linalg.inv(l.T)

    # divide intrinsic by its scale
    intrinsic = intrinsic / intrinsic[2,2]

    extrinsic_matrices = np.zeros((len(H_set), 6))

    for i, h in enumerate(H_set):

        intrinsic_inverse = np.linalg.inv(intrinsic)

        lambda_value = 1 / np.linalg.norm(np.matmul(intrinsic_inverse, h[:, 0]))

        r1 = np.matmul(lambda_value * intrinsic_inverse, h[:, 0])
        r2 = np.matmul(lambda_value * intrinsic_inverse, h[:, 1])
        r3 = np.cross(r1, r2)

        t = np.matmul(lambda_value * intrinsic_inverse, h[:, 2])

        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[:,0] = r1
        rotation_matrix[:,1] = r2
        rotation_matrix[:,2] = r3

        r_rodrigues, _ = cv2.Rodrigues(rotation_matrix)

        extrinsic_matrices[i,:] = [r_rodrigues[0], r_rodrigues[1], r_rodrigues[2], t[0], t[1], t[2]]
        
    return intrinsic