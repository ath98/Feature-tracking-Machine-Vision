import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import glob

# Refrenced from lab 9

def extract_frames(filename, frames):
    result = {}
    camera = cv2.VideoCapture(filename)
    last_frame = max(frames)
    frame = 0
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            break
        if frame in frames:
            result[frame] = img
        frame += 1
        if frame > last_frame:
            break
    return result

# Refrenced from lab 9

def calculate_epipoles(F): # Calculate epipoles
    U, S, V = np.linalg.svd(F) # SVD
    e1 = V[2, :]# Extract epipoles
    U, S, V = np.linalg.svd(F.T)
    e2 = V[2, :]
    return e1, e2

# Refrence from lab 10

def track_extract_frames(filename):# Extract tracks from video
    camera = cv2.VideoCapture(filename)# Open video
    # initialise features to track
    while camera.isOpened():# While video is open
        ret, img = camera.read()# Read frame
        if ret:# If frame is read correctly
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            p0 = cv2.goodFeaturesToTrack(new_img, 200, 0.3, 7)# Find features
            break
    # initialise tracks
    index = np.arange(len(p0))# Create index
    tracks = {}
    for i in range(len(p0)):# For each feature
        tracks[index[i]] = {0: p0[i]}

    frame = 0
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            break
        frame += 1
        old_img = new_img
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # calculate samples_outtical flow
        if len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_img, new_img, p0, None)# Calculate optical flow

            # visualise points
            for i in range(len(st)):
                if st[i]:# If feature is tracked
                    cv2.circle(img, (int(p1[i, 0, 0]), int(p1[i, 0, 1])), 2, (0, 0, 255),
                               2)# Draw circle
                    cv2.line(img, (int(p0[i, 0, 0]), int(p0[i, 0, 1])), (int(p0[i, 0, 0] +
                            (p1[i][0, 0]-p0[i, 0, 0])*5), int(p0[i, 0, 1]+(p1[i][0, 1]-p0[i, 0, 1])*5)),
                             (0, 0, 255), 2)# Draw line

            p0 = p1[st == 1].reshape(-1, 1, 2)# Update features
            index = index[st.flatten() == 1]# Update index

        # refresh features, if too many lost
        if len(p0) < 100:
            new_p0 = cv2.goodFeaturesToTrack(new_img, 200-len(p0), 0.3, 7)# Find new features
            for i in range(len(new_p0)):
                if np.min(np.linalg.norm((p0 -
                                          new_p0[i]).reshape(len(p0), 2), axis=1)) > 10:
                    p0 = np.append(p0, new_p0[i].reshape(-1, 1, 2), axis=0)# Add new features
                    index = np.append(index, np.max(index)+1)#
        # update tracks
        for i in range(len(p0)):# For each feature
            if index[i] in tracks:# If feature is tracked
                tracks[index[i]][frame] = p0[i]# Update track
            else:
                tracks[index[i]] = {frame: p0[i]}
        # visualise last frames of active tracks
        for i in range(len(index)):
            for f in range(frame-20, frame):
                if (f in tracks[index[i]]) and (f+1 in tracks[index[i]]):
                    cv2.line(img,
                             (int(tracks[index[i]][f][0, 0]), int(tracks[index[i]][f]
                                                                  [0, 1])),
                             (int(tracks[index[i]][f+1][0, 0]), int(tracks[index[i]]
                                                                    [f+1][0, 1])),
                             (0, 255, 0), 1)
        cv2.imshow('frame', img)
        cv2.waitKey(0)
    return tracks, frame


def task1():
    # Loading data
    images = glob.glob('Assignment_MV_02_calibration/*.png')
    images.sort()

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((35, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

    objpoints = []  # 3d array points
    imgpoints = []  # 2d array points
    for img in images:
        img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 5), corners2, ret)
            cv2.imshow('img',img)
            cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    imageDetails = {
        'principal_point': mtx[0, 2], 'focal_length': mtx[0, 0], 'aspect_ratio': mtx[1, 1] / mtx[0, 0]}
    for i in imageDetails.items():
        print(i)

    images = extract_frames("Assignment_MV_02_video.mp4", [1]) # Extract frames from video
    img = images[1].copy() # Copy image
    next = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)# Convert to grayscale
    corners = cv2.goodFeaturesToTrack(next, 200, 0.3, 7)# Find corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)# Termination criteria
    p0 = cv2.cornerSubPix(next, np.float32(corners),
                          (11, 11), (-1, -1), criteria)# Refine corners
    for i, points in enumerate(p0):# Draw and display the corners
        x, y = points[0][0], points[0][1]
        cv2.circle(next, center=(round(x), round(y)),
                   radius=10, color=(0, 255, 0), thickness=-1)
    cv2.imshow('img', next)
    cv2.waitKey(0)

    tracks, frames = track_extract_frames("Assignment_MV_02_video.mp4")# Extract tracks from video
    return tracks, frames,images, mtx


def task2(tracks, frame, images):

    def task2A(f1, f2, images, tracks):
        # Euclideans calculations
        Fframe = images[f1].copy()
        Lframe = images[f2].copy()
        cords = []
        for track in tracks:
            if f1 in tracks[track] and f2 in tracks[track]:
                x = round(tracks[track][f1][0, 0]), round(
                    tracks[track][f1][0, 1]), 1
                X = round(tracks[track][f2][0, 0]), round(
                    tracks[track][f2][0, 1]), 1
                cv2.circle(Fframe, center=(x[0], x[1]), radius=3, color=(
                    0, 255, 0), thickness=-1)
                cv2.circle(Lframe, center=(X[0], X[1]), radius=3, color=(
                    0, 255, 0), thickness=-1)
                cords.append([np.array(x), np.array(X)])
        cv2.imshow('First Frame', Fframe)
        cv2.waitKey(0)
        cv2.imshow('Last Frame', Lframe)
        cv2.waitKey(0) 
        return np.array(cords)

    def task2B(cords): 
        Fmean = np.array(np.array(cords)[:, 0].mean(axis=0)) # Mean of first frame
        Lmean = np.array(np.array(cords)[:, 1].mean(axis=0)) # Mean of last frame

        Fsigma = np.std(np.array(cords)[:, 0], axis=0) # Standard deviation of first frame
        Lsigma = np.std(np.array(cords)[:, 1], axis=0) # Standard deviation of last frame

        Ncords = []
        print(len(cords))
        for i in range(len(cords)): # For each pair of points
            T = np.array([
                [1 / Fsigma[0], 0, -Fmean[0] / Fsigma[0]],
                [0, 1 / Fsigma[1], -Fmean[1] / Fsigma[1]],
                [0, 0, 1]
            ]) # Transformation matrix

            TD = np.array([
                [1 / Lsigma[0], 0, -Lmean[0] / Lsigma[0]],
                [0, 1 / Lsigma[1], -Lmean[1] / Lsigma[1]],
                [0, 0, 1]
            ]) # Transformation matrix
            # Normalize points
            Ncords.append([np.matmul(T, (cords[:, 0][i])),
                          np.matmul(TD, (cords[:, 1][i]))]) 
        return np.array(Ncords), T, TD

    # Task [C-F]
    def Task2Rest(Ncords, T, TD, cords):
        for i in range(10000): 
            outliers = list()
            inliners = list()
            inliner_test_statistics = 0   
            samples_in = set(random.sample(set([i for i in range(len(cords))]),8)) # Randomly select 8 samples
            samples_out = list(set([i for i in range(len(cords))]) - samples_in) # Select the rest
            # matrix of 8 points
            A = np.zeros((0, 9)) # Matrix of 8 points
            for j in range(len(cords)): 
                Ai = np.kron(Ncords[j][0].T, Ncords[j][1].T) # Matrix of 8 points
                A = np.append(A, np.array([Ai]), axis=0) 

            U, S, V = np.linalg.svd(A) # SVD
            f = V[8, :].reshape(3, 3).T # Last column of V

            U, S, V = np.linalg.svd(f) # SVD
            f = np.matmul(U, np.matmul(np.diag([S[0], S[1], S[0]]), V)) 
            F = np.matmul(TD.T, np.matmul(f, T)) # Fundamental matrix

            outlier = {
                'best_in': np.inf,
                'best_out_count': len(cords) + 1,
                'best_F': np.eye(3), 
                'best_mat': None,
                'best_inlier': None,
            }
            for x, X in cords[samples_out, :, :]:
                gi = np.matmul(np.matmul(X.reshape(3, 1).T, F), x.reshape(3, 1)) 
                variance = np.matmul(np.matmul(np.matmul(np.matmul(X.reshape(3, 1).T, F), outlier['best_F']), F.T), X.reshape(3, 1)) +\
                 np.matmul(np.matmul(np.matmul(np.matmul(x.reshape(3, 1).T, F.T), outlier['best_F']), F), x.reshape(3, 1)) # Variance of the inliers

                Ti = gi**2 / variance # Test statistics

                if Ti > 6.635: 
                    outliers.append(j)
                else:
                    inliners.append([x, X]) 
                    
                    inliner_test_statistics += Ti

                j += 1


            if len(outliers) == outlier['best_out_count']:
                if inliner_test_statistics < outlier['best_in']:
                    outlier['best_in'] = inliner_test_statistics
                    outlier['best_mat'] = F
                    outlier['best_inlier'] = inliners

            elif len(outliers) < outlier['best_out_count']:
                outlier['best_out_count'] = len(outliers)
                outlier['best_mat'] = F
                outlier['best_inlier'] = inliners


        Fframe = images[f1].copy()
        Lframe = images[f2].copy()
        e1, e2 = calculate_epipoles(outlier['best_mat'])

        width = Fframe.shape[1]
        height = Lframe.shape[0]

        x = np.array([0.5 * width, 0.5 * height, 1])

        cv2.circle(images[0], (int(e1[0] / e1[2]), int(e1[1] / e1[2])), 3, (0, 0, 255), 2)
        cv2.circle(images[30], (int(e2[0] / e2[2]), int(e2[1] / e2[2])), 3, (0, 0, 255), 2)

        cv2.circle(Fframe, (int(x[0] / x[2]), int(x[1] / x[2])), 3, (0, 0, 255), 2)

 
        cv2.imshow("img2", images[30])
        cv2.waitKey(0)
        cv2.imshow("img3", Fframe)
        cv2.waitKey(0)
        return outlier

    f1 = 0
    f2 = frame
    tracks, frame = track_extract_frames("Assignment_MV_02_video.mp4")
    images = extract_frames("Assignment_MV_02_video.mp4", [f1, f2])
    cordinates = task2A(f1, f2, images, tracks)
    Ncords, T, TD = task2B(cordinates)
    F = Task2Rest(Ncords, T, TD, cordinates)
    return F



tracks, frames,images, K = task1()
F = task2(tracks, frames, images)