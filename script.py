import CameraCal
import pickle

cameraCal = False

if cameraCal:
    [mtx, dist] = CameraCal.calibrateCamera()
    pickle.dump([mtx, dist], open( "calibration.p", "wb"))

[mtx, dist] =pickle.load( open( "calibration.p", "rb" ) )
