# ARK Task 4: Kalman Filter

### Project Structure
main.py is the entry-point into the project, 
which imports the KalmanFilter class from kalman.py.
The input CSVs are in the data directory while
the calculated positions and trajectory plots are saved to the out directory.

The exploration directory has a Jupyter notebook which was used
to visualize the data at the start and to explore 3D plotting.

The tuning of the filter is robust and generalizes to all 4 input files.
main.py iterates over all 4 of them and generates the required trajectories.

Each case has 5 plots. The 3 trajectory_ plots depict the entire trajectory
of the drone with respect to the first ground station. Of these, the _combined
plot contains both the trajectories plotted on the same axes. However, as the
measured trajectory is radially outwards from the updated one,
it entirely covers the updated one except for at the very start.
For this reason, the opacity has been set to a very low value
to allow at least some visualization of the radial dispersion.
Owing to the sheer number of data points, the accuracy of the Filter as
measured by the smoothness of the trajectory is difficult to visualize.
To this end, the partial_trajectory plots contain only the last 200
data points and allow for much better visualization of the accuracy. 

Rationale for decisions such as using
the Joseph Form of the Process Covariance Update Equation
and for some parameters and covariance values
are given in code comments or in the documentation file.
