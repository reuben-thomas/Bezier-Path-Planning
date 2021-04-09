import nlopt
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interpolate
import scipy.optimize as optimize
from cubic_bezier_planner import calc_6points_bezier_path
from cubic_bezier_planner import calc_4points_bezier_path
from cubic_spline_planner import calc_spline_course


class Path:
    def __init__(self, x, y, yaw, k):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.k = k

class Spline:

    def __init__(self, ax, ay, bound):
        
        #input waypoint coordinates
        ayaw, k = self.calc_yaw_curvature(ax, ay)
        self.waypoints = Path(ax, ay, ayaw, k)
        
        # defines and sets left and right boundary lines
        self.bound = bound
        lax, lay, rax, ray = self.init_boundary()
        self.left_bound = Path(lax, lay, None, None)
        self.right_bound = Path(rax, ray, None, None)

        # default unoptimized cubic bezier path to initialize curvature
        bx, by, _, _ = self.cubic_bezier_path(ax, ay)
        byaw, bk = self.calc_yaw_curvature(bx, by)
        self.default_path = Path(bx, by, byaw, bk)

        # optimized path
        self.optimized_path = Path([], [], [], [])
        self.ctr_points = Path([], [], [], [])

    # Calculates the first derivative of input arrays
    def calc_d(self, x, y):

        dx = []
        dy = []

        for i in range(0, len(x)-1):
            dx.append(x[i+1] - x[i])
            dy.append(y[i+1] - y[i])
        
        dx.append(dx[-1])
        dy.append(dy[-1])
        return dx, dy

    # Calculates yaw and curvature given input path
    def calc_yaw_curvature(self, x, y):

        dx, dy = self.calc_d(x,y)
        ddx, ddy = self.calc_d(dx, dy)
        yaw = []
        k = []

        for i in range(0, len(x)):
            yaw.append(math.atan2(dy[i], dx[i]))
            k.append( (ddy[i] * dx[i] - ddx[i] * dy[i]) / ((dx[i]**2 + dy[i]**2)**(3/2)) )
    
        return yaw, k

    # Calculates total distance of the path
    def calc_path_dist(self, x, y):

        dx = np.absolute(self.calc_d(np.zeros(len(x)), x))
        dy = np.absolute(self.calc_d(np.zeros(len(y)), y))
        ddist = np.hypot(dx, dy)

        return np.sum(ddist)

    # Bezier path one as per the approach suggested in
    # https://users.soe.ucsc.edu/~elkaim/Documents/camera_WCECS2008_IEEE_ICIAR_58.pdf
    def cubic_bezier_path(self, ax, ay):

        dyaw, _ = self.calc_yaw_curvature(ax, ay)
        
        # control point and path array
        cx, cy, ctr_pt_x, ctr_pt_y = [], [], [], []

        ayaw = dyaw.copy()

        for n in range(1, len(ax)-1):
            yaw = 0.5*(dyaw[n] + dyaw[n-1])
            ayaw[n] = yaw

        # for n waypoints, there are n-1 bezier curves
        for i in range(len(ax)-1):

            path, points = calc_4points_bezier_path(ax[i], ay[i], ayaw[i], ax[i+1], ay[i+1], ayaw[i+1], 2.0)
            cx = np.concatenate((cx, path.T[0][:-1]))
            cy = np.concatenate((cy, path.T[1][:-1]))
            cyaw, k = self.calc_yaw_curvature(cx, cy)

            for p in points:
                ctr_pt_x.append( p[0] )
                ctr_pt_y.append( p[1] )

        return cx, cy, ctr_pt_x, ctr_pt_y

    # Approximated quintic bezier path with curvature continuity
    def quintic_bezier_path(self, ax, ay, offset_1, offset_2):

        dyaw, _ = self.calc_yaw_curvature(ax, ay)
        
        # control point and path array
        cx, cy, ctr_pt_x, ctr_pt_y = [], [], [], []

        ayaw = dyaw.copy()

        for n in range(1, len(ax)-1):
            yaw = 0.5*(dyaw[n] + dyaw[n-1])
            ayaw[n] = yaw

        # for n waypoints, there are n-1 bezier curves
        for i in range(len(ax)-1):

            path, points = calc_6points_bezier_path(ax[i], ay[i], ayaw[i], ax[i+1], ay[i+1], ayaw[i+1], offset_1[i], offset_2[i])
            cx = np.concatenate((cx, path.T[0][:-1]))
            cy = np.concatenate((cy, path.T[1][:-1]))
            cyaw, k = self.calc_yaw_curvature(cx, cy)
            
            for p in points:
                ctr_pt_x.append( p[0] )
                ctr_pt_y.append( p[1] )

        return cx, cy, ctr_pt_x, ctr_pt_y

    # Objective function of cost to be minimized
    def cubic_objective_func(self, deviation):

        ax = self.waypoints.x.copy()
        ay = self.waypoints.y.copy()

        for n in range(0, len(deviation)):
            ax[n+1] -= deviation[n]*np.sin(self.waypoints.yaw[n+1])
            ay[n+1] += deviation[n]*np.cos(self.waypoints.yaw[n+1])

        bx, by, _, _ = self.cubic_bezier_path(ax, ay)
        yaw, k = self.calc_yaw_curvature(bx, by)

        # cost of curvature continuity
        t = np.zeros((len(k)))
        dk = self.calc_d(t, k)
        absolute_dk = np.absolute(dk)
        continuity_cost = 10.0 * np.mean(absolute_dk)

        # curvature cost
        absolute_k = np.absolute(k)
        curvature_cost = 14.0 * np.mean(absolute_k)
        
        # cost of deviation from input waypoints
        absolute_dev = np.absolute(deviation)
        deviation_cost = 1.0 * np.mean(absolute_dev)

        distance_cost = 0.5 * self.calc_path_dist(bx, by)

        return curvature_cost + deviation_cost + distance_cost + continuity_cost

    # Objective function for quintic bezier
    def quintic_objective_func(self, params):

        ax = self.waypoints.x.copy()
        ay = self.waypoints.y.copy()

        # calculate offsets and input waypoints
        waypoints = len(self.waypoints.yaw)
        deviation = params[ :(waypoints-2) ]
        offset_1 = params[ (waypoints-2) : (2*waypoints-3) ]
        offset_2 = params[ (2*waypoints-3) : ]

        for n in range(0, len(self.waypoints.yaw)-2):
            ax[n+1] -= deviation[n]*np.sin(self.waypoints.yaw[n+1])
            ay[n+1] += deviation[n]*np.cos(self.waypoints.yaw[n+1])

        bx, by, _, _ = self.quintic_bezier_path(ax, ay, offset_1, offset_2)
        yaw, k = self.calc_yaw_curvature(bx, by)

        # cost of curvature continuity
        t = np.zeros((len(k)))
        dk = self.calc_d(t, k)
        absolute_dk = np.absolute(dk)
        continuity_cost = 95.0 * np.mean(absolute_dk)

        # cost of distance
        distance_cost = 0.5 * self.calc_path_dist(bx, by)

        # curvature cost
        absolute_k = np.absolute(k)
        curvature_cost = 25.0 * np.mean(absolute_k)
        
        # cost of deviation from input waypoints
        absolute_dev = np.absolute(deviation)
        deviation_cost = 1.0 * np.mean(absolute_dev)

        return curvature_cost + deviation_cost + distance_cost + continuity_cost

    # Determines position of boundary lines for visualization
    def init_boundary(self):

        rax = []
        ray = []
        lax = []
        lay = []

        for n in range(0, len(self.waypoints.yaw)):
            lax.append(self.waypoints.x[n] - self.bound*np.sin(self.waypoints.yaw[n]))
            lay.append(self.waypoints.y[n] + self.bound*np.cos(self.waypoints.yaw[n]))
            rax.append(self.waypoints.x[n] + self.bound*np.sin(self.waypoints.yaw[n]))
            ray.append(self.waypoints.y[n] - self.bound*np.cos(self.waypoints.yaw[n]))
        
        return lax, lay, rax, ray

    # Minimize objective function using scipy optimize minimize
    def optimize_min_cubic(self):

        print("Attempting optimization minima")

        initial_guess = [0, 0, 0, 0, 0]
        bnds = ((-self.bound, self.bound), (-self.bound, self.bound), (-self.bound, self.bound), (-self.bound, self.bound), (-self.bound, self.bound))
        result = optimize.minimize(self.cubic_objective_func, initial_guess, bounds=bnds)

        ax = self.waypoints.x.copy()
        ay = self.waypoints.y.copy()

        if result.success:
            print("optimized true")
            deviation = result.x
            for n in range(0, len(deviation)):
                ax[n+1] -= deviation[n]*np.sin(self.waypoints.yaw[n+1])
                ay[n+1] += deviation[n]*np.cos(self.waypoints.yaw[n+1])

            x, y, _, _ = self.cubic_bezier_path(ax, ay)
            yaw, k = self.calc_yaw_curvature(x, y)
            self.optimized_path = Path(x, y, yaw, k)

        else:
            print("optimization failure, defaulting")
            exit()

    # Minimize quintic objective function
    def optimize_min_quintic(self):

        print("Attempting optimization minima")

        initial_guess = [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0]

        bnds = ((-self.bound, self.bound), (-self.bound, self.bound), (-self.bound, self.bound), (-self.bound, self.bound), (-self.bound, self.bound), 
                (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), 
                (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        result = optimize.minimize(self.quintic_objective_func, initial_guess, bounds=bnds)

        ax = self.waypoints.x.copy()
        ay = self.waypoints.y.copy()

        if result.success:
            print("optimized true")
            params = result.x
            
            # collects offsets for individual bezier curves
            waypoints = len(self.waypoints.yaw)
            deviation = params[ :(waypoints-2) ]
            offset_1 = params[ (waypoints-2) : (2*waypoints-3) ]
            offset_2 = params[ (2*waypoints-3) : ]       

            # updated set of waypoints
            for n in range(0, len(self.waypoints.yaw)-2):
                ax[n+1] -= deviation[n]*np.sin(self.waypoints.yaw[n+1])
                ay[n+1] += deviation[n]*np.cos(self.waypoints.yaw[n+1])

            x, y, ctr_pt_x, ctr_pt_y = self.quintic_bezier_path(ax, ay, offset_1, offset_2)
            yaw, k = self.calc_yaw_curvature(x, y)

            # update path optimized path and control points
            self.optimized_path = Path(x, y, yaw, k)
            self.ctr_points = Path(ctr_pt_x, ctr_pt_y, [], [])        

        else:
            print("optimization failure, defaulting")
            exit()

def main():
    # define input path
    ax = [0.0, 2.3, 6.25, 8.6, 8.2, 5.3, 2.6]
    ay = [0.0, 7.16, 13.68, 22.3, 30.64, 39.6, 50.4]
    boundary = 2.5

    spline = Spline(ax, ay, boundary)
    # spline.optimize_min_quintic()

    # Path plot
    plt.subplots(1)
    plt.plot(spline.left_bound.x, spline.left_bound.y, '--r', alpha=0.5, label="left boundary")
    plt.plot(spline.right_bound.x, spline.right_bound.y, '--g', alpha=0.5, label="right boundary")
    #plt.plot(spline.default_path.x, spline.default_path.y, '-y', label="default")
    plt.plot(spline.optimized_path.x, spline.optimized_path.y, '-m', label="optimized")
    plt.plot(spline.ctr_points.x, spline.ctr_points.y, '.r', label="control points")
    plt.plot(spline.waypoints.x, spline.waypoints.y, '.', label="waypoints")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")

    # Heading plot
    plt.subplots(1)
    plt.plot([np.rad2deg(iyaw) for iyaw in spline.default_path.yaw], "-y", label="original")
    plt.plot([np.rad2deg(iyaw) for iyaw in spline.optimized_path.yaw], "-m", label="optimized")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    # Curvature plot
    plt.subplots(1)
    plt.plot(spline.default_path.k, "-y", label="original")
    plt.plot(spline.optimized_path.k, "-m", label="optimized")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()

if __name__ == '__main__':
    main()