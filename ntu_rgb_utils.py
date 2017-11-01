from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys
import datetime as dt
import time
import numpy as np
import pickle


width, height = 1000, 800
step_size0 = 0.05
rotation_angle0 = 2 * np.pi/180
xyz0     = np.array([0., .2, -3.])
forward0 = np.array([0., 0., 1.])
up0      = np.array([0., 1., 0.])


class Camera:
    def __init__(self):
        self.xyz     = xyz0.copy()
        self.forward = forward0.copy()
        self.up      = up0.copy()
        self.right   = np.cross(self.forward, self.up)

    def get_viewing_matrix(self):
        reference_point = self.xyz + self.forward
        all_vals = *self.xyz, *reference_point, *self.up
        return all_vals

    def reset(self):
        self.xyz     = xyz0.copy()
        self.forward = forward0.copy()
        self.up      = up0.copy()

    def rotate_camera_right(self, angle):
        x, y, z = self.forward
        self.forward = self.forward * np.cos(angle) - self.right * np.sin(angle)
        self.forward /= np.linalg.norm(self.forward)
        self.right   = np.cross(self.forward, self.up)

    def rotate_camera_up(self, angle):
        x, y, z = self.forward
        self.forward = self.forward * np.cos(angle) + self.up * np.sin(angle)
        self.forward /= np.linalg.norm(self.forward)
        self.up = np.cross(self.forward, self.right) * -1


    def move_forward(self, step_size):
        self.xyz += step_size * self.forward

    def move_right(self, step_size):
        self.xyz += step_size * self.right



class PointCloudViewer:
    def __init__(self, pointclouds=None, op_flow=None):
        self.last_frame_change = time.time()
        self.last_draw = time.time()
        self.pointclouds = pointclouds
        self.op_flow = op_flow
        self.frame = 0
        self.draw_fps = 2
        self.fps = 0
        self.last_key   = None
        self.last_key_t = dt.datetime.now()

        # Camera + parameters
        self.camera = Camera()
        self.step_size = step_size0
        self.rotation_angle = rotation_angle0

        # Additional stuff to avoid extra function calls
        self.quadric = gluNewQuadric()



    def view(self):
        """
        Main function to create window and register callbacks for displaying
        """
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(0, 0)
        glutCreateWindow("Point Cloud Viewer")
        glutMouseFunc(self.mouse_button)
        glutMotionFunc(self.mouse_motion)
        glutDisplayFunc(self.display)
        glutIdleFunc(self.display)
        glutKeyboardFunc(self.key_pressed)
        glutSpecialFunc(self.sp_key_pressed)
        self.init_gl()
        glutMainLoop()



    def init_gl(self):
        glClearColor(0., 0., 0., 1.)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_TEXTURE_2D)
        glShadeModel(GL_SMOOTH)



    def sp_key_pressed(self, key, x, y):
        # Accelerate if continuing to press a key
        if key == self.last_key and (dt.datetime.now() - self.last_key_t).total_seconds() < 0.1:
            if self.rotation_angle < 5*rotation_angle0:
                self.rotation_angle = self.rotation_angle * 1.02
        else:
            self.rotation_angle = rotation_angle0

        self.last_key = key
        self.last_key_t = dt.datetime.now()

        if key == GLUT_KEY_LEFT:
            self.camera.rotate_camera_right(self.rotation_angle)
        elif key == GLUT_KEY_RIGHT:
            self.camera.rotate_camera_right(-self.rotation_angle)
        elif key == GLUT_KEY_UP:
            self.camera.rotate_camera_up(self.rotation_angle)
        elif key == GLUT_KEY_DOWN:
            self.camera.rotate_camera_up(-self.rotation_angle)


    def key_pressed(self, key, x, y):
        """
        Callback to handle keyboard interactions
        """

        # Accelerate if continuing to press a key
        if key == self.last_key and (dt.datetime.now() - self.last_key_t).total_seconds() < 0.1:
            if self.step_size < step_size0*5:
                self.step_size = self.step_size * 1.1
        else:
            self.step_size = step_size0

        self.last_key = key
        self.last_key_t = dt.datetime.now()

        # Esc or q to exit
        if (key == b'\x1b' or key == b'q'): sys.exit()
        # WASD movement controls
        elif key == b'w': self.camera.move_forward(self.step_size)
        elif key == b'a': self.camera.move_right(-self.step_size)
        elif key == b's': self.camera.move_forward(-self.step_size)
        elif key == b'd': self.camera.move_right(self.step_size)
        # Reset view
        elif key == b'r': self.camera.reset()



    def mouse_button(self, button, mode, x, y):
        """
        Callback to handle mouse clicks
        """
        print("mb:", button, mode, x, y)

        if mode == GLUT_DOWN:
            self.mouse_down = True
            self.mouse_start = (x, y)
        else:
            self.mouse_down = False



    def mouse_motion(self, x, y):
        """
        Callback to handle mouse motion
        """
        print("mm:", x, y)



    def display(self):
        """
        Callback to set the camera position and draw the scene
        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*self.camera.get_viewing_matrix())
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glColorMaterial(GL_FRONT,GL_AMBIENT_AND_DIFFUSE)
        # glEnable(GL_LIGHTING)
        # glLightfv(GL_LIGHT0, GL_AMBIENT, (1.0, 1.0, 1.0, 1.0))
        # glLightfv(GL_LIGHT1, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        # glLightfv(GL_LIGHT2, GL_POSITION, (0.0, 3.0, -5.0, 1.0))
        # glEnable(GL_LIGHT0)
        # glEnable(GL_LIGHT1)
        # glEnable(GL_LIGHT2)
        self.draw_axes()
        if self.op_flow:
            self.draw_optical_flow()
        else:
            self.draw_point_cloud()
        glutSwapBuffers()



    def draw_axes(self):
        """
        Draws points at origin to help with debugging
        """
        glBegin(GL_LINES)
        glColor3f(1, 1, 0)      # x-axis
        glVertex3f(-1000, 0, 0)
        glVertex3f(1000, 0, 0)
        glColor3f(1, 0, 1)      # y-axis
        glVertex3f(0, -1000, 0)
        glVertex3f(0, 1000, 0)
        glColor3f(0, 0, 1)      # z-axis
        glVertex3f(0, 0, -1000)
        glVertex3f(0, 0, 1000)
        glEnd()



    def draw_point_cloud(self):
        """
        Draws all spheres in the pointcloud
        """
        self.set_fps()
        xs, ys, zs = self.pointclouds[self.get_frame(self.pointclouds)]
        glColor3f(1, 0, 0)
        for i in range(0, len(xs), 1):
            glPushMatrix()
            glTranslatef(xs[i], ys[i], zs[i])
            glutSolidSphere(0.008, 5, 5)
            glPopMatrix()



    def draw_optical_flow(self):
        """
        Draws 3D optical flow as cones
        """
        self.set_fps()
        op_flow_list = self.op_flow[self.get_frame(self.op_flow)]
        glColor3f(1, 0, 0)
        for i in range(0, len(op_flow_list), 1):
            flow = op_flow_list[i]
            glPushMatrix()
            # Get perpendicular vector
            glTranslatef(flow[0], -flow[1], flow[2])
            # glRotatef(10, 1, 0, 0)
            # glutSolidCone(0.02, 0.1, 5, 5)
            glutSolidSphere(0.008, 5, 5)
            glPopMatrix()



    def set_fps(self):
        seconds = time.time()
        if (seconds - self.last_draw >= 1):
            self.last_draw = seconds
            glutSetWindowTitle("{} FPS".format(self.fps))
            self.fps = 0
        self.fps += 1


    def get_frame(self, points):
        now = time.time()
        if (now - self.last_frame_change) > 1.0/self.draw_fps:
            self.frame = 0 if self.frame == len(points)-1 else self.frame + 1
            self.last_frame_change = now
        return self.frame



def get_point_clouds():
    depth_ims = np.load('depth_im.npy')
    pointclouds = []
    mean_z = np.mean(depth_ims[depth_ims > 0])
    for i in range(depth_ims.shape[0]):
        xs, ys, zs = [], [], []
        nonzeros = np.nonzero(depth_ims[i])
        for y, x in zip(nonzeros[0], nonzeros[1]):
            xs.append((x-300)/100.0)
            ys.append(-(y-200)/100.0)
            z = depth_ims[i, y, x]
            zs.append((z - mean_z)/1000.0)
        pointclouds.append((xs, ys, zs))


if __name__ == '__main__':
    pointclouds = None
    optical_flow = None

    # import ntu_rgb
    # dataset = ntu_rgb.NTU()

    # Point clouds
    # pickle.dump(pointclouds, open('pointclouds_2.pickle', 'wb'))
    # pointclouds = pickle.load(open('pointclouds_2.pickle', 'rb'))

    # RGB Point Cloud
    # rgb_3D = dataset.get_rgb_3D_maps(0)
    # pickle.dump(rgb_3D[:50], open('3D_maps_0.pickle', 'wb'))
    # pickle.dump(rgb_3D[50:], open('3D_maps_1.pickle', 'wb'))
    # pc1 = pickle.load(open('3D_maps_0.pickle', 'rb'))
    # pc2 = pickle.load(open('3D_maps_1.pickle', 'rb'))
    # pointclouds = pc1.append(pc2)
    bytes_in = bytearray(0)
    max_bytes = 2**31 - 1
    input_size = os.path.getsize("3D_maps_0.pickle")
    with open("3D_maps_0.pickle", 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data1 = pickle.loads(bytes_in)
    bytes_in = bytearray(0)
    input_size = os.path.getsize("3D_maps_1.pickle")
    with open("3D_maps_1.pickle", 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data2 = pickle.loads(bytes_in)
    pointclouds = np.concatenate([data1, data2])
    print(pointclouds.shape)

    # Optical Flow
    # optical_flow = dataset.get_3D_optical_flow(0)
    # pickle.dump(optical_flow, open('optical_flow.pickle', 'wb'))
    # optical_flow = pickle.load(open('optical_flow.pickle', 'rb'))

    viewer = PointCloudViewer(pointclouds=pointclouds, op_flow=optical_flow)
    viewer.view()









