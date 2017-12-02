import sys
import datetime as dt
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

from opengl_viewer.voxel_flow import Voxel_Flow_3D
from opengl_viewer.optical_flow import Optical_flow_3D
from opengl_viewer.camera import Camera
from opengl_viewer.shapes import *


width, height = 1000, 800
step_size0 = 0.05
rotation_angle0 = 2 * np.pi/180




class OpenGlViewer:
    def __init__(self, op_flow, record=False):
        self.record = record
        self.last_frame_change = time.time() + 5
        self.last_draw = time.time()
        self.frame = 0
        self.draw_fps = 20
        self.fps = 0
        self.last_key   = None
        self.last_key_t = dt.datetime.now()

        # Camera + parameters
        self.camera = Camera()
        self.step_size = step_size0
        self.rotation_angle = rotation_angle0

        # Additional stuff to avoid extra function calls
        self.quadric = gluNewQuadric()

        # self.op_flow = Optical_flow_3D(op_flow)
        self.op_flow = Voxel_Flow_3D(op_flow)
        self.num_frames = self.op_flow.num_frames
        self.buffers = None



    def draw(self):
        '''
        Callback to draw everything in the glut windows
        '''

        # Only change camera angle if frame isn't different
        if self.frame == self.get_frame():
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(*self.camera.get_viewing_matrix())
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*self.camera.get_viewing_matrix())

        self.set_fps()
        self.draw_axes()

        # draw objects
        if self.buffers is None:
            self.create_vbo()
        self.draw_vbo()

        glFlush()
        glutSwapBuffers()

        if self.record:
            screenshot = glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE)
            image = Image.frombytes("RGB", (width, height), screenshot)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            import glob
            frame_num = len(glob.glob('/Users/mpeven/Documents/PhD/Activity_Recognition/screenshots/*'))
            image.save('screenshots/frame_{:05}.jpg'.format(frame_num))
            print(self.frame)
            if self.frame == (self.num_frames - 1):
                glutLeaveMainLoop()



    def create_vbo(self):
        '''
        Builds the buffers on the GPU
        '''
        self.buffers = glGenBuffers(self.op_flow.num_frames * 3)
        for frame in tqdm(range(self.op_flow.num_frames), "Sending VBO to GPU"):
            # Vertex colors
            glBindBuffer(GL_ARRAY_BUFFER, self.buffers[frame*3])
            glBufferData(GL_ARRAY_BUFFER,
                         len(self.op_flow.get_colors(frame))*4,
                         (ctypes.c_float*len(self.op_flow.get_colors(frame)))(*self.op_flow.get_colors(frame)),
                         GL_DYNAMIC_DRAW)
            # Vertex indices
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[1 + frame*3])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                         len(self.op_flow.get_indices(frame))*4,
                         (ctypes.c_uint*len(self.op_flow.get_indices(frame)))(*self.op_flow.get_indices(frame)),
                         GL_DYNAMIC_DRAW)
            # Vertex locations
            glBindBuffer(GL_ARRAY_BUFFER, self.buffers[2 + frame*3])
            glBufferData(GL_ARRAY_BUFFER,
                         len(self.op_flow.get_vertices(frame))*4,
                         (ctypes.c_float*len(self.op_flow.get_vertices(frame)))(*self.op_flow.get_vertices(frame)),
                         GL_DYNAMIC_DRAW)




    def draw_vbo(self):
        '''
        Binds the buffer objects for the current frame
        '''
        frame = self.get_frame()
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[2 + frame*3])
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0 + frame*3])
        glColorPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[1 + frame*3])
        glDrawElements(GL_TRIANGLES, len(self.op_flow.get_indices(frame)), GL_UNSIGNED_INT, None)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)




    def view(self):
        '''
        Main function to create window and register callbacks for displaying
        '''
        # Initialize glut
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(0, 0)
        glutCreateWindow("Optical Flow Viewer")
        glutMouseFunc(self.mouse_button)
        glutMotionFunc(self.mouse_motion)
        glutDisplayFunc(self.draw)
        glutIdleFunc(self.draw)
        glutReshapeFunc(self.reshape_func)
        glutKeyboardFunc(self.key_pressed)
        glutSpecialFunc(self.sp_key_pressed)

        # Initialize opengl environment
        glClearColor(0., 0., 0., 1.)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_TEXTURE_2D)
        glShadeModel(GL_SMOOTH)

        # Enter main loop - returns on glutLeaveMainLoop
        glutMainLoop()



    def reshape_func(self, Width, Height):
        '''
        Callback to change the camera on a window resize
        '''
        if Height == 0: Height = 1
        glViewport(0, 0, Width, Height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(Width)/float(Height), 0.1, 50.0)




    def sp_key_pressed(self, key, x, y):
        '''
        Arrow keys callback
        '''

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
        '''
        Keyboard callback
        '''

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
        # Change speed
        elif key == b'z': self.draw_fps -= 1 if self.draw_fps > 0 else 0
        elif key == b'x': self.draw_fps += 1




    def mouse_button(self, button, mode, x, y):
        '''
        Mouse click callback
        '''
        if mode == GLUT_DOWN:
            self.mouse_down = True
            self.mouse_start = (x, y)
        else:
            self.mouse_down = False




    def mouse_motion(self, x, y):
        '''
        Mouse motion callback
        '''
        self.camera.rotate_camera_right((x - self.mouse_start[0])*0.001)
        self.camera.rotate_camera_up((y - self.mouse_start[1])*0.001)
        self.mouse_start = (x, y)





    def draw_axes(self):
        '''
        Draws x, y, and z axes
        '''
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




    def set_fps(self):
        '''
        Set the window title to the current FPS
        '''
        seconds = time.time()
        if (seconds - self.last_draw >= 1):
            self.last_draw = seconds
            glutSetWindowTitle("{} FPS".format(self.fps))
            self.fps = 0
        self.fps += 1




    def get_frame(self):
        '''
        Return the frame to draw based on the set fps
        '''
        now = time.time()
        if self.draw_fps == 0:
            return self.frame
        if (now - self.last_frame_change) > 1.0/self.draw_fps:
            self.frame = 0 if self.frame == self.num_frames-1 else self.frame + 1
            self.last_frame_change = now
        return self.frame
