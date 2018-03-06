import cv2
import numpy as np
import ntu_rgb
import matplotlib.pyplot as plt



def get_animation(images, flow_maps):
    ''' Draw optical flow arrows on all the images and return the animation '''

    from matplotlib import animation
    def get_op_flow_img(i):
        # Get the gray image but add rgb channels (for color arrows)
        img = images[i].copy()
        img = np.stack((img,)*3, axis=2)
        # img = (img/2).astype(np.uint8)

        # Return an arrow if the flow vector is greater than 2.0
        def get_arrow(p):
            if np.linalg.norm(flow_maps[i,:,pt[1],pt[0]]) > 2.0:
                return (flow_maps[i,:,pt[1],pt[0]]*3+pt).astype(np.int32) # Scale up for visibility
            else:
                return None

        # Draw all the arrows to the image
        step_size = int(img.shape[0]/100) # draw arrows every 0.5% of the image size
        for y in range(int(img.shape[0]/step_size)):
            for x in range(int(img.shape[1]/step_size)):
                pt = (x*step_size,y*step_size)
                new_pt = get_arrow(pt)
                if new_pt is not None:
                    cv2.arrowedLine(img, pt1=tuple(pt), pt2=tuple(new_pt), color=(0, 255, 0), thickness=1, tipLength=.2)
        return img

    # Plot all the images
    fig = plt.figure(figsize=(15, 10))
    ims = []
    for i in range(len(images) - 1):
        im = plt.imshow(get_op_flow_img(i), animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    plt.close()
    return ani
