import os
import cv2
import json
import random
import argparse
import numpy as np
from PIL import Image
from itertools import count
from math import tau, sin, cos
import matplotlib.pyplot as plt


def Phi(psi):
    R = np.array([[1,        0,        0],
                  [0, cos(psi), sin(psi)],
                  [0,-sin(psi), cos(psi)]])
    return R

def Theta(theta):
    R = np.array([[cos(theta), 0, -sin(theta)],
                  [         0, 1,           0],
                  [sin(theta), 0, cos(theta)]])
    return R

def Psi(phi):
    R = np.array([[ cos(phi), sin(phi), 0],
                  [-sin(phi), cos(phi), 0],
                  [        0,        0, 1]])
    return R


def smooth1d(x, window_len):
    # copied from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    s = np.r_[2*x[0] - x[window_len:1:-1], x, 2*x[-1] - x[-1:-window_len:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


def smooth2d(A, sigma=3):
    window_len = max(int(sigma), 3) * 2 + 1
    A = np.apply_along_axis(smooth1d, 0, A, window_len)
    A = np.apply_along_axis(smooth1d, 1, A, window_len)
    return A


class BaseFilter:

    def get_pad(self, dpi):
        return 0

    def process_image(self, padded_src, dpi):
        raise NotImplementedError("Should be overridden by subclasses")

    def __call__(self, im, dpi):
        pad = self.get_pad(dpi)
        pad = max(pad, 10) #🤔
        padded_src = np.pad(im, [(pad, pad), (pad, pad), (0, 0)], "constant")
        tgt_image = self.process_image(padded_src, dpi)
        return tgt_image, -pad, -pad


class GaussianFilter(BaseFilter):
    "simple gauss filter"

    def __init__(self, sigma, alpha=0.5, color=None):
        self.sigma = sigma
        self.alpha = alpha
        if color is None:
            self.color = (0, 0, 0)
        else:
            self.color = color

    def get_pad(self, dpi):
        return int(self.sigma*3/72.*dpi)

    def process_image(self, padded_src, dpi):
        tgt_image = np.zeros_like(padded_src)        
        aa = smooth2d(padded_src[:, :, -1]*self.alpha,
                      self.sigma/72.*dpi)
        tgt_image[:, :, -1] = aa
        tgt_image[:, :, :-1] = self.color
        return tgt_image


def get_feret_max(polygon):
    diff = polygon - polygon[:, None]
    dist_arr = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    return np.max(dist_arr)
    

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    

def generate_polyhedron(base_contour, fmin, fmax, platiness):
    '''
    Generates a five-layer polyhedron based on the contour and dimensions passed
    '''
    polygon = np.hstack([base_contour, np.zeros((base_contour.shape[0], 1))])
    polygon -= np.mean(polygon, axis=0)
    scale = fmax / get_feret_max(polygon)
    polygon *= scale

    psi = random.uniform(0, tau)
    polygon @= Psi(psi)
    
    top_point = polygon[np.argmax(polygon[:,1]), :]
    bottom_point = polygon[np.argmin(polygon[:,1]), :]

    r = random.uniform(0,1)
    
    intersection_point = r*top_point + (1-r)*bottom_point
    ax_height = platiness*fmin + (1-platiness)*fmax
    ax_pos = random.uniform(0.35, 0.65)

    axis = np.array([[0, 0, -ax_pos], [0, 0, 1-ax_pos]])
    axis *= ax_height/(axis[1,2]-axis[0,2]) # scale so that the HEIGHT of the axis (not magnitude) is equal to ax_height
    axis += intersection_point

    layers = []
    # First layer is the bottommost point
    layers.append(axis[0])
    # Second layer is the lower polygon
    lower_polygon = []
    for point in polygon:
        coef = random.uniform(0.5, 0.7)
        new_point = coef*point + (1-coef)*axis[0]
        new_point[2] = 0.5*(point[2] + axis[0,2])
        lower_polygon.append(new_point)
    lower_polygon = np.array(lower_polygon)
    layers.append(lower_polygon)
    # Third layer is the middle polygon
    layers.append(polygon)
    # Fourth layer is the upper polygon
    upper_polygon = []
    for point in polygon:
        coef = random.uniform(0.5, 0.8)
        new_point = coef*point + (1-coef)*axis[1]
        new_point[2] = 0.5*(point[2] + axis[1,2])
        upper_polygon.append(new_point)
    upper_polygon = np.array(upper_polygon)
    layers.append(upper_polygon)
    # Fifth layer is the uppermost point
    layers.append(axis[1])

    return layers


def _compute_volume(layers):
    '''
    Computes volume of a 5-layer grain polyhedron
    '''
    # Compute the areas of the three polygonal layers
    area_layer1 = PolyArea(*zip(*layers[1][:,:2]))
    area_layer2 = PolyArea(*zip(*layers[2][:,:2]))
    area_layer3 = PolyArea(*zip(*layers[3][:,:2]))

    # Get the distance between layers
    h01 = layers[1][0,2] - layers[0][2]
    h12 = layers[2][0,2] - layers[1][0,2]
    h23 = layers[3][0,2] - layers[2][0,2]
    h34 = layers[4][2] - layers[3][0,2]
    
    # Volumes of two irregular pyramids and two truncated irregular pyramids
    v_bottom = h01* area_layer1 / 3
    v_lower = h12 * (area_layer1 + area_layer2 + np.sqrt(area_layer1*area_layer2))
    v_upper = h23 * (area_layer2 + area_layer3 + np.sqrt(area_layer2*area_layer3))
    v_top = h34 * area_layer3 / 3

    # Add it all up
    volume = v_bottom + v_lower + v_upper + v_top

    return volume
    

class Grain:
    def __init__(self, contour, fmin, fmax, platiness, position, velocity, orientation, angular_velocity, grainId=0):

        self.id = grainId
        
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.angular_velocity = angular_velocity

        # Create a polyhedron from the base contour
        contour = contour.copy()
        self.layers = generate_polyhedron(contour, fmin, fmax, platiness)
        self.volume = _compute_volume(self.layers)
        self.feret_min = fmin
        

    def step(self, delta_t):
        self.position += self.velocity * delta_t
        self.orientation += self.angular_velocity * delta_t
        

    def display(self, ax=None):
        colors = 4*['k']
        if ax is None:
            colors = ['r','y','g','b']
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.view_init(elev=90, azim=-90, roll=0)
        rotation = Phi(self.orientation[0,0]) @ Theta(self.orientation[1,0]) @ Psi(self.orientation[2,0])
        translation = self.position
        layers = [(layer.copy() @ rotation) + translation.transpose() for layer in self.layers]

        if abs(self.position[2,0]) > 0.001:
            gauss = GaussianFilter(sigma=abs(self.position[2,0]), alpha=1) #depth in image determines bluriness
        else:
            gauss = None
        
        x1, y1, z1 = zip(*np.concatenate([layers[2], layers[2][:1,:]])) # Concatenate to make it closed
        x2, y2, z2 = zip(*np.concatenate([layers[3], layers[3][:1,:]]))
        ax.fill_between(x1, y1, z1, x2, y2, z2, edgecolor='k', facecolor=colors[0], agg_filter=gauss)
        
        x1, y1, z1 = zip(*np.concatenate([layers[3], layers[3][:1,:]])) # Concatenate to make it closed
        x2, y2, z2 = zip(*np.repeat(layers[0].reshape((1,-1)), layers[3].shape[0]+1, axis=0))
        ax.fill_between(x1, y1, z1, x2, y2, z2, edgecolor='k', facecolor=colors[1], agg_filter=gauss)
        
        x1, y1, z1 = zip(*np.concatenate([layers[2], layers[2][:1,:]])) # Concatenate to make it closed
        x2, y2, z2 = zip(*np.concatenate([layers[1], layers[1][:1,:]]))
        ax.fill_between(x1, y1, z1, x2, y2, z2, edgecolor='k', facecolor=colors[2], agg_filter=gauss)
        
        x1, y1, z1 = zip(*np.concatenate([layers[1], layers[1][:1,:]])) # Concatenate to make it closed
        x2, y2, z2 = zip(*np.repeat(layers[0].reshape((1,-1)), layers[1].shape[0]+1, axis=0))
        ax.fill_between(x1, y1, z1, x2, y2, z2, edgecolor='k', facecolor=colors[3], agg_filter=gauss)



def sample_distribution(distribution):
    r = random.uniform(0,1)
    prev_x, prev_y = distribution[0]
    if len(distribution) == 1:
        return prev_y
    for x, y in distribution[1:]:
        if r < x:
            break
        prev_x, prev_y = x, y
    interp_y = (y-prev_y)*(r-prev_x)/(x-prev_x) + prev_y
    return interp_y

    
class SandSimulator:
    def __init__(self, config):
        '''
        @Config: json object containing values for the following keys:
        REQUIRED:
         - contour_library
         - time_step
         - size_distribution
         - position_x_distribution
         - position_y_distribution
         - position_z_distrubition
         - velocity_x_distribution
         - velocity_y_distribution
         - velocity_z_distribution
         - frequency_distribution (distribution of grains per second)
        OPTIONAL:
         - background_image: filepath to image to be used as background for state renders
        Distributions are defined by lists of sequential setpoints (e.g. [(0.0, -10), (0.5, 10), (1.0, 20)])
        '''
        self.grainIndexer = count()
        self.grains = []
        self.config = config
        self.freq_error = 0

        self.background = Image.open(config['background_image'])
        with open(config['contour_library'], 'r') as file:
            data = json.load(file)
        self.contourLibrary = data['annotations']
        self.time = 0

        assert os.path.isdir(self.config['output_image_dir']), f"Invalid image output directory: {self.config['output_image_dir']}"
        assert not os.path.exists(self.config['output_log_filepath']), "Log file already exists"

    def test_config(self, n_steps=3):
        '''
        Runs n steps of a simulation and displays each frame
        '''
        for i in range(n_steps):
            self.step(CONFIG['time_step'])
            self.render_state()
        
        
    def run_simulation(self):
        self.record_state()
        for i in range(self.config['num_steps']):
            self.step(self.config['time_step'])
            self.record_state()
        

    def add_grain(self):
        x = sample_distribution(self.config['position_x_distribution'])
        y = sample_distribution(self.config['position_y_distribution'])
        z = sample_distribution(self.config['position_z_distribution'])
        position = np.array([[x],[y],[z]])

        u = sample_distribution(self.config['velocity_x_distribution'])
        v = sample_distribution(self.config['velocity_y_distribution'])
        w = sample_distribution(self.config['velocity_z_distribution'])
        velocity = np.array([[u],[v],[w]])
        
        phi = random.uniform(0, tau)
        theta = random.uniform(0, tau/2)
        psi = random.uniform(0, tau)
        orientation = np.array([[phi],[theta],[psi]])
        
        mag = sample_distribution(self.config['angular_velocity_mag_distribution'])
        elev = random.uniform(0, tau)
        azim = random.uniform(0, tau/2)
        angular_velocity =  Theta(elev) @ Psi(azim) @ np.array([[mag],[0],[0]])
        
        grainDict = random.choice(self.contourLibrary)
        contourVertices = grainDict['segmentation'][0]
        base_polygon = np.array(contourVertices).reshape((-1,2))
        
        fmin = sample_distribution(self.config['size_distribution'])
        fmax = fmin * grainDict['feret_max'] / grainDict['feret_min']
        platiness = sample_distribution(self.config['platiness_distribution'])

        grainId = next(self.grainIndexer)
        g = Grain(base_polygon, fmin=fmin, fmax=fmax, platiness=platiness, position=position, velocity=velocity, orientation=orientation, angular_velocity=angular_velocity, grainId=grainId)
        
        self.grains.append(g)
        
        
    def step(self, delta_t):
        self.time += delta_t
        # Add grains above frame
        n = delta_t*sample_distribution(self.config['frequency_distribution']) + self.freq_error
        num_grains = int(n)
        print(f'Adding {num_grains} grains.')
        self.freq_error = n - num_grains
        for i in range(num_grains):
            self.add_grain()
        # Apply timestep
        for grain in self.grains:
            grain.step(delta_t)
        # Remove grains that've fallen out of frame
        self.grains = [grain for grain in self.grains if grain.position[1,0] > -0.5]
        
        
    def render_state(self):
        '''
        FRAME SIZE IS NORMALIZED. Supply input data appropriately.
        '''
        fig = plt.figure()

        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=90, azim=-90, roll=0)

        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_zlim([0,1])
        ax.set_aspect('equal')

        fig.subplots_adjust(0,0,1,1) # Should remove white border

        if self.background is not None:
            ax.imshow(self.background, extent=[-1, 1, -1, 1], origin='upper')

        for grain in self.grains:
            grain.display(ax)

        return fig, ax


    def save_image(self):
        outputFilename = f'timestamp-{self.time}'.replace('.', '_') + '.png'
        outputFilepath = os.path.join(self.config['output_image_dir'], outputFilename)
        fig, _ = self.render_state()
        fig, ax = self.render_state()
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = img[:, 80:img.shape[1]-80]
        cv2.imwrite(outputFilepath, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.close('all')
        return outputFilepath

        
    def record_state(self):
        imageFp = self.save_image()
        recordDict = {
            'time':self.time,
            'image_filepath':imageFp,
            'grains':[{
                'id':grain.id,
                'feret_min':grain.feret_min,
                'volume':grain.volume,
                'layers':{i:layer.tolist() for i, layer in enumerate(grain.layers)},
                'state': {
                    'position':grain.position.tolist(),
                    'velocity':grain.velocity.tolist(),
                    'orientation':grain.orientation.tolist(),
                    'angular_velocity':grain.angular_velocity.tolist(),   
                    }
                } for grain in self.grains]
        }
        jsonStr = json.dumps(recordDict)
        with open(self.config['output_log_filepath'], 'a') as file:
            file.write(jsonStr+'\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",)
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = json.load(file)
    simulator = SandSimulator(config)
    simulator.run_simulation()