import numpy as np
from scipy import stats
#from Distribution import Distribution  #wrapper for stats distribution, samples bulkwise
import itertools
import networkx as nx
import network2tikz
import matplotlib.pyplot as plt
import datetime
import os
import tikzplotlib as plt2tikz

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
plt.rcParams['text.usetex'] = True

class HyperbolicGraphSimulation:

    def __init__(self,seed=123):
        #bulkwise sampler is quicker then sampling one by one
        self.uniformSampler = Distribution(stats.uniform(), seed)
        self.dir = ""
        
    def setOutputFolder(self, dir):
        if not os.path.isdir(dir): os.mkdir(dir)
        self.dir = dir 

    def generateGraph(self, n, average_degrees, negative_curvature):
        self.v = average_degrees
        self.a = negative_curvature
        self.R = 2*np.log(n/self.v)

        self.samplePoints(n)

        self.G = nx.Graph()
        # add all nodes
        for i in self.pointsPositions.keys():
            self.G.add_node(i)
        # add all edges between nodes if distance is smaller then radius
        for pair in itertools.combinations(self.pointsPositions.keys(), r=2):
            p1 = self.pointsPositions[pair[0]]
            p2 = self.pointsPositions[pair[1]]
            if self.calculateDistance(p1, p2) <= self.R:
                self.G.add_edge(pair[0], pair[1])

    '''
    returns given size of random random points
    '''
    def samplePoints(self, size):
        self.pointsPositions = {}
        for i in range(size):
            self.pointsPositions[i] = (self.sampleRadius(), self.sampleAngle())

    '''
    returns random radius
    '''
    def sampleRadius(self):

        '''the inverse of the density function'''         
        def invDensityRadius(y):
             return np.arcsinh((y*(np.cosh(self.a*self.R)-1))/self.a) / self.a

        return invDensityRadius(self.uniformSampler.rvs())

    '''
    returns random angle
    '''
    def sampleAngle(self):
        return self.uniformSampler.rvs() * 2* np.pi

    '''
    returns hyperbolic distance between two points
    '''
    def calculateDistance(self, p1, p2):
        #p1= (r, angle), p2 = (r', angle')
        dist_ang = np.pi - abs(np.pi - abs(p1[1]-p2[1]))
        return np.arccosh(np.cosh(p1[0])*np.cosh(p2[0]) - np.sinh(p1[0])*np.sinh(p2[0])* np.cos(dist_ang))

    '''
    convert polar coordinates to euclidian coordinates
    '''
    def convertPolarToEuclidian(self, p):
        #p= (r, angle)
        x = p[0]*np.cos(p[1])
        y = p[0]*np.sin(p[1])
        return (x, y)

    '''
    output to tex and display graph
    '''
    def draw(self, part=None, fp="", style="", show = False):
        #get default filepath
        if fp == "":
            fp = datetime.datetime.now().strftime("%Hh%Mm%Ss %d-%m-%Y") +".tikz"
        
        #get the positions of all nodes in euclidian coordinates
        layout = {k : self.convertPolarToEuclidian(self.pointsPositions[k]) for k in self.pointsPositions.keys()}
        #convert the network to tikz
        network2tikz.plot(self.G, self.dir+fp.replace('.tikz', '.tex'), layout=layout, canvas=(10,10))
        style = ["\SetVertexStyle[MinSize=0.3\DefaultUnit, LineWidth=0pt, FillColor=mycolor1]\n", "\SetEdgeStyle[LineWidth=0.25pt]\n"]
        self.addStyleString(fp, style)

        #display the matplotlib view of the graph
        cmap = 'jet'
        nc = [p**.45 for p in part.values()]
        if show:
            if part:
                nx.draw(self.G, pos=layout, node_size=50, node_color=nc, width=.5, edgecolors='k',cmap=cmap)
            else:
                nx.draw(self.G, pos=layout, node_size=50, node_color='forestgreen', width=.5, edgecolors='k',cmap=cmap)
            plt.grid(True)
            plt.show()

    '''
    add the given style options to the output.tex at appropriate place
    '''
    def addStyleString(self, fp, style):

        #read in the output file and interlace the styling options
        allLines = []
        shouldSave = False #filter to only save the tikz picture
        with open(self.dir+fp.replace('.tikz', '.tex'), "r") as f:
            for line in f:
                if not shouldSave:
                    if "begin{tikzpicture}" in line: #start of tikzpicutre
                        for style_line in style:
                            allLines.append(style_line)
                        shouldSave = True
                if shouldSave:
                    allLines.append(line)
                    if "end{tikzpicture}" in line: #end of tikzpicutre
                        shouldSave = False

        #output the file again
        with open(self.dir+fp.replace('.tex', '.tikz'), "w") as f:
            for line in allLines:
                f.write(line)

        #remove original .tex file
        os.remove(self.dir+fp.replace('.tikz', '.tex'))   
    
    '''
    saves the degree distribution of the currently generated graph

    parameters: 
    fp: filepath to save file to
    show: whether to display the plot on screen
    '''
    def saveDegreeDistribution(self, fp, show=False):
        print("Saving: " + fp)
        assert ".tikz" in fp, "filepath does not end in .tikz"

        degrees = [self.G.degree(n) for n in self.G.nodes()]
        degrees.sort()
        total = sum(degrees)
        cumulative = [sum(degrees[n::])/total for n in range(len(degrees))]
        prev = degrees[0]
        index = [prev]
        freq = [cumulative[0]]
        for i in range(1, len(degrees)):
            if not (degrees[i] == prev):
                prev = degrees[i]
                index.append(degrees[i])
                freq.append(cumulative[i])
                
        plt.loglog(index, freq)

        plt2tikz.save(self.dir+fp) #output

        if show:
            plt.show()
        
        plt.clf() #clear plot



'''
The scipy stats package has great functionality for probability distributions
and random variables, but its random number generator is extremely slow
when one samples only one random number at a time. It is extremely fast,
however, when one samples multiple random numbers simultaneously.
For this reason, we have created this class that acts somewhat as a wrapper
around the scipy probability distributions. It will make sure that random
numbers are always generated in batches of n = 10000, and the rvs() function
simply returns the next random number from this list (and resamples when necessary).

@author: Marko Boon
'''


class Distribution :

    n = 10000 # standard random numbers to generate

    '''
    Constructor for this Distribution class.
    
    Args:
            dist (scipy.stats random variable): A random variable from the scipy stats libary.
    
    Attributes:
            dist (scipy.stats random variable): A random variable from the scipy stats libary.
            n (int): a number indicating how many random numbers should be generated in one batch
            randomNumbers: a list of n random numbers generated from 'dist'
            idx (int): a number keeping track of how many random numbers have been sampled
    
    '''

    def __init__(self, dist, seed=123):
        self.dist = dist
        self.seed=seed
        self.resample()
        
    def __str__(self):
        return str(self.dist)
    
    def resample(self):
        self.randomNumbers = self.dist.rvs(self.n, random_state=self.seed)
        self.idx = 0
    
    def rvs(self, n=1):
        '''
        A function that returns n (=1 by default) random numbers from the specified distribution.
        
        Returns:
            One random number (float) if n=1, and a list of n random numbers otherwise.
        '''
        if self.idx >= self.n - n :
            while n > self.n :
                self.n *= 10
            self.resample()
        if n == 1 :
            rs = self.randomNumbers[self.idx]
        else :
            rs = self.randomNumbers[self.idx:(self.idx+n)]
        self.idx += n
        return rs

    def mean(self):
        return self.dist.mean()
    
    def std(self):
        return self.dist.std()
    
    def var(self):
        return self.dist.var()
    
    def cdf(self, x):
        return self.dist.cdf(x)

