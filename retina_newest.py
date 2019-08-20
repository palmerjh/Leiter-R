import math
import numpy as np
#import heapq as hq
from random import randrange
from PIL import Image, ImageDraw
from copy import deepcopy

from collections import namedtuple, deque
RelativeRetina = namedtuple("RelativeRetina","ret start") # simple container used to hold a Retina object and its location in its TargetSlice
Result = namedtuple("Result","start wedge_index sim")	# simple container used to hold results for primary,secondary,etc. regions for each choice
FlipResult = namedtuple("Result","start wedge_index flipped sim")

pi = math.pi

# used if nRings isn't specified in Template ctor
# if less than 1, is fraction of total_size
# else is the number of pixels
# *****only an approximation
default_blindspot_diameter = 0.2

fix_hn = True	# if False, input hn (default or ctor) is ideal; hn will be shifted lower to get integer number of nWedges and maintain radius of retina as size / 2
				# else input hn is exact and retinal radius and Rn will be lowered to get integer number of nWedges

# used if hn isn't specified in Template ctor
default_hn = 1

# # used if threshold isn't specified in TargetSlice ctor
# # is fraction of densest region of target
# # purpose: discard anomalous blackest regions
# # Note: no longer used now that variance is the slice selection heuristic
# default_threshold = 0.75

# threshold below which is considered a "transparent pixel"
threshold = 0.05

bs_bias = 2	# weighting of blindspot pixels in density search relative to those in retina

visual_h0 = 2	# how large sectors in the innermost ring should be represented in the visualization

class Template(object):
	"""Performs all the calculations to construct a Retina object but does not actually create one
	   Useful for finding weights before pixels to be converted are known"""

	# initialized in findWeights()
	unrefined_density_wts = None
	ring_area_wts = None

	# initialized in createVisualizer()
	visualization = None

	def __init__(self, size, nRings=None, blindspot_diameter=default_blindspot_diameter, hn=None):
		self.size = size 	# side length of square template
		midpoint = (self.size - 1) / 2.0
		self.center = np.array([midpoint,midpoint])

		hn_ideal = default_hn if hn == None else hn
		Rn_ideal = 0.5 * (self.size - hn_ideal)
		nWedges_ideal = 2*pi * Rn_ideal / hn_ideal

		if fix_hn:
			self.hn = hn_ideal
			self.nWedges = int(math.floor(nWedges_ideal))
		else:
			self.nWedges = int(math.ceil(nWedges_ideal))
			self.hn = pi * self.size / (self.nWedges + pi)

		self.Rn = self.nWedges * self.hn / (2*pi)
		self.retina_radius = self.Rn + 0.5*self.hn 		# equals self.size / 2.0 if not fix_hn

		self.ratio = (2*self.Rn - self.hn) / (2*self.Rn + self.hn)
		print self.Rn, self.ratio, self.hn
		self.radius_split = math.log(1 - self.hn/(2*self.Rn),self.ratio)

		self.nRings = 42	# placeholder

		self.bs_radius = -1	# placeholder

		if nRings is None:
			if blindspot_diameter < 1:
				bs_radius_ideal = 0.5*(blindspot_diameter * self.size)
			else:
				bs_radius_ideal = 0.5*(blindspot_diameter)

			self.nRings -= self.findRing(bs_radius_ideal)
		else:
			self.nRings = nRings

		self.bs_radius = self.Rn*math.pow(self.ratio,self.nRings - 1 + self.radius_split)

		print('size: %d x %d' % (self.size, self.size))
		print('number of rings: %d' % self.nRings)
		print('number of wedges: %d' % self.nWedges)
		print('blindspot_radius: %f\n' % self.bs_radius)

		# Initializes self.unrefined_density_wts and self.ring_area_wts
		#
		# self.unrefined_density_wts are 1.0 in retina, 1.0 * bs_bias in blindspot, and 0.0 otherwise
		# used in initial densest search
		#
		# self.ring_area_wts are 1.0 in retina, 0.0 otherwise
		# used in calculating variance for selecting slice regions (in TargetSlice class)
		self.findWeights()

		# Returns: closure that accepts pix and creates Retina object, closure that accepts pix and calculates density
		self.createRetina, self.calcDensity = self.createClosures()

		# closure that accepts Retina object and visualizes it
		self.visualize = self.createVisualizer()

	# finds weights for both unrefined_density_wts calculations as well as
	# identifying the pixels that would map to retinal sectors (i.e. excludes
	# blindspot and areas outside of retina); used for calculating STD
	def findWeights(self):
		unrefined_density_wts = np.ones((self.size,self.size))
		ring_area_wts = np.ones((self.size,self.size))

		for px in range(self.size):
			for py in range(self.size):
				p = np.array([px,py])
				diff = p - self.center
				ring = self.findRing(norm(diff))

				if (ring >= self.nRings):
					unrefined_density_wts[px,py] = 0.0
					ring_area_wts[px,py] = 0.0
				elif (ring == -1):
					unrefined_density_wts[px,py] *= bs_bias
					ring_area_wts[px,py] = 0.0

		self.unrefined_density_wts = unrefined_density_wts
		self.ring_area_wts = ring_area_wts

	def findRing(self,r):
		#print r
		if (r < self.bs_radius):
			return -1	# inside central, "special" pixel

		level = math.log(r/self.Rn,self.ratio)
		if (level < 0):
			if (self.radius_split - 1 < level):
				return self.nRings - 1
			else:
				return self.nRings

		level_lb = int(level)

		if (level - level_lb < self.radius_split):
			return self.nRings - 1 - level_lb
		else:
			return self.nRings - 1 - (level_lb + 1)

	# locates which wedge/ring sector the pixel is inside
	def findSector(self,pixel):
		p = np.array(pixel)
		diff = p - self.center
		ring = self.findRing(norm(diff))

		angle = math.atan2(diff[1],diff[0])
		if (angle < 0):
			angle += 2*pi

		wedge = int(self.nWedges * angle / (2*pi))

		return wedge, ring

	# returns set of pixels within ring/wedge sector starting with closest pixel to its center (which may or may not actually be in sector)
	def findPixels(self,sector,start):
		pixels = []
		# BFS
		q = deque()
		q.append(start)
		while len(q) > 0:
			cur_pixel = q.popleft()
			if self.findSector(cur_pixel) == sector:
				pixels.append(cur_pixel)
				x,y = cur_pixel
				for pixel in [(x-1,y-1),(x-1,y),(x-1,y+1),
								(x,y-1),          (x,y+1),
								(x+1,y-1),(x+1,y),(x+1,y+1)]:
					if pixel not in q and pixel not in pixels:
						q.append(pixel)

		return pixels

	# Creates the following closures:
	#
	# createRetina(pix): for each sector, finds set of pixels within ring/wedge sector
	# unioned with the singleton set of the pixel closest to the sector's center
	# then it uses this to create a closure that generates a retina object
	#
	# calcDensity(retina,pix): for each ring, unions all of the pixels within each of its wedges
	# then it uses this to create a closure that calculates the density
	#
	# calcRingwiseVar(pix): calculates ringwise variance ONLY within rings
	# does not include pixels within imaginary blindspot or outside retina
	# used in selecting regions in slices --- want most variance in rings
	# to maximize information and disambiguate
	def createClosures(self):
		sector_pixels = np.empty(self.retina_size(),dtype=list)		# list of pixels in each sector for each sector
		density_ring_mappers = []									# mappers that channel pixels to the correct ring for each ring
		for i in range(self.nRings):
			density_ring_mappers.append(set([]))

		for wedge in range(self.nWedges):
			central_angle = (wedge + 0.5) / self.nWedges * 2*pi
			unit_vector = np.array([math.cos(central_angle),math.sin(central_angle)])

			for ring in range(self.nRings):
				R_ring = self.Rn*math.pow(self.ratio,self.nRings-1-ring)
				sector_center = self.center + R_ring * unit_vector
				closest_pixel = (int(round(sector_center[0])),int(round(sector_center[1])))

				pixels_inside_sector = self.findPixels((wedge,ring),closest_pixel)

				sector_pixels[wedge,ring] = pixels_inside_sector
				if closest_pixel not in pixels_inside_sector:
					sector_pixels[wedge,ring].append(closest_pixel)

				density_ring_mappers[ring] = density_ring_mappers[ring].union(sector_pixels[wedge,ring])

		density_bs_mapper = []				# list of pixels that map to blindspot
		for x in range(self.size):
			for y in range(self.size):
				if dist(np.array([x,y]),self.center) < self.bs_radius:
					density_bs_mapper.append((x,y))

		density_wts = np.ones((self.nRings+1,))
		density_wts[0] = bs_bias

		density_mappers = [density_bs_mapper] + density_ring_mappers

		# closure that creates Retina object
		def createRetina(pix, relevant=False):
			retina = np.empty(self.retina_size(),dtype=list)
			for wedge in range(self.nWedges):
				for ring in range(self.nRings):
					retina[wedge,ring] = [pix[sector_pixels[wedge,ring][0]]]	# each sector has at least one pixel mapped to it - its closest pixel
					for other_pixel in sector_pixels[wedge,ring][1:]:
						retina[wedge,ring].append(pix[other_pixel])

			retina = np.array([[np.average(values) for values in row] for row in retina])

			return Retina(self,pix,retina,relevant=relevant)

		# closure that calculates density
		def calcDensity(retina=None,pixels=None,relevant=False):
			if retina is None:
				pix = pixels
			else:
				pix = retina.pix

			densities = []
			min_densities = []
			max_densities = []
			white_counts = []
			transparency_counts = []

			# must consider transparent pixels
			if relevant:
				relevant_pix = pix >= 0
				white_pix = pix < threshold
				for mapper in density_mappers:
					pix_values = [pix[pixel] for pixel in mapper]
					wts = [relevant_pix[pixel] for pixel in mapper]
					nRelevantPix = sum(wts)
					relevant_density = -1
					if nRelevantPix == 0:
						relevant_density = 0
					else:
						relevant_density = np.average(pix_values,weights=wts)

					relevant_ratio = float(nRelevantPix) / len(wts)
					# all transparent pixel values are 0
					min_density = relevant_ratio * relevant_density
					# all transparent pixel values are 1
					max_density = min_density + (1 - relevant_ratio)

					densities.append(relevant_density)
					min_densities.append(min_density)
					max_densities.append(max_density)

					isWhite = [white_pix[pixel] for pixel in mapper]
					white_counts.append(sum(isWhite))
					transparency_counts.append(len(wts) - nRelevantPix)

					# print nRelevantPix, len(wts)
					# print relevant_density, min_density, max_density

				densities = np.array(densities)
				min_densities = np.array(min_densities)
				max_densities = np.array(max_densities)

			else:
				# list comprehension that holds the averages of values of pixels mapping to blindspot ([0]) and ring r ([r+1]) respectively
				# thus len(densities) = 1 + self.nRings
				densities = np.array([np.mean([pix[pixel] for pixel in mapper]) for mapper in density_mappers])
				min_densities = densities
				max_densities = densities

			avg_densities = 0.5*(max_densities + min_densities)
			buffer_densities = 0.5*(max_densities - min_densities)

			# print avg_densities
			# print buffer_densities

			if pixels is None:
				retina.density_mappers = density_mappers
				retina.densities = densities
				retina.density_wts = density_wts

				retina.min_densities = min_densities
				retina.max_densities = max_densities
				retina.avg_densities = avg_densities
				retina.buffer_densities = buffer_densities

				retina.white_counts = np.array(white_counts)
				retina.transparency_counts = np.array(transparency_counts)

			return np.average(densities,weights=density_wts)

		# # closure that calculates density
		# def calcDensity(retina=None,pixels=None):
		# 	if retina is None:
		# 		pix = pixels
		# 	else:
		# 		pix = retina.pix
		#
		# 	# list comprehension that holds the averages of values of pixels mapping to blindspot ([0]) and ring r ([r+1]) respectively
		# 	# thus len(densities) = 1 + self.nRings
		# 	densities = np.array([np.mean([pix[pixel] for pixel in mapper]) for mapper in density_mappers])
		#
		# 	if pixels is None:
		# 		retina.density_mappers = density_mappers
		# 		retina.densities = densities
		# 		retina.density_wts = density_wts
		#
		# 	return np.average(densities,weights=density_wts)

		# closure that calculates rignwise variance; TODO if interested (already have simpler variance)
		def calcRingwiseVar(pix):
			pass

		return createRetina, calcDensity

	def calcVar(self, pix):
		relevant_pix = pix >= 0
		wts = self.ring_area_wts * relevant_pix
		if np.sum(wts) == 0:
			return 0.0
		# calculates average of retina pix
		avg = np.average(pix, weights=wts)
		# only considers variance of retina pixels w.r.t. retina pix average
		var = np.average((pix-avg)**2, weights=wts)

		return var

	# creates template retina visualizer (using ImageDraw to draw each ring/wedge sector)
	# then it uses this to create a closure that will take retina object info and visualize it
	def createVisualizer(self):
		h0 = self.hn*math.pow(self.ratio,self.nRings-1)
		factor = visual_h0 / h0 	# factor by which retina must be scaled for visualization

		visual_size = int(math.ceil(factor * self.size))
		visual_midpoint = (visual_size - 1) / 2.0
		visual_center = np.array([visual_midpoint,visual_midpoint])
		visual_retina_radius = factor * self.retina_radius

		im = Image.new('RGB',(visual_size,visual_size))
		draw = ImageDraw.Draw(im)

		delta_angle = 2*pi / self.nWedges
		angle = 0
		start = tuple(visual_center)

		# this draws all the "spokes" of the retina
		for i in range(self.nWedges):
			end_unit = np.array([math.sin(angle),math.cos(angle)])		# note: coordinates are flipped because draw.line uses diff. coord. system
			end = tuple(visual_center + visual_retina_radius*end_unit)

			draw.line([start,end],fill='red')
			angle += delta_angle

		# this draws the circles that partition the retina into rings
		for ring in range(-1,self.nRings):
			radius = factor * self.Rn*math.pow(self.ratio,ring + self.radius_split)
			xy = (visual_midpoint - radius, visual_midpoint - radius,
				  visual_midpoint + radius, visual_midpoint + radius)	# bounding box for circular ring partition

			if ring < self.nRings - 1:
				draw.ellipse(xy,outline='red')
			else:
				draw.ellipse(xy,fill='black',outline='red')	# erases extraneous wedge lines that clutter the blindspot

		self.visualization = im

		visual_pix = np.asarray(im)
		mapper = np.empty((visual_size,visual_size),dtype=list)		# initializes each element to None super quickly; 1000 times faster than using fill()
		for x in range(visual_size):
			for y in range(visual_size):
				if np.array_equal(visual_pix[x,y],[255,0,0]):
					continue

				p = np.array([x,y])
				diff = p - visual_center
				dist = norm(diff)

				if dist > visual_retina_radius:
					continue

				ring = self.findRing(dist / factor)

				if ring == -1:	# inside blindspot
					continue

				angle = math.atan2(diff[1],diff[0])
				if (angle < 0):
					angle += 2*pi

				wedge = int(self.nWedges * angle / (2*pi))

				mapper[x,y] = (wedge,ring)

		def visualize(retina,fname=None):
			visual_pix.flags.writeable = True	# need this for some reason

			r_pix = retina.retina.copy()
			transparent_r_pix = r_pix < 0

			# converts from our faux grayscale values to "real" ones from 0 to 255 (unrounded)
			r_pix = (1.0 - r_pix) * 255

			# converts each grayscale value v to RGB list [v,v,v]
			r_pix = np.array([[[x,x,x] for x in [int(round(y)) for y in z]] for z in r_pix])

			for x in range(visual_size):
				for y in range(visual_size):
					index = mapper[x,y]
					if index is None:
						continue

					if transparent_r_pix[index]:
						# light green color for transparent pixels
						visual_pix[x,y] = [127,255,0]
					else:
						visual_pix[x,y] = r_pix[index]

			retina.visualization = Image.fromarray(visual_pix.astype('uint8'))

			if not fname is None:
				retina.visualization.save(fname)

			return retina.visualization

		return visualize

	# assumes file is square image with size = self.size
	# TODO add error checking
	def file2Retina(self,file):
		pix = file2Pix(file)

		self.createRetina(pix)

	def printRings(self):
		for i in range(self.nRings):
			print('%d\t%f\t%f' % (self.nRings-1-i,self.Rn*math.pow(self.ratio,i),self.hn*math.pow(self.ratio,i)))

	# Note: pix must be of dimension self.size by self.size (same as self.unrefined_density_wts)
	# does not distinguish between various pixel densities
	# good for initial densest search in TargetSlice; faster than refined ring-based density calculation
	# bad for later density comparison search in choices - use self.calcDensity(pixels=pix) for that
	def calcUnrefinedDensity(self,pix):
		return np.average(pix,weights=self.unrefined_density_wts)
		'''
		return np.mean([np.average(1-pix,weights=self.wts),
						np.average(pix,weights=self.inverse_wts)])
		'''

	def retina_size(self):
		return (self.nWedges,self.nRings)

# Note: requires template object to create
class Retina(object):
	"""Numpy array of pixels arranged in polar fashion"""

	# initialized or updated whenever self.visualize is called
	# PIL.Image object
	visualization = None

	density = -1				# ring-based; calculated only if explicity asked to with getDensity()
	unrefinedDensity = -1		# calculated only if explicity asked to with getUnrefinedDensity()

	variance = -1 				# variance of pixels within retinal rings

	# -------------------------------------------------------------------------------------------------
	# the following nine arrays are instantiated when self.getDensity() is called for the first time
	# used in calculating self.ringwiseDensityDelta(pix2)

	# density_mappers[0] is density_bs_mapper and density_mappers[r+1] is mapper for ring r
	density_mappers = None
	# densities[0] is bs_density and densities[r+1] is density of ring r
	densities = None
	density_wts = None

	# differs from densities iff transparent pixels are possible (relevant flag to template calcDensity is True)
	min_densities = None
	max_densities = None

	# used in ringwiseDensityDelta
	avg_densities = None
	buffer_densities = None
	white_counts = None
	transparency_counts = None
	# -------------------------------------------------------------------------------------------------

	def __init__(self, template, pix, retina, relevant=False):
		self.template = template
		self.pix = pix
		self.retina = retina
		self.relevant = relevant

	# if pix is not specified, returns deep copy
	# else returns new Retina object with same template but different pix
	def copy(self,pix=None):
		if pix == None:
			return deepcopy(self)
		else:
			return self.template.createRetina(pix,relevant=self.relevant)

	def getDensity(self):
		if self.density == -1:
			self.density = self.template.calcDensity(retina=self,relevant=self.relevant)

		return self.density

	def getUnrefinedDensity(self):
		if self.unrefinedDensity == -1:
			self.unrefinedDensity = self.template.calcUnrefinedDensity(self.pix)

		return self.unrefinedDensity

	def getVar(self):
		if self.variance == -1:
			self.variance = self.template.calcVar(self.pix)

		return self.variance

	def findRing(self,r):
		return self.template.findRing(r)

	def findSector(self,pixel):
		return self.template.findSector(pixel)

	def printRings(self):
		return self.template.printRings()

	def pix_size(self):
		return self.pix.shape

	def retina_size(self):
		return self.template.retina_size()

	# calculates the density deltas of the bs & ring regions of self.pix and another pix2 and averages based on self.density_wts
	def ringwiseDensityDelta(self,pix2):
		# list comprehension that holds the averages of values of pix2 pixels mapping to blindspot ([0]) and ring r ([r+1]) respectively
		# thus len(pix2_densities) = 1 + self.template.nRings
		pix2_densities = np.array([np.mean([pix2[pixel] for pixel in mapper]) for mapper in self.density_mappers])
		if not self.relevant:
			return np.average(deltas,weights=self.density_wts)

		white_pix2 = pix2 < threshold
		white_counts2 = np.array([sum([white_pix2[pixel] for pixel in mapper]) for mapper in self.density_mappers])
		# since "transparent" pixels are labelled based on their whiteness,
		# some of these may actually indeed be white in the easel
		# (i.e. some transparent pixels are false positives)
		# thus this gives an extra boost to regions where white overlapped in
		# both the target slice and the easel slot
		isDiscount = white_counts2 <= self.white_counts
		discounts = white_counts2 - (self.white_counts - self.transparency_counts)
		discounts *= isDiscount
		discounts = discounts.astype('float64') * threshold / (self.transparency_counts+1) * 0.0

		# 0 iff density in between min and max density; otherwise delta is closest distance to either min or max_density
		deltas = np.maximum(abs(pix2_densities - self.avg_densities) - self.buffer_densities, 0)

		# for i in range(len(discounts)):
		# 	print self.white_counts[i], self.transparency_counts[i], white_counts2[i]
		# 	print discounts[i], deltas[i], deltas[i] - discounts[i]

		# return np.average(abs(self.densities - pix2_densities),weights=self.density_wts)
		return np.average(deltas - discounts,weights=self.density_wts)

	def rotate(self,nRotations=None):
		if nRotations == None:
			self.retina = np.roll(self.retina, 1, axis=0)
		else:
			self.retina = np.roll(self.retina, nRotations, axis=0)

	# saves and returns retina as PIL.Image object
	def save(self,fname):
		temp = (1.0 - self.retina) * 255
		temp = np.array([[int(round(value)) for value in row] for row in temp])
		im = Image.fromarray(temp.astype('uint8'))

		im.save(fname)

		return im

	def visualize(self,fname=None):
		return self.template.visualize(self,fname)

class Slice(object):
	"""Creates retina object for entire slice for slice approach"""
	def __init__(self, taskpic, prob_name, t_name, sliceRet=True, useTransparentPix=False, template=None, size=None, nRings=None, hn=None):
		self.taskpic = taskpic
		self.pix = taskpic.pix
		self.transparent_pix = self.taskpic.getTransparent()
		self.prob_name = prob_name
		self.t_name = t_name
		self.useTransparentPix = useTransparentPix

		if template is None:
			self.size = min(self.pix.shape) - 1 if size is None else size
			self.template = Template(self.size,nRings,hn)
		else:
			self.size = template.size
			self.template = template

		if sliceRet:
			if self.useTransparentPix:
				self.ret = self.template.createRetina(self.transparent_pix,relevant=True)
			else:
				self.ret = self.template.createRetina(self.pix,relevant=False)

			self.ret.getDensity()

		self.results = []

	# flips slice and returns new TargetSlice
	def flip(self,direction='horizontal'):
		t_name = self.t_name + '_flipped'
		return Slice(self.taskpic.flip(direction), self.prob_name, t_name, useTransparentPix=self.useTransparentPix, template=self.template)

	def getTransparent(self):
		return self.taskpic.getTransparent()

	# returns deep copy
	def copy(self):
		return deepcopy(self)

class Subslice(Slice):
	"""Collection of Retina Objects used in Subslice Approach"""
	def __init__(self, taskpic, prob_name, t_name, useTransparentPix=False, template=None, size=None, nRings=None, hn=None, nRegions=2):
		super(Subslice,self).__init__(taskpic, prob_name, t_name, sliceRet=False, useTransparentPix=useTransparentPix,template=template, size=size, nRings=nRings, hn=hn)
		self.ordered_variances = self.findVariances()	# format: (variance,start)
		self.primary = self.createRelativeRetina(0)
		self.secondary = self.createRelativeRetina(self.findSecondaryIndex())

		self.sp_difference = self.secondary.start - self.primary.start
		self.sp_dist = norm(self.sp_difference)
		self.sp_angle = math.atan2(self.sp_difference[1],self.sp_difference[0])

		# Note: for primary or secondary, the most_sim heuristic only depends on maximising the similarity of either the primary or secondary region
		# Note: for composite, the most_sim heuristic depends on maximising both the similarity of the primary and secondary regions
		self.results = {'primary':	[],		# format: (primary_result, associated_secondary_result) for each choice
						'secondary':[],		# format: (associated_primary_result, secondary_result) for each choice
						'composite':[]}		# format: (coupled_primary_result, coupled_secondary_result) for each choice

		self.nRegions = nRegions
		if nRegions == 3:
			self.tertiary = self.createRelativeRetina(self.findTertiaryIndex())

			self.ts_difference = self.tertiary.start - self.secondary.start
			self.ts_dist = norm(self.ts_difference)
			self.ts_angle = math.atan2(self.ts_difference[1],self.ts_difference[0])

			self.results.update({'tertiary':	[]})

	def findSecondaryIndex(self):
		p_start = self.primary.start
		ratio = 2.0
		while True:
			ratio *= 0.5
			for i, (_, s_start) in enumerate(self.ordered_variances):
				sp_distance = dist(p_start,s_start)
				if self.goldilocks2(sp_distance,ratio=ratio):
					return i

	def findTertiaryIndex(self):
		p_start = self.primary.start
		s_start = self.secondary.start
		sameSize = self.size == min(self.taskpic.size())
		ratio = 1.0
		while True:
			ratio *= 0.5
			for i, (_, t_start) in enumerate(self.ordered_variances):
				ts_distance = dist(t_start,s_start)

				tp_difference = t_start - p_start
				tp_distance = norm(tp_difference)
				tp_angle = math.atan2(tp_difference[1],tp_difference[0])
				d_angle = tp_angle - self.sp_angle
				isValidTriangle = sameSize or abs(math.cos(d_angle)) < 0.9999
				if isValidTriangle and self.goldilocks3(ts_distance,tp_distance,ratio=ratio):
					return i

	# flips slice and returns new TargetSlice
	def flip(self,direction='horizontal'):
		t_name = self.t_name + '_flipped'
		flipped = Subslice(self.taskpic.flip(direction), self.prob_name, t_name, \
						useTransparentPix=self.useTransparentPix, template=self.template, nRegions=self.nRegions)

		assert self.primary.start[0] == flipped.primary.start[0]
		assert self.secondary.start[0] == flipped.secondary.start[0]
		assert abs(self.sp_dist - flipped.sp_dist) < 0.00001
		# angles are either the opposite sign or both equal to pi
		assert self.sp_angle + flipped.sp_angle < 0.00001 or abs(self.sp_angle - flipped.sp_angle) < 0.00001

		# if self.nRegions == 3:
		# 	assert self.tertiary.start[0] == flipped.tertiary.start[0]
		# 	assert abs(self.ts_dist - flipped.ts_dist) < 0.00001
		# 	# angles are either the opposite sign or both equal to pi
		# 	assert self.ts_angle + flipped.ts_angle < 0.00001 or abs(self.ts_angle - flipped.ts_angle) < 0.00001

		# print self.ts_dist, flipped.ts_dist
		# print self.ts_angle, flipped.ts_angle

		'''
		print(self.primary.start)
		print(self.secondary.start)
		print(self.sp_dist)
		print(self.sp_angle)

		print(flipped.primary.start)
		print(flipped.secondary.start)
		print(flipped.sp_dist)
		print(flipped.sp_angle)
		'''

		return flipped

	# checks to see if distance is greater than a ratio of retina size and smaller than an acceptable margin of error
	def goldilocks2(self,distance,ratio=1.0):
		return self.size * ratio <= distance <= self.template.nWedges * self.size / (2*pi) # margin of error is more than self.size

	# checks to see if distance to s and p regions is greater than a ratio of retina size and smaller than an acceptable margin of error
	def goldilocks3(self,ts_distance,tp_distance,ratio=1.0):
		farEnough = self.size * ratio <= ts_distance and self.size * ratio <= tp_distance
		closeEnough = ts_distance <= self.template.nWedges * self.size / (2*pi) # margin of error is more than self.size
		return farEnough and closeEnough

	# i is the index of self.ordered_unrefined_densities at which the RelativeRetina container will be created
	def createRelativeRetina(self,i):
		pix = self.transparent_pix if self.useTransparentPix else self.pix
		var, start = self.ordered_variances[i]
		sx, sy = start

		ret = self.template.createRetina(pix[sx:sx+self.size,sy:sy+self.size],relevant=self.useTransparentPix)
		#ret.unrefinedDensity = unrefinedDensity
		ret.variance = var

		ret.getDensity()

		return RelativeRetina(ret,np.array(start))

	def findVariances(self):
		pix = self.transparent_pix if self.useTransparentPix else self.pix
		relevant_pix = pix >= 0
		variances = []
		for sx in range(self.pix.shape[0] - self.size + 1):
			for sy in range(self.pix.shape[1] - self.size + 1):
				var = self.template.calcVar(pix[sx:sx+self.size,sy:sy+self.size])
				ratio_relative = np.sum(relevant_pix[sx:sx+self.size,sy:sy+self.size])
				ratio_relative /= float(self.size**2)
				var *= ratio_relative
				variances.append((var,(sx,sy)))

		variances.sort(reverse=True)

		return variances

	def findUnrefinedDensities(self):
		pix = self.transparent_pix if self.useTransparentPix else self.pix
		unrefinedDensities = []
		for sx in range(self.pix.shape[0] - self.size + 1):
			for sy in range(self.pix.shape[1] - self.size + 1):
				unrefinedDensities.append((self.template.calcUnrefinedDensity(pix[sx:sx+self.size,sy:sy+self.size]),(sx,sy)))

		unrefinedDensities.sort(reverse=True)

		cutOff = -1
		for i in range(len(unrefinedDensities)):
			if unrefinedDensities[i][0] < self.threshold*unrefinedDensities[0][0]:
				cutOff = i
				break

		return unrefinedDensities[cutOff:]

class HybridSlice(Slice):
	"""Collection of Retina Objects used in Hybrid Approach
		- primary subslice density and roational search
		- slice based approach from there """
	def __init__(self, taskpic, prob_name, t_name, useTransparentPix=True, template=None, hn=None, subtemplate=None, subsize=None, sub_hn=None):
		super(HybridSlice,self).__init__(taskpic, prob_name, t_name, useTransparentPix=useTransparentPix, template=template, hn=hn)
		if subtemplate is None:
			self.subsize = subsize
			self.subtemplate = Template(self.subsize,hn=sub_hn)
		else:
			self.subsize = subtemplate.size
			self.subtemplate = subtemplate

		self.ordered_variances = self.findVariances()	# format: (variance,start)
		self.subslice = self.createRelativeRetina(0)

		slice_midpt = (self.size - 1) / 2.0
		self.slice_center = np.array([slice_midpt,slice_midpt])
		subslice_midpt = (self.subsize - 1) / 2.0
		self.subslice_center = self.subslice.start + np.array([subslice_midpt,subslice_midpt])
		# difference from slice center to subslice center
		self.center_delta = self.slice_center - self.subslice_center

		# Note: for primary or secondary, the most_sim heuristic only depends on maximising the similarity of either the primary or secondary region
		# Note: for composite, the most_sim heuristic depends on maximising both the similarity of the primary and secondary regions
		self.results = {'subslice':	[],		# format: (subslice_result, associated_slice_result) for each choice
						'slice':	[],		# format: (associated_subslice_result, slice_result) for each choice
						'hybrid':	[]}		# format: (coupled_subslice_result, coupled_slice_result) for each choice

	# flips slice and returns new TargetSlice
	def flip(self,direction='horizontal'):
		t_name = self.t_name + '_flipped'
		flipped = HybridSlice(self.taskpic.flip(direction), self.prob_name, t_name, \
								useTransparentPix=self.useTransparentPix, template=self.template, \
		 						subtemplate=self.subtemplate)

		assert self.subslice.start[0] == flipped.subslice.start[0]
		assert self.subslice.start[1] == self.size - self.subsize - flipped.subslice.start[1]

		'''
		print(self.subslice.start)

		print(flipped.subslice.start)
		'''

		return flipped

	# i is the index of self.ordered_unrefined_densities at which the RelativeRetina container will be created
	def createRelativeRetina(self,i):
		pix = self.transparent_pix if self.useTransparentPix else self.pix
		var, start = self.ordered_variances[i]
		sx, sy = start

		ret = self.subtemplate.createRetina(pix[sx:sx+self.subsize,sy:sy+self.subsize],relevant=self.useTransparentPix)
		#ret.unrefinedDensity = unrefinedDensity
		ret.variance = var

		ret.getDensity()

		return RelativeRetina(ret,np.array(start))

	def findVariances(self):
		pix = self.transparent_pix if self.useTransparentPix else self.pix
		relevant_pix = pix >= 0
		variances = []
		for sx in range(self.pix.shape[0] - self.subsize + 1):
			for sy in range(self.pix.shape[1] - self.subsize + 1):
				var = self.subtemplate.calcVar(pix[sx:sx+self.subsize,sy:sy+self.subsize])
				ratio_relative = np.sum(relevant_pix[sx:sx+self.subsize,sy:sy+self.subsize])
				ratio_relative /= float(self.subsize**2)
				var *= ratio_relative
				variances.append((var,(sx,sy)))

		variances.sort(reverse=True)

		return variances

	def findUnrefinedDensities(self):
		pix = self.transparent_pix if self.useTransparentPix else self.pix
		unrefinedDensities = []
		for sx in range(self.pix.shape[0] - self.subsize + 1):
			for sy in range(self.pix.shape[1] - self.subsize + 1):
				unrefinedDensities.append((self.subtemplate.calcUnrefinedDensity(pix[sx:sx+self.subsize,sy:sy+self.subsize]),(sx,sy)))

		unrefinedDensities.sort(reverse=True)

		cutOff = -1
		for i in range(len(unrefinedDensities)):
			if unrefinedDensities[i][0] < self.threshold*unrefinedDensities[0][0]:
				cutOff = i
				break

		return unrefinedDensities[cutOff:]

# returns random integer in [a,b) that is not in list of invalids (if specified)
def randomInt(a,b,invalids=None):
	i = randrange(a,b)
	if invalids is None:
		return i

	while i in invalids:
		i = randrange(a,b)

	return i

def calcDensity(pix,nRings=None,hn=None,unrefined=True):
	size,size = pix.shape
	if unrefined:
		return Template(size,nRings,hn).calcUnrefinedDensity(pix)
	else:
		return Template(size,nRings,hn).calcDensity(pixels=pix)

def pix2ret(pix,nRings=None,hn=None):
	size,size = pix.shape
	return Template(size,nRings,hn).createRetina(pix)

def dist(v1,v2):
	#return math.sqrt(math.pow(v1[0]-v2[0],2) + math.pow(v1[1]-v2[1],2))
	return norm(v1 - v2)

def norm(v):
	return np.sqrt((v*v).sum(axis=0))

def file2Pix(file):
	im = Image.open(file).convert('L')
	pix = 1.0 - np.asarray(im) / 255.0

	return pix

def createRetina(file,nRings=None,hn=None):
	pix = file2Pix(file)

	size = min(pix.shape)
	template = Template(size,nRings,hn)

	return template.createRetina(pix)

def save(pix,fname):
	temp = (1.0 - pix) * 255
	temp = np.array([[int(round(value)) for value in row] for row in temp])
	im = Image.fromarray(temp.astype('uint8'))

	im.save(fname)

	return im

# returns angle made between the line connecting points 1 and 2 and the +x axis
def angle(point1, point2):
	diff = point2 - point1
	return math.atan2(diff[1],diff[0])

'''
def getRatio():
	return pi * (2*R0 + h0) / (nWedges - pi) / h0

def detRadiiAndHeights():
	R0 = h0 * nWedges / (2*pi)
	radii = [R0]
	heights = [h0]

	for i in range(1,nRings):
		h = pi * (2*radii[i-1] + heights[i-1]) / (nWedges - pi)
		radii.append(h * nWedges / (2*pi))
		heights.append(h)

	return (radii,heights)
'''
