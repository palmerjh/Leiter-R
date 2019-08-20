import numpy as np
from PIL import ImageDraw
import matplotlib.pyplot as plt
import csv
import heapq as hq
import os
import sys
sys.path.append(r'C:\Users\palmerjh\Documents\AIVAS')
import cProfile
import time

import math
pi = math.pi

import similarity2 as sim
import taskpic as tp
#import pix2retina as p2r
sys.path.append(r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual')
import retina_subpixel as r

#square_size = 35
priority_queue_size = 1400   # size of pq for each choice in primary density search
secondary_queue_size = 5    # size of pq for each possible primary location

search_size = 35            # size of squares to search for in choices

similarity_wts = [1,5]      # the weight of the similarity of each region (primary,secondary,etc.) to calculate overall simularity for composite results
box_colors = ['red','green']    # colors[0] for primary, colors[1] for secondary, etc. when drawing boxes

problems_dir = r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\problems'
problem_list = ['03',
				'04',
				'05',
				'06',
				'07',
				'08']
				#'15']

choices_dir = 'easel_slots'
targets_dir = 'choices'
results_dir = os.path.join('results', 'run3D_early_problems')

#slice_index = 2

from collections import namedtuple
Result = namedtuple("Result","start wedge_index sim")   # simple container used to hold results for primary,secondary,etc. regions for each choice

template = r.Template(search_size)

def main():
	for prob in problem_list:
		cwd = os.path.join(problems_dir, prob)
		if os.path.isdir(cwd):
			print('Solving problem %s....\n' % prob)
			solve_problem(prob)
		else:
			print('Problem %s doesn\'t exist' % prob)

def solve_problem(prob):
	start_t = time.time()
	cwd = os.path.join(problems_dir, prob)

	print('Initializing choices...')
	# for later problems (09 thru 15) where there are easel choices
	#choices = [tp.TaskPic(os.path.join(cwd,choices_dir,c)) for c in os.listdir(os.path.join(cwd,choices_dir))]
	# for earlier problems (03 thru 08) where there are no easel choices
	choices = [tp.TaskPic(os.path.join(cwd,'easel.jpg'))]
	#targets = [tp.TaskPic(os.getcwd() + '\\%s\\%s' % (targets_dir,t)) for t in os.listdir(targets_dir)]
	print('...DONE!\n')

	target_path = os.path.join(cwd, targets_dir)
	outfile_path = os.path.join(os.getcwd(), results_dir, prob)

	for target in os.listdir(target_path):
		# TODO identify slices automatically

		target_outfile_path = os.path.join(outfile_path, target)
		if not os.path.exists(target_outfile_path):
			os.makedirs(target_outfile_path)

		with open(os.path.join(target_outfile_path,'console_readout.txt'), 'w') as f:
			print(target + ':\n')

			path = os.path.join(target_path, target)

			print('Initializing slices...')
			slicePics = [tp.TaskPic(os.path.join(path, s)) for s in os.listdir(path)]
			slice_index = np.random.randint(0,len(slicePics))   #********** TODO this is arbitrary; could use all of them

			pix = slicePics[slice_index].pix
			s = None

			print('\n\t....Slice %d chosen....\n' % slice_index)

			# slice is too small for default search_size
			if search_size > min(pix.shape):
				print('Using smaller template for this slice...')
				s = r.TargetSlice(pix,prob,target,size=min(pix.shape))
			else:
				s = r.TargetSlice(pix,prob,target,template=template)

			print('...DONE!\n')

			temp = s.template
			f.write('Using retina with following parameters:\n')
			f.write('\t-size: %d x %d\n' % (temp.size, temp.size))
			f.write('\t-number of rings: %d\n' % temp.nRings)
			f.write('\t-number of wedges: %d\n' % temp.nWedges)
			f.write('\t-blindspot_radius: %f\n\n' % temp.bs_radius)

			print('Starting density search...')
			locations = sim_density_search(choices,s.primary.ret,priority_queue_size)
			'''
			locations = []
			for i in range(len(choices)):
				locs = []
				for n in range(20):
					locs.append((i,n))
				locations.append(locs)
			'''
			#print locations
			print('...DONE!\n')

			print('Starting rotational search...')
			primarySearch(choices,s,locations)
			print('...DONE!\n')

			sliceIm = slicePics[slice_index].im
			boxes = []
			box_index = 0
			for start in [s.primary.start,s.secondary.start]:
				end = start + np.array([s.size-1,s.size-1])
				boxes.append(([tuple(start),tuple(end)],box_colors[box_index]))
				box_index += 1

			drawBoxes(sliceIm,boxes,target_outfile_path,'slice_%d_regions.jpg' % slice_index)
			print('Writing results...')

			f.write('Regions identified in target slice%d:...\n\n' % slice_index)

			f.write('Primary region:\n')
			f.write('\t-Location in target slice: ' + str(s.primary.start) + '\n')
			f.write('\t-Variance (within retinal rings): ' + str(s.primary.ret.getVar()) + '\n')
			#f.write('\t-Unrefined Density: ' + str(s.primary.ret.getUnrefinedDensity()) + '\n')
			f.write('\t-Density: ' + str(s.primary.ret.getDensity()) + '\n')

			f.write('\nSecondary region:\n')
			f.write('\t-Location in target slice: ' + str(s.secondary.start) + '\n')
			f.write('\t-Variance (within retinal rings): ' + str(s.secondary.ret.getVar()) + '\n')
			#f.write('\t-Unrefined Density: ' + str(s.secondary.ret.getUnrefinedDensity()) + '\n')
			f.write('\t-Density: ' + str(s.secondary.ret.getDensity()) + '\n')

			f.write('\nResults:...\n\n')

			writeResults(choices,s,f)
			print('...DONE!\n')

	end_t = time.time()

	print('\nTime elapsed: %f' % (end_t - start_t))


# searches through queue of possibilities, invoking a rotational search for each
# locations is a list of regions to investigate: 3-tuples of form (choice_index, sx, sy)
def primarySearch(choices,s,locations):
	nIterations = 0
	for c_locs in locations:
		nIterations += len(c_locs)

	counter = 0
	for c_index in range(len(choices)):
		s.results['primary'].append({'p':Result((0,0),0,0),
								  's':Result((0,0),0,0)})

		s.results['secondary'].append({'p':Result((0,0),0,0),
									's':Result((0,0),0,0)})

		s.results['composite'].append({'p':Result((0,0),0,0),
									's':Result((0,0),0,0),
									'comp_sim':0})
		for p_loc in locations[c_index]:
			sx, sy = p_loc

			# creates subspace specified by loc and transforms it to retina-form
			goal_ret = s.template.createRetina(choices[c_index].pix[sx:sx+s.size,sy:sy+s.size])

			p_sim, p_wedge = rotational_search(s.primary.ret,goal_ret)    # format: (most_sim, wedge_index)
			s_sim, s_loc = secondarySearch(choices[c_index].pix,s,np.array(p_loc),p_wedge)   # format: (most_sim,choice_loc)

			'''
			ss = secondarySearch(choices[c_index].pix,s,np.array(p_loc),p_wedge)   # format: (most_sim,choice_loc)
			if not ss[0] == 0:
				print(p_loc)
				print(ss[1])
				print(p_wedge)
				print(p_sim)
				print(ss[0])
			s_sim, s_loc = ss
			'''

			if p_sim > s.results['primary'][c_index]['p'].sim:
			   s.results['primary'][c_index] = {'p':Result(p_loc,p_wedge,p_sim),
												's':Result(s_loc,p_wedge,s_sim)}

			if s_sim >s.results['secondary'][c_index]['s'].sim:
				s.results['secondary'][c_index] = {'p':Result(p_loc,p_wedge,p_sim),
												   's':Result(s_loc,p_wedge,s_sim)}

			comp_sim = np.average([p_sim,s_sim],weights=similarity_wts)

			if comp_sim > s.results['composite'][c_index]['comp_sim']:
				s.results['composite'][c_index] = {'p':Result(p_loc,p_wedge,p_sim),
												   's':Result(s_loc,p_wedge,s_sim),
												   'comp_sim':comp_sim}

			if counter % 50 == 0:
				print(str(round(100*(float(counter)/nIterations),2)) + r'% done')

			counter += 1

def secondarySearch(choiceSpace,s,primary_loc,primary_wedge_index):
	min_angle = (primary_wedge_index - 0.5) / s.template.nWedges * 2*pi
	max_angle = (primary_wedge_index + 0.5) / s.template.nWedges * 2*pi

	delta_angle = 2*pi * s.sp_dist / (2*s.template.nWedges)     # delta_angle is small enough to ensure all the closest pixels to the circumference are visited
	angle = min_angle
	choiceSpace_locs = []
	while angle < max_angle:
		pixel = closestPixel(primary_loc + rotate(s.sp_difference,angle))
		if pixel not in choiceSpace_locs and pixel[0] >= 0 and pixel[1] >= 0:
			choiceSpace_locs.append(pixel)

		angle += delta_angle

	density_deltas = []
	for cs_loc in choiceSpace_locs:
		sx, sy = cs_loc
		subSpace = choiceSpace[sx:sx+s.size,sy:sy+s.size]
		x,y = subSpace.shape
		if x == s.size and y == s.size:
			density_deltas.append((s.secondary.ret.ringwiseDensityDelta(subSpace),cs_loc))

	if len(density_deltas) == 0:
		return (0,(0,0))

	density_deltas.sort()

	cutOff = min(secondary_queue_size,len(choiceSpace_locs))
	choiceSpace_locs = [tup[1] for tup in density_deltas[:cutOff]]

	target_ret = s.secondary.ret.copy()
	target_ret.rotate(primary_wedge_index)

	most_sim = (0,(0,0))
	for loc in choiceSpace_locs:
		sx, sy = loc

		# creates subspace specified by loc and transforms it to retina-form
		goal_ret = s.template.createRetina(choiceSpace[sx:sx+s.size,sy:sy+s.size])

		mm = sim.maxMin(target_ret.retina,goal_ret.retina)

		if mm > most_sim[0]:
			most_sim = (mm,loc)

	return most_sim

def closestPixel(loc):
	return (int(round(loc[0])),int(round(loc[1])))

def rotate(v,angle):
	cos = math.cos(angle)
	sin = math.sin(angle)
	return np.array([cos*v[0] - sin*v[1], sin*v[0] + cos*v[1]])

def rotational_search(rot_ret,goal_ret):
	most_sim = (0,0) # format: (most_sim, wedge_index)

	for wedge_index in range(rot_ret.template.nWedges):

		mm = sim.maxMin(rot_ret.retina,goal_ret.retina)
		if mm > most_sim[0]:
			most_sim = (mm,wedge_index)

		rot_ret.rotate()

	return most_sim

# searches through all the choices for areas of similar density to rot_ret.density
def sim_density_search(choices,rot_ret,queueSize):
	most_sim = []   # format: [(1.0 - density, (sx, sy))] for each choice

	size = rot_ret.template.size

	'''# BEGIN DEBUG STUFF ------------------------------------------------

	density_delta_bin_mins =   [0.0,        # mins for bins that categorize density deltas
								0.6,
								0.7,
								0.75,
								0.8,
								0.82,
								0.84,
								0.85,
								0.86,
								0.87,
								0.88,
								0.89,
								0.9,
								0.92,
								0.94,
								0.95,
								0.96,
								0.965,
								0.970,
								0.975,
								0.98,
								0.985,
								0.99]

	nBins = len(density_delta_bin_mins)

	density_delta_bin_counts = np.zeros((len(choices),nBins),dtype='uint32') # initializes np array (size=nBins) with zeros for each choice

	x_range = range(480,485)
	y_range = range(45,50)
	choice_real = 4

	target0_real = []
	for x in x_range:
		for y in y_range:
			target0_real.append((x,y))

	nIterations_choice = []
	for c in choices:
		cx,cy = c.size()
		nIterations_choice.append((cx-size+1) * (cy-size+1))

	real_deltas = []

	best_deltas = []
	for _ in range(len(choices)):
		best_deltas.append((-1,(0,0)))

	average = 0.0

	'''# END DEBUG STUFF ---------------------------------------------------

	nIterations = 0
	for c in choices:
		cx,cy = c.size()
		nIterations += (cx-size+1) * (cy-size+1)

	print nIterations
	global_counter = 0
	c_index = 0
	for c in choices:
		most_sim.append([])
		#density_data.append([])
		cx,cy = c.size()
		counter = 0
		#print cx,cy
		for sx in range(cx-size+1):
			for sy in range(cy-size+1):
				density_delta = rot_ret.ringwiseDensityDelta(c.pix[sx:sx+size,sy:sy+size])
				res = 1.0 - density_delta

				'''# BEGIN DEBUG STUFF -----------------------------------

				bin_index = 0
				for bin_index in range(1,nBins):
					if res < density_delta_bin_mins[bin_index]:
						bin_index -= 1
						break

				density_delta_bin_counts[c_index][bin_index] += 1

				if c_index == choice_real and sx in x_range and sy in y_range:
					real_deltas.append((res,(sx,sy)))

				if res > best_deltas[c_index][0]:
					best_deltas[c_index] = (res,(sx,sy))

				average += res

				'''# END DEBUG STUFF -----------------------------------

				if counter < queueSize:
					hq.heappush(most_sim[c_index],(res,(sx,sy)))
				else:
					hq.heappushpop(most_sim[c_index],(res,(sx,sy)))

				if global_counter % 50000 == 0:
					print(str(round(100*(float(global_counter)/nIterations),2)) + r'% done')

				#print(str(counter) + ': ' + str(most_sim['sp']))

				counter += 1
				global_counter += 1

		c_index += 1

	'''
	for i in range(len(choices)):
		most_sim[i].sort(reverse=True)
		print most_sim[i][-1]

		plt.figure() # <- makes a new figure and sets it active (add this)
		plt.hist(density_data[i],bins=10000) # <- finds the current active axes/figure and plots to it
		plt.title('Density Similarity Distribution')
		plt.xlabel('1.0 - Absolute Difference')
		plt.ylabel('Count')
		#plt.axis([0, 1, 0, ymax])
		# plt.figure() # <- makes new figure and makes it active (remove this)
		plt.savefig('density_distrib_%d.svg' % i) # <- saves the currently active figure (which is empty in your code)
	'''

	result = [[tup[1] for tup in choice] for choice in most_sim]

	'''# BEGIN DEBUG STUFF ------------------------------------------------

	with open('debug_target1.txt', 'w') as f_debug:
		global_counts = np.zeros((nBins,),dtype='uint32')
		normalized_counts = []
		for i in range(len(choices)):
			for bin_index in range(nBins):
				global_counts[bin_index] += density_delta_bin_counts[i][bin_index]
			normalized_counts.append(density_delta_bin_counts[i] * (1 / float(nIterations_choice[i])))
		normalized_global_counts = global_counts * (1 / float(nIterations))

		f_debug.write('Overall delta distribution:\n\n')
		for bin_index in range(nBins):
			f_debug.write('%f:\t%d\t%f\n' % (density_delta_bin_mins[bin_index],global_counts[bin_index],normalized_global_counts[bin_index]))

		f_debug.write('Choice delta distribution decomposition:\n')
		for i in range(len(choices)):
			f_debug.write('\nChoice %d:\n\n' % i)
			for bin_index in range(nBins):
				f_debug.write('%f:\t%d\t%f\n' % (density_delta_bin_mins[bin_index],density_delta_bin_counts[i][bin_index],normalized_counts[i][bin_index]))

		f_debug.write('\nBest deltas by choice:\n\n')
		for i in range(len(choices)):
			f_debug.write('choice %d: %f\tat %s\n' % (i,best_deltas[i][0],str(best_deltas[i][1])))

		f_debug.write('\nAverage delta: %f' % (average / nIterations))

		real_deltas.sort(reverse=True)
		f_debug.write('\nDeltas for real locations (in choice %d):\n\n' % choice_real)
		for data in real_deltas:
			delta,pixel = data
			f_debug.write('%f\tat %s\n' % (delta,str(pixel)))

		most_sim_choice = most_sim[choice_real]
		most_sim_choice.sort(reverse=True)
		most_sim_choice_locs = [tup[1] for tup in most_sim_choice]
		f_debug.write('\nReal location(s) in choice %d queue:\n\n' % choice_real)
		for real in target0_real:
			try:
				i = most_sim_choice_locs.index(real)
			except ValueError:
				i = -1
			if not i == -1:
				f_debug.write('%d: %f\tat%s\n' % (i,most_sim_choice[i][0],str(real)))

	'''# END DEBUG STUFF -----------------------------------------------------

	return result

	#return [[tup[1] for tup in choice] for choice in most_sim]

def writeResults(choices,s,f):
	f.write('\nComposite Results:\n\n')
	writeSubResults(choices,s,'composite',f)

	f.write('\nPrimary Results:\n\n')
	writeSubResults(choices,s,'primary',f)

	f.write('\nSecondary Results:\n\n')
	writeSubResults(choices,s,'secondary',f)

def writeSubResults(choices,s,result_type,f):
	size = s.size
	nWedges = s.template.nWedges

	best_choice = -1
	if result_type == 'composite':
		best_choice = np.argmax([res['comp_sim'] for res in s.results[result_type]])
	elif result_type == 'primary':
		best_choice = np.argmax([res['p'].sim for res in s.results[result_type]])
	elif result_type == 'secondary':
		best_choice = np.argmax([res['s'].sim for res in s.results[result_type]])
	else:
		raise NameError('%s is not a valid result type' % result_type)

	print('Best slot (for %s): %d' % (result_type, best_choice))
	f.write('\tBest Slot: %d\n\n' % best_choice)
	for c_index, choice in enumerate(choices):
		directory = os.path.join(os.getcwd(), results_dir, s.prob_name, s.t_name, 'slot%d'%c_index, result_type)
		if not os.path.exists(directory):
			os.makedirs(directory)

		res = s.results[result_type][c_index]
		#choice = choices[c_index]
		f.write('\tSlot %d:\n\n' % c_index)

		f.write('\t\tPrimary Region:\n')
		f.write('\t\t\t-Location: %s\n' % str(res['p'].start))
		f.write('\t\t\t-Rotation: wedge %d out of %d, which corresponds to the\n\t\t\tprimary region being rotated %.2f degrees anti-clockwise\n'
			% (res['p'].wedge_index,nWedges,round(float(res['p'].wedge_index)/nWedges * 360, 2)))
		f.write('\t\t\t-Similarity: %f\n\n' % res['p'].sim)
		p_ret = s.primary.ret.copy()
		p_ret.rotate(res['p'].wedge_index)
		p_ret.visualize(os.path.join(directory,'primary_target.jpg'))
		px, py = res['p'].start
		p_subspace = choices[c_index].pix[px:px+s.size,py:py+s.size]
		s.template.createRetina(p_subspace).visualize(os.path.join(directory,'primary_choice.jpg'))

		f.write('\t\tSecondary Region:\n')
		f.write('\t\t\t-Location: %s\n' % str(res['s'].start))
		f.write('\t\t\t-Rotation: wedge %d out of %d, which corresponds to the\n\t\t\tsecondary region being rotated %.2f degrees anti-clockwise\n'
			% (res['s'].wedge_index,nWedges,round(float(res['s'].wedge_index)/nWedges * 360, 2)))
		f.write('\t\t\t-Similarity: %f\n' % res['s'].sim)
		s_ret = s.secondary.ret.copy()
		s_ret.rotate(res['s'].wedge_index)
		s_ret.visualize(os.path.join(directory,'secondary_target.jpg'))
		sx, sy = res['s'].start
		s_subspace = choices[c_index].pix[sx:sx+s.size,sy:sy+s.size]
		s.template.createRetina(s_subspace).visualize(os.path.join(directory,'secondary_choice.jpg'))

		if result_type == 'composite':
			f.write('\n\t\tComposite Similarity: %f\n' % res['comp_sim'])

		f.write('\n')

		choiceIm = choice.im
		boxes = []
		box_index = 0
		for start in [res['p'].start,res['s'].start]:
			end = start + np.array([size-1,size-1])
			boxes.append(([tuple(start),tuple(end)],box_colors[box_index]))
			box_index += 1

		drawBoxes(choiceIm,boxes,directory,'results.jpg')

def debug(pix,fname):
	with open(fname + '.csv', 'wb') as file:
		wr = csv.writer(file,delimiter=',')
		wr.writerows(pix)

# creates copy of imag im and overlays box(es) and then saves it using fname
# boxes format: ([(xo,y0),(x1,y1)],color)
def drawBoxes(im,boxes,directory,fname):
	base = im.copy().convert('RGB')
	boxer = ImageDraw.Draw(base)

	for box in boxes:
		xy, color = box
		xy = [xy[0][1],xy[0][0],xy[1][1],xy[1][0]]
		boxer.rectangle(xy,outline=color)

	if not os.path.exists(directory):
		os.makedirs(directory)

	base.save(os.path.join(directory,fname))

'''
# needed in case of equal first column values in heapq
class FirstList(list):
	def __lt__(self, other):
		return self[0] < other[0]
'''

if __name__ == "__main__":
	results_path = os.path.join(os.getcwd(),results_dir)
	if not os.path.exists(results_path):
		os.makedirs(results_path)
	cProfile.run('main()', os.path.join(os.getcwd(),results_dir,'stats'))
	#main()

# using all of the similarity heuristics:

'''
def rotational_search(rotated,goal):

	most_sim = {'sp':[0,0],   # format: [most_sim, wedge_index]
				'mm':[0,0],
				'alt':[0,0]}

	nWedges, nRings = rotated.shape

	for wedge_index in range(nWedges):

		sp = sim.sumProd(rotated.pix,original.pix)
		mm = sim.maxMin(rotated.pix,original.pix)
		a = sim.alt(rotated.pix,original.pix)

		#print('%d\tsp: %f\tmm: %f\ta: %f' % (i,sp,mm,a))

		if sp > most_sim['sp'][0]:
			most_sim['sp'] = [sp,i]
		if mm > most_sim['mm'][0]:
			most_sim['mm'] = [mm,i]
		if a > most_sim['alt'][0]:
			most_sim['alt'] = [a,i]

		rotated = np.roll(rotated, 1, axis=0)

	return most_sim
'''

'''
	for c in choices:
		most_sim.append(choice_density(c,rot_ret,queueSize))

	return [[tup[1] for tup in choice] for choice in most_sim]

def choice_density(c,rot_ret,queueSize):
	size = rot_ret.template.size
	cx,cy = c.size()
	#print cx,cy
	choice_most_sim = []
	for sx in range(cx-size+1):
		for sy in range(cy-size+1):
			density = rot_ret.template.calcDensity(c.pix[sx:sx+size,sy:sy+size])
			delta = abs(density - rot_ret.density)

			if counter < queueSize:
				hq.heappush(choice_most_sim,(1.0 - delta,(sx,sy)))
			else:
				hq.heappushpop(choice_most_sim,(1.0 - delta,(sx,sy)))

				#print(str(counter) + ': ' + str(most_sim['sp']))

	return choice_most_sim
'''
