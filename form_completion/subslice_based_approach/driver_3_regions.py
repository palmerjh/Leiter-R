import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import csv
import heapq as hq
import os
import sys
sys.path.append(r'C:\Users\palmerjh\Documents\AIVAS')
import cProfile
import time
from collections import deque

import math
pi = math.pi
tao = 2*pi
sqrt_2 = math.sqrt(2)

import similarity2 as sim
import taskpic as tp
#import pix2retina as p2r
sys.path.append(r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual')
import retina_newest as r

useTransparentPix = True

#square_size = 35
priority_queue_size = 1000   # size of pq for each choice in primary density search
secondary_queue_size = 5    # size of pq for each possible secondary location
tertiary_queue_size = 5		# size of pq for each possible tertiary location

search_size = 35            # size of squares to search for in choices
hn = 2

similarity_wts = [1,3,5]      # the weight of the similarity of each region (primary,secondary,etc.) to calculate overall simularity for composite results
box_colors = ['red','green','indigo']    # colors[0] for primary, colors[1] for secondary, etc. when drawing boxes

problems_dir = r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\problems'

# problem_list = ['09',
# 				'10_BW',
# 				'11_scaled',
# 				'12',
# 				'13',
# 				'14']

# problem_list = ['03_scaled',
# 				'04_scaled',
# 				'05_scaled',
# 				'06_scaled',
# 				'07_scaled',
# 				'08_scaled',
# 				'09',
# 				'10_BW',
# 				'11_scaled',
# 				'12',
# 				'13',
# 				'14']

problem_list = ['sample_multi_slot_hard']

# problem_list = ['09']

choices_dir = 'easel_slots'
targets_dir = 'choices'
results_dir = os.path.join('three_region_results', 'run_transparent')

#slice_index = 2

from collections import namedtuple
Result = namedtuple("Result","start wedge_index sim")   # simple container used to hold results for primary,secondary,etc. regions for each choice

template = r.Template(search_size,hn=hn)

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
	choices = [tp.TaskPic(os.path.join(cwd,choices_dir,c)) for c in os.listdir(os.path.join(cwd,choices_dir))]
	# for earlier problems (03 thru 08) where there are no easel choices
	#choices = [tp.TaskPic(os.path.join(cwd,'easel.jpg'))]
	#targets = [tp.TaskPic(os.getcwd() + '\\%s\\%s' % (targets_dir,t)) for t in os.listdir(targets_dir)]
	print('...DONE!\n')

	target_path = os.path.join(cwd, targets_dir)
	outfile_path = os.path.join(os.getcwd(), results_dir, prob)

	for target in os.listdir(target_path):
		# TODO identify slices automatically

		target_outfile_path = os.path.join(outfile_path, 'choice_%s' % target)
		target_infile_path = os.path.join(target_path, target)
		for c_slice in os.listdir(target_infile_path):
			slice_name = c_slice.split('.')[0]
			slice_path = os.path.join(target_outfile_path, slice_name)
			if not os.path.exists(slice_path):
				os.makedirs(slice_path)

			with open(os.path.join(slice_path,'console_readout.txt'), 'w') as f:
				print('Target: %s\nSlice: %s\n' % (target, slice_name))

				slicePic = tp.TaskPic(os.path.join(target_infile_path, c_slice))

				pix = slicePic.pix
				s = None

				# slice is too small for default search_size
				if search_size > min(pix.shape):
					print('Using smaller template for this slice...')
					s = r.Subslice(slicePic,prob,target,useTransparentPix=useTransparentPix,size=min(pix.shape)-1,hn=hn,nRegions=3)
				else:
					s = r.Subslice(slicePic,prob,target,useTransparentPix=useTransparentPix,template=template,nRegions=3)

				temp = s.template
				f.write('Using retina with following parameters:\n')
				f.write('\t-size: %d x %d\n' % (temp.size, temp.size))
				f.write('\t-number of rings: %d\n' % temp.nRings)
				f.write('\t-number of wedges: %d\n' % temp.nWedges)
				f.write('\t-blindspot_radius: %f\n\n' % temp.bs_radius)

				padded_choices = []
				# pads each choice with zeros just in case subslice is on edge
				for choice in choices:
					cx, cy = choice.size()
					start = s.size / 2
					padded_pix = np.zeros((cx + s.size, cy + s.size), dtype=choice.pix.dtype)
					padded_pix[start:start + cx, start:start + cy] = choice.pix

					new_choice = choice.copy()
					new_choice.pix = padded_pix
					new_choice.update()

					padded_choices.append(new_choice)

				print('Starting density search...')
				locations = sim_density_search(padded_choices,s.primary.ret,priority_queue_size,slice_path)
				'''
				locations = [[np.array([83,15])],
							 [np.array([1,15])],
							 [np.array([117,126])],
							 [np.array([129,100])]]
							 #[np.array([129,100])]]
				'''

				#print locations
				print('...DONE!\n')

				print('Starting rotational search...')
				s_flipped = s.flip()
				# sys.exit(0)
				primarySearch(padded_choices,s,s_flipped,locations)
				print('...DONE!\n')

				'''
				for slice in [s, s_flipped]:
					im = slice.taskpic.im
					name = slice.taskpic.fname.split('\\')[-1]
					boxes = []
					box_index = 0
					for start in [slice.primary.start,slice.secondary.start]:
						end = start + np.array([s.size-1,s.size-1])
						boxes.append(([tuple(start),tuple(end)],box_colors[box_index]))
						box_index += 1

					drawBoxes(im,boxes,slice_path,name)
				'''

				#sys.exit(1)

				sliceIm = slicePic.im
				boxes = []
				box_index = 0
				for start in [s.primary.start,s.secondary.start,s.tertiary.start]:
					end = start + np.array([s.size-1,s.size-1])
					boxes.append(([tuple(start),tuple(end)],box_colors[box_index]))
					box_index += 1

				drawBoxes(sliceIm,boxes,slice_path,'pst_regions.jpg')

				sliceImFlipped = s_flipped.taskpic.im
				boxes = []
				box_index = 0
				for start in [s_flipped.primary.start,s_flipped.secondary.start,s_flipped.tertiary.start]:
					end = start + np.array([s.size-1,s.size-1])
					boxes.append(([tuple(start),tuple(end)],box_colors[box_index]))
					box_index += 1

				drawBoxes(sliceIm,boxes,slice_path,'pst_regions_flipped.jpg')
				print('Writing results...')

				f.write('Regions identified in %s:...\n\n' % slice_name)

				for name, region in [('Primary', s.primary), \
									 ('Secondary', s.secondary), \
									 ('Tertiary', s.tertiary)]:

					f.write('%s region:\n' % name)
					f.write('\t-Location in target slice: ' + str(region.start) + '\n')
					f.write('\t-Variance (within retinal rings): ' + str(region.ret.getVar()) + '\n')
					#f.write('\t-Unrefined Density: ' + str(s.primary.ret.getUnrefinedDensity()) + '\n')
					f.write('\t-Density: ' + str(region.ret.getDensity()) + '\n')

				f.write('\nResults:...\n\n')

				writeResults(padded_choices,s,s_flipped,f,slice_path)
				print('...DONE!\n')

	end_t = time.time()

	print('\nTime elapsed: %f' % (end_t - start_t))


# searches through queue of possibilities, invoking a rotational search for each
# locations is a list of regions to investigate: 3-tuples of form (choice_index, sx, sy)
def primarySearch(choices,s,s_flipped,locations):
	nIterations = 0
	for c_locs in locations:
		nIterations += len(c_locs)

	counter = 0
	for c_index in range(len(choices)):
		s.results['primary'].append({'p':Result((0,0),0,0),
								  's':Result((0,0),0,0),
								  't':Result((0,0),0,0)})

		s.results['secondary'].append({'p':Result((0,0),0,0),
									's':Result((0,0),0,0),
  								  	't':Result((0,0),0,0)})

		s.results['tertiary'].append({'p':Result((0,0),0,0),
									's':Result((0,0),0,0),
  								  	't':Result((0,0),0,0)})

		s.results['composite'].append({'p':Result((0,0),0,0),
									's':Result((0,0),0,0),
  								  	't':Result((0,0),0,0),
									'comp_sim':0})

		s_flipped.results['primary'].append({'p':Result((0,0),0,0),
								  's':Result((0,0),0,0),
								  't':Result((0,0),0,0)})

		s_flipped.results['secondary'].append({'p':Result((0,0),0,0),
									's':Result((0,0),0,0),
  								  	't':Result((0,0),0,0)})

		s_flipped.results['tertiary'].append({'p':Result((0,0),0,0),
									's':Result((0,0),0,0),
  								  	't':Result((0,0),0,0)})

		s_flipped.results['composite'].append({'p':Result((0,0),0,0),
									's':Result((0,0),0,0),
  								  	't':Result((0,0),0,0),
									'comp_sim':0})

		for p_loc in locations[c_index]:
			sx, sy = p_loc

			# creates subspace specified by loc and transforms it to retina-form
			goal_ret = s.template.createRetina(choices[c_index].pix[sx:sx+s.size,sy:sy+s.size])

			p_sim, p_wedge = rotational_search(s.primary.ret,goal_ret)    # format: (most_sim, wedge_index)
			s_sim, s_loc = secondarySearch(choices[c_index].pix,s,np.array(p_loc),p_wedge)   # format: (most_sim,choice_loc)
			choice_sp_angle = r.angle(np.array(p_loc),np.array(s_loc))
			t_sim, t_loc = tertiarySearch(choices[c_index].pix,s,np.array(s_loc),choice_sp_angle)

			if p_sim > s.results['primary'][c_index]['p'].sim:
			   s.results['primary'][c_index] = {'p':Result(p_loc,p_wedge,p_sim),
												's':Result(s_loc,p_wedge,s_sim),
												't':Result(t_loc,p_wedge,t_sim)}

			if s_sim > s.results['secondary'][c_index]['s'].sim:
				s.results['secondary'][c_index] = {'p':Result(p_loc,p_wedge,p_sim),
												   's':Result(s_loc,p_wedge,s_sim),
												   't':Result(t_loc,p_wedge,t_sim)}

			if t_sim > s.results['tertiary'][c_index]['t'].sim:
				s.results['tertiary'][c_index] = {'p':Result(p_loc,p_wedge,p_sim),
												  's':Result(s_loc,p_wedge,s_sim),
												  't':Result(t_loc,p_wedge,t_sim)}

			comp_sim = np.average([p_sim,s_sim,t_sim],weights=similarity_wts)

			if comp_sim > s.results['composite'][c_index]['comp_sim']:
				s.results['composite'][c_index] = {'p':Result(p_loc,p_wedge,p_sim),
												   's':Result(s_loc,p_wedge,s_sim),
												   't':Result(t_loc,p_wedge,t_sim),
												   'comp_sim':comp_sim}

			p_sim, p_wedge = rotational_search(s_flipped.primary.ret,goal_ret)    # format: (most_sim, wedge_index)
			s_sim, s_loc = secondarySearch(choices[c_index].pix,s_flipped,np.array(p_loc),p_wedge)   # format: (most_sim,choice_loc)
			choice_sp_angle = r.angle(np.array(p_loc),np.array(s_loc))
			t_sim, t_loc = tertiarySearch(choices[c_index].pix,s_flipped,np.array(s_loc),choice_sp_angle)

			if p_sim > s_flipped.results['primary'][c_index]['p'].sim:
			   s_flipped.results['primary'][c_index] = {'p':Result(p_loc,p_wedge,p_sim),
														's':Result(s_loc,p_wedge,s_sim),
														't':Result(t_loc,p_wedge,t_sim)}

			if s_sim > s_flipped.results['secondary'][c_index]['s'].sim:
				s_flipped.results['secondary'][c_index] = {'p':Result(p_loc,p_wedge,p_sim),
												   		   's':Result(s_loc,p_wedge,s_sim),
												   		   't':Result(t_loc,p_wedge,t_sim)}

			if t_sim > s_flipped.results['tertiary'][c_index]['t'].sim:
				s_flipped.results['tertiary'][c_index] = {'p':Result(p_loc,p_wedge,p_sim),
												  		  's':Result(s_loc,p_wedge,s_sim),
												  		  't':Result(t_loc,p_wedge,t_sim)}

			comp_sim = np.average([p_sim,s_sim,t_sim],weights=similarity_wts)

			if comp_sim > s_flipped.results['composite'][c_index]['comp_sim']:
				s_flipped.results['composite'][c_index] = {'p':Result(p_loc,p_wedge,p_sim),
												   's':Result(s_loc,p_wedge,s_sim),
												   't':Result(t_loc,p_wedge,t_sim),
												   'comp_sim':comp_sim}

			if counter % 50 == 0:
				print(str(round(100*(float(counter)/nIterations),2)) + r'% done')

			counter += 1

def secondarySearch(choiceSpace,s,primary_loc,primary_wedge_index):
	min_angle = (primary_wedge_index - 0.5) / s.template.nWedges * tao
	max_angle = (primary_wedge_index + 0.5) / s.template.nWedges * tao

	secondary_locs = set()

	# print primary_loc
	# print primary_wedge_index
	# print min_angle
	# print max_angle
	# print('')
	# print s.sp_angle
	# print s.sp_dist

	min_pixel = closestPixel(primary_loc + rotate(s.sp_difference,min_angle))
	# print min_pixel

	# adds some "wiggle room"; equals angle of sector with arc length = half the diagonal of one pixel (i.e. 0.5*sqrt_2)
	min_relative = (min_angle + s.sp_angle) % (tao) - 0.5*sqrt_2 / s.sp_dist
	# print min_relative
	max_relative = (max_angle + s.sp_angle) % (tao) + 0.5*sqrt_2 / s.sp_dist
	# print max_relative
	# print('')

	q = deque()
	q.append(min_pixel)
	visited = []
	while len(q) > 0:
		s_loc = q.popleft()
		visited.append(s_loc)
		s_loc = np.array(s_loc)
		# print s_loc
		# pixel's center is within half the diagonal of one pixel (i.e. 0.5*sqrt_2) of arc
		isRadiallyClose = abs(r.dist(primary_loc, s_loc) - s.sp_dist) < 0.5*sqrt_2
		# print isRadiallyClose

		angle = r.angle(primary_loc, s_loc) % tao
		isAngularlyClose = min_relative < angle < max_relative
		# print angle
		# print isAngularlyClose

		if isRadiallyClose and isAngularlyClose:
			# print('trying to add %s' % str(tuple(s_loc)))
			secondary_locs.add(tuple(s_loc))
			# print secondary_locs
			for move in [[1,0], [0,1], [-1,0], [0,-1]]:
				new_loc = tuple(s_loc + move)
				if new_loc not in q and new_loc not in visited:
					q.append(new_loc)

		'''
		user_input = raw_input('enter...')
		if user_input == 'q':
			exit()
		'''

	secondary_locs = [loc for loc in secondary_locs if isValid(loc, choiceSpace.shape)]

	density_deltas = []
	for s_loc in secondary_locs:
		sx, sy = s_loc
		subSpace = choiceSpace[sx:sx+s.size,sy:sy+s.size]
		x,y = subSpace.shape
		if x == s.size and y == s.size:
			density_deltas.append((s.secondary.ret.ringwiseDensityDelta(subSpace),s_loc))

	if len(density_deltas) == 0:
		return (0,(0,0))

	density_deltas.sort()

	cutOff = min(secondary_queue_size,len(secondary_locs))
	secondary_locs = [tup[1] for tup in density_deltas[:cutOff]]

	target_ret = s.secondary.ret.copy()
	target_ret.rotate(primary_wedge_index)

	most_sim = (0,(0,0))
	for loc in secondary_locs:
		sx, sy = loc

		# creates subspace specified by loc and transforms it to retina-form
		goal_ret = s.template.createRetina(choiceSpace[sx:sx+s.size,sy:sy+s.size])

		mm = sim.maxMin(target_ret.retina,goal_ret.retina)

		if mm > most_sim[0]:
			most_sim = (mm,loc)

	return most_sim

def tertiarySearch(choiceSpace,s,secondary_loc,choice_sp_angle):
	rotation = choice_sp_angle - s.sp_angle
	centerPixel = closestPixel(secondary_loc + rotate(s.ts_difference,rotation))
	cx,cy = centerPixel
	target_ret = s.tertiary.ret.copy()

	most_sim = (0,(0,0))
	for x in range(cx-1,cx+2):
		for y in range(cy-1,cy+2):
			subSpace = choiceSpace[x:x+s.size,y:y+s.size]
			sx,sy = subSpace.shape
			if sx == s.size and sy == s.size:
				# creates subspace specified by loc and transforms it to retina-form
				goal_ret = s.template.createRetina(subSpace)

				mm = sim.maxMin(target_ret.retina,goal_ret.retina)

				if mm > most_sim[0]:
					most_sim = (mm,(x,y))

	return most_sim

def isValid(loc, size):
	return 0 <= loc[0] < size[0] and 0 <= loc[1] < size[1]

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

# searches through all the easel slots (choices) for areas of similar density to rot_ret.density
def sim_density_search(choices,rot_ret,queueSize,cur_results_dir):
	most_sim = []   # format: [(1.0 - density, (sx, sy))] for each choice

	size = rot_ret.template.size

	nIterations = 0
	for c in choices:
		cx,cy = c.size()
		nIterations += (cx-size+1) * (cy-size+1)

	print nIterations
	global_counter = 0
	for c_index, c in enumerate(choices):
		slice_easel_choice_path = os.path.join(cur_results_dir, 'slot%d' % c_index)
		if not os.path.exists(slice_easel_choice_path):
			os.makedirs(slice_easel_choice_path)

		all_sim_choice = []	# format: [(1.0 - density, (sx, sy))]
		most_sim.append([])
		#density_data.append([])
		cx,cy = c.size()
		print('Easel %d: %d total locations' % (c_index, (cx-size+1) * (cy-size+1)))
		#print('Slot %d:\nc.size(): %s\nc.im.size: %s\n' % (c_index, str(c.size()), str(c.im.size)))
		#assert(c.size() == c.im.size)
		counter = 0
		#print cx,cy
		for sx in range(cx-size+1):
			for sy in range(cy-size+1):
				density_delta = rot_ret.ringwiseDensityDelta(c.pix[sx:sx+size,sy:sy+size])
				res = 1.0 - density_delta

				all_sim_choice.append((res,(sx,sy)))

				if counter < queueSize:
					hq.heappush(most_sim[c_index],(res,(sx,sy)))
				else:
					hq.heappushpop(most_sim[c_index],(res,(sx,sy)))

				if global_counter % 50000 == 0:
					print(str(round(100*(float(global_counter)/nIterations),2)) + r'% done')

				#print(str(counter) + ': ' + str(most_sim['sp']))

				counter += 1
				global_counter += 1

		all_sim_choice.sort(reverse=True)
		most_sim[c_index] = sorted(most_sim[c_index],reverse=True)
		deltas, locs = tuple(zip(*most_sim[c_index]))

		# need to flip axes to make points
		point_locs = [(y,x) for (x,y) in locs]

		cutoff = 0.5 * (all_sim_choice[0][0] + all_sim_choice[-1][0])
		cutoff_index = queueSize - 1
		while all_sim_choice[cutoff_index][0] > cutoff and cutoff_index < counter:
			cutoff_index += 1

		cutoff_index += 1

		with open(os.path.join(slice_easel_choice_path, 'all_deltas.csv'), 'wb') as deltas_file:
			wr = csv.writer(deltas_file,delimiter=',')
			wr.writerows(all_sim_choice[:cutoff_index])

		# format: (cutoff, color)
		# cutoffs refer to fraction of distance between worst and best deltas in queue
		heatmap_input_alt = [	(0.0 , 'violet'),
							(0.5 , 'indigo'),
							(0.8 , 'blue'),
							(0.95 , 'green'),
							(0.97 , 'yellow'),
							(0.98 , 'orange'),
							(0.99 , 'red')	]

		heatmap_input = [	(0.0 , 'violet'),
							(0.5 , 'indigo'),
							(0.8 , 'blue'),
							(0.95 , 'green'),
							(0.97 , 'yellow'),
							(0.98 , 'orange'),
							(0.99 , 'red')	]

		cutoffs, colors = tuple(zip(*heatmap_input))

		heatmap_cutoffs = [co*deltas[0] + (1-co)*deltas[-1] for co in cutoffs]
		heatmap = {	cutoff	:	color for cutoff, color in zip(heatmap_cutoffs, colors)}

		points_im = Image.new('RGB',(cy,cx),color='white')
		pointer = ImageDraw.Draw(points_im)

		end_index = 0
		for cutoff in reversed(heatmap_cutoffs[1:]):
			start_index = end_index
			while deltas[end_index] > cutoff:
				end_index += 1

			pointer.point(point_locs[start_index:end_index],fill=heatmap[cutoff])

		final_color = heatmap[heatmap_cutoffs[0]]
		pointer.point(point_locs[end_index:],fill=final_color)

		slot_im = c.im.convert('RGB')
		overlay_im = Image.blend(points_im, slot_im, alpha=0.2)

		overlay_im.save(os.path.join(slice_easel_choice_path, 'sim_deltas.png'))

	return [[tup[1] for tup in choice] for choice in most_sim]


	#return [[tup[1] for tup in choice] for choice in most_sim]

def writeResults(choices,s,s_flipped,f,path):
	for result_type in ['composite',\
						'primary',\
						'secondary',\
						'tertiary']:

		f.write('\n%s Results:\n\n' % result_type.capitalize())
		writeSubResults(choices,s,s_flipped,result_type,f,path)

def writeSubResults(choices,s,s_flipped,result_type,f,path):
	size = s.size
	nWedges = s.template.nWedges

	best_choice = -1
	flipped = False
	if result_type == 'composite':
		best_choice_nonflipped = np.argmax([res['comp_sim'] for res in s.results[result_type]])
		best_choice_flipped = np.argmax([res['comp_sim'] for res in s_flipped.results[result_type]])
		flipped = s.results[result_type][best_choice_nonflipped]['comp_sim'] < \
					s_flipped.results[result_type][best_choice_flipped]['comp_sim']
		best_choice = best_choice_flipped if flipped else best_choice_nonflipped
	elif result_type == 'primary':
		best_choice_nonflipped = np.argmax([res['p'].sim for res in s.results[result_type]])
		best_choice_flipped = np.argmax([res['p'].sim for res in s_flipped.results[result_type]])
		flipped = s.results[result_type][best_choice_nonflipped]['p'].sim < \
					s_flipped.results[result_type][best_choice_flipped]['p'].sim
		best_choice = best_choice_flipped if flipped else best_choice_nonflipped
	elif result_type == 'secondary':
		best_choice_nonflipped = np.argmax([res['s'].sim for res in s.results[result_type]])
		best_choice_flipped = np.argmax([res['s'].sim for res in s_flipped.results[result_type]])
		flipped = s.results[result_type][best_choice_nonflipped]['s'].sim < \
					s_flipped.results[result_type][best_choice_flipped]['s'].sim
		best_choice = best_choice_flipped if flipped else best_choice_nonflipped
	elif result_type == 'tertiary':
		best_choice_nonflipped = np.argmax([res['t'].sim for res in s.results[result_type]])
		best_choice_flipped = np.argmax([res['t'].sim for res in s_flipped.results[result_type]])
		flipped = s.results[result_type][best_choice_nonflipped]['t'].sim < \
					s_flipped.results[result_type][best_choice_flipped]['t'].sim
		best_choice = best_choice_flipped if flipped else best_choice_nonflipped
	else:
		raise NameError('%s is not a valid result type' % result_type)

	flip_string = 'Flipped' if flipped else 'Not flipped'
	print('(%s) Best slot: %d' % (flip_string, best_choice))
	f.write('\t(%s) Best Slot: %d\n\n' % (flip_string, best_choice))
	for c_index, choice in enumerate(choices):
		directory = os.path.join(path,'slot%d'%c_index, result_type)
		if not os.path.exists(directory):
			os.makedirs(directory)

		res_nonflipped = s.results[result_type][c_index]
		res_flipped = s_flipped.results[result_type][c_index]

		flipped = False
		if result_type == 'composite':
			flipped = s.results[result_type][c_index]['comp_sim'] < \
						s_flipped.results[result_type][c_index]['comp_sim']
		elif result_type == 'primary':
			flipped = s.results[result_type][c_index]['p'].sim < \
						s_flipped.results[result_type][c_index]['p'].sim
		elif result_type == 'secondary':
			flipped = s.results[result_type][c_index]['s'].sim < \
						s_flipped.results[result_type][c_index]['s'].sim

		res = res_flipped if flipped else res_nonflipped
		#choice = choices[c_index]
		f.write('\tSlot %d:\n\n' % c_index)

		for name, index, region, region_flipped in [('primary','p',s.primary,s_flipped.primary),\
													('secondary','s',s.secondary,s_flipped.secondary),\
													('tertiary','t',s.tertiary,s_flipped.tertiary)]:

			subres = res[index]
			f.write('\t\t%s Region:\n' % name.capitalize())
			f.write('\t\t\t-Flipped? %r\n' % flipped)
			f.write('\t\t\t-Location: %s\n' % str(subres.start))
			f.write('\t\t\t-Rotation: wedge %d out of %d, which corresponds to the\n\t\t\tregion being rotated %.2f degrees anti-clockwise\n'
				% (subres.wedge_index,nWedges,round(float(subres.wedge_index)/nWedges * 360, 2)))
			f.write('\t\t\t-Similarity: %f\n\n' % subres.sim)
			region_ret = region_flipped.ret.copy() if flipped else region.ret.copy()
			region_ret.rotate(subres.wedge_index)
			region_ret.visualize(os.path.join(directory,'%s_target.jpg' % name))
			px, py = subres.start
			subspace = choices[c_index].pix[px:px+s.size,py:py+s.size]
			s.template.createRetina(subspace).visualize(os.path.join(directory,'%s_choice.jpg' % name))

		if result_type == 'composite':
			f.write('\n\t\tComposite Similarity: %f\n' % res['comp_sim'])

		f.write('\n')

		choiceIm = choice.im
		boxes = []
		box_index = 0
		for start in [res['p'].start,res['s'].start,res['t'].start]:
			end = start + np.array([size-1,size-1])
			boxes.append(([tuple(start),tuple(end)],box_colors[box_index]))
			box_index += 1

		drawBoxes(choiceIm,boxes,directory,'results.jpg')

def debug(pix,fname):
	with open(fname + '.csv', 'wb') as file:
		wr = csv.writer(file,delimiter=',')
		wr.writerows(pix)

# creates copy of image im and overlays box(es) and then saves it using fname
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

	# print directory
	# print fname

	base.save(os.path.join(directory,fname))

if __name__ == "__main__":
	results_path = os.path.join(os.getcwd(),results_dir)
	if not os.path.exists(results_path):
		os.makedirs(results_path)
	cProfile.run('main()', os.path.join(os.getcwd(),results_dir,'stats'))
	#main()
