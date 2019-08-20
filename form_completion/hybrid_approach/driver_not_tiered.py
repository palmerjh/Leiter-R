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
import similarity as sim_old
import taskpic as tp
#import pix2retina as p2r
sys.path.append(r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual')
import retina_newest as r

tiered = True
resize_factor = 3
# for initial coarse density delta search
sub_hn_coarse = 1

#square_size = 35
priority_queue_size = 50   # size of pq for each choice in subslice density search
useTransparentPix = True

subsize = 34            # size of squares to search for in choices
sub_hn = 1

similarity_wts = [1,5]      # the weight of the similarity of each region (subslice,slice) to calculate overall simularity for composite results
box_colors = ['red','green']    # colors[0] for subslice, colors[1] for slice when drawing boxes

problems_dir = r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\problems'

'''
problem_list = ['03',
				'04',
				'05',
				'06',
				'07',
				'08',
				'09',
				'10',
				'11',
				'12',
				'13',
				'14']
'''

# problem_list = ['03_scaled',
# 				'04_scaled',
# 				'05_scaled',
# 				'06_scaled',
# 				'07_scaled',
# 				'08_scaled']

# problem_list = ['07_scaled',
# 				'08_scaled',
# 				'09',
# 				'10_BW',
# 				'11_scaled',
# 				'12',
# 				'14']

# problem_list = ['03_scaled',
# 				'04_scaled',
# 				'05_scaled',
# 				'06_scaled',
# 				'13']

problem_list = ['sample_multi_slot_hard']

choices_dir = 'easel_slots'
targets_dir = 'choices'
results_dir = os.path.join('results', 'run_transparent_sample_rf=3_1_5')

#slice_index = 2

from collections import namedtuple
Result = namedtuple("Result","start wedge_index sim")   # simple container used to hold results for primary,secondary,etc. regions for each choice

subtemplate = r.Template(subsize,hn=sub_hn)
subtemplate_small = r.Template(subsize / resize_factor,hn=sub_hn_coarse)

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

				sx, sy = slicePic.size()
				max_dim = max(sx,sy)

				start_x = int((max_dim - sx)) / 2
				start_y = int((max_dim - sy)) / 2

				# creates a square matrix around pix, padding with zeros
				# adds 1 to make sure getTransparent() will reach all possible pixels
				padded = np.zeros((max_dim+1,max_dim+1), dtype=pix.dtype)
				padded[start_x:start_x + sx, start_y:start_y + sy] = pix
				slicePic.pix = np.copy(padded)
				slicePic.update()
				pix = slicePic.pix

				# resizes image for faster density_delta search
				slicePic_small = slicePic.resize(resize_factor)
				pix_small = slicePic_small.pix

				padded_choices = []
				padded_choices_small = []
				# pads each choice with zeros just in case slice is on edge
				for choice in choices:
					cx, cy = choice.size()
					start = max_dim / 2
					padded_pix = np.zeros((cx + max_dim, cy + max_dim), dtype=choice.pix.dtype)
					padded_pix[start:start + cx, start:start + cy] = choice.pix

					new_choice = choice.copy()
					new_choice.pix = padded_pix
					new_choice.update()

					# resizes image for faster density_delta search
					new_choice_small = new_choice.resize(resize_factor)

					padded_choices.append(new_choice)
					padded_choices_small.append(new_choice_small)

				# empirically this has been found to be a good tradeoff of speed vs.
				# efficiency as the retina gets bigger
				# hn = int(round(max_dim+1 / 25.0))
				hn = max(int(round((max_dim+1) / 50.0)), 1)
				print('Creating slice template...')
				template = r.Template(max_dim+1, hn=hn)
				print('....DONE!!!\n')

				small_x, small_y = slicePic_small.size()
				assert small_x == small_y

				hn_small = max(int(round(small_x / 50.0)), 1)
				print('Creating small slice template...')
				template_small = r.Template(small_x, hn=hn_small)
				print('....DONE!!!\n')

				# slice is too small for default subsize
				subsize_actual = min(subsize, min(pix.shape)-1)
				if subsize > min(pix.shape):
					print('Using smaller subtemplate for this slice...')
					s = r.HybridSlice(slicePic,prob,target,useTransparentPix=useTransparentPix,template=template,subsize=subsize_actual,sub_hn=sub_hn)
				else:
					s = r.HybridSlice(slicePic,prob,target,useTransparentPix=useTransparentPix,template=template,subtemplate=subtemplate)

				s_flipped = s.flip()

				s_small = None
				subsize_small_actual = min(subsize / resize_factor, min(pix_small.shape)-1)
				if subsize / resize_factor > min(pix_small.shape):
					print('Using smaller subtemplate for small version of slice...')
					s_small = r.HybridSlice(slicePic_small,prob,target,useTransparentPix=useTransparentPix,\
								template=template_small,subsize=subsize_small_actual,sub_hn=sub_hn_coarse)
				else:
					s_small = r.HybridSlice(slicePic_small,prob,target,useTransparentPix=useTransparentPix,\
								template=template_small,subtemplate=subtemplate_small)

				s_small_flipped = s_small.flip()

				subtemp = s.subtemplate
				f.write('Using subretina with following parameters:\n')
				f.write('\t-size: %d x %d\n' % (subtemp.size, subtemp.size))
				f.write('\t-number of rings: %d\n' % subtemp.nRings)
				f.write('\t-number of wedges: %d\n' % subtemp.nWedges)
				f.write('\t-blindspot_radius: %f\n\n' % subtemp.bs_radius)

				subtemp_small = s_small.subtemplate
				f.write('Using small subretina with following parameters:\n')
				f.write('\t-size: %d x %d\n' % (subtemp_small.size, subtemp_small.size))
				f.write('\t-number of rings: %d\n' % subtemp_small.nRings)
				f.write('\t-number of wedges: %d\n' % subtemp_small.nWedges)
				f.write('\t-blindspot_radius: %f\n\n' % subtemp_small.bs_radius)

				temp = s.template
				f.write('Using retina with following parameters:\n')
				f.write('\t-size: %d x %d\n' % (temp.size, temp.size))
				f.write('\t-number of rings: %d\n' % temp.nRings)
				f.write('\t-number of wedges: %d\n' % temp.nWedges)
				f.write('\t-blindspot_radius: %f\n\n' % temp.bs_radius)

				sliceIm = slicePic.im
				boxes = []
				box_index = 0
				for start in [s.subslice.start]:
					end = start + np.array([s.subsize-1,s.subsize-1])
					boxes.append(([tuple(start),tuple(end)],box_colors[box_index]))
					box_index += 1

				drawBoxes(sliceIm,boxes,slice_path,'subslice.jpg')

				# sliceImFlipped = s_flipped.taskpic.im
				# boxes = []
				# box_index = 0
				# for start in [s_flipped.subslice.start]:
				# 	end = start + np.array([s.subsize-1,s.subsize-1])
				# 	boxes.append(([tuple(start),tuple(end)],box_colors[box_index]))
				# 	box_index += 1
				#
				# drawBoxes(sliceImFlipped,boxes,slice_path,'subslice_flipped.jpg')

				s.subslice.ret.visualize(os.path.join(slice_path,'subslice_retina.jpg'))
				# s_flipped.subslice.ret.visualize().show()

				sliceIm = slicePic_small.im
				boxes = []
				box_index = 0
				for start in [s_small.subslice.start]:
					end = start + np.array([s_small.subsize-1,s_small.subsize-1])
					boxes.append(([tuple(start),tuple(end)],box_colors[box_index]))
					box_index += 1

				drawBoxes(sliceIm,boxes,slice_path,'subslice_small.jpg')

				# sliceImFlipped = s_small_flipped.taskpic.im
				# boxes = []
				# box_index = 0
				# for start in [s_small_flipped.subslice.start]:
				# 	end = start + np.array([s_small_flipped.subsize-1,s_small_flipped.subsize-1])
				# 	boxes.append(([tuple(start),tuple(end)],box_colors[box_index]))
				# 	box_index += 1
				#
				# drawBoxes(sliceImFlipped,boxes,slice_path,'subslice_small_flipped.jpg')

				s_small.subslice.ret.visualize(os.path.join(slice_path,'subslice_small_retina.jpg'))
				# s_small_flipped.subslice.ret.visualize().show()

				# continue

				print('Starting density search...')
				locations = []
				if tiered:
					locations = sim_density_search(padded_choices_small,s_small.subslice.ret,priority_queue_size,slice_path)
				else:
					locations = sim_density_search(padded_choices,s.subslice.ret,priority_queue_size,slice_path)
				#locations = sim_density_search(padded_choices,s.subslice.ret,priority_queue_size,slice_path)
				'''
				locations = [[np.array([141, 56])],
							 [np.array([60, 97])],
							 [np.array([116, 188])],
							 [np.array([60, 81])]]
							 #[np.array([129,100])]]
				'''

				#print locations
				print('...DONE!\n')

				print('Starting rotational search...')
				if tiered:
					slices = (s_small,s_small_flipped,s,s_flipped)
					choices_tuple = (padded_choices_small, padded_choices)
					primarySearch(choices_tuple,slices,locations)
				else:
					slices = (s,s_flipped,s_small,s_small_flipped)
					choices_tuple = (padded_choices, padded_choices_small)
					primarySearch(choices_tuple,slices,locations)
				print('...DONE!\n')

				print('Writing results...')

				s_rel = s_small if tiered else s
				s_rel_flipped = s_small_flipped if tiered else s_flipped

				choices_rel = padded_choices_small if tiered else padded_choices

				f.write('subslice identified in %s:...\n\n' % slice_name)
				f.write('\t-Location in target slice: ' + str(s_rel.subslice.start) + '\n')
				f.write('\t-Variance (within retinal rings): ' + str(s_rel.subslice.ret.getVar()) + '\n')
				#f.write('\t-Unrefined Density: ' + str(s.primary.ret.getUnrefinedDensity()) + '\n')
				f.write('\t-Density: ' + str(s_rel.subslice.ret.getDensity()) + '\n')

				f.write('\nResults:...\n\n')

				writeResults(choices_rel,s_rel,s_rel_flipped,f,slice_path)
				print('...DONE!\n')

	end_t = time.time()

	print('\nTime elapsed: %f' % (end_t - start_t))


# searches through queue of possibilities, invoking a rotational search for each
# locations is a list of regions to investigate: 3-tuples of form (choice_index, sx, sy)
def primarySearch(choices_tuple,slices,locations):
	s,s_flipped,_,_ = slices
	choices,_ = choices_tuple
	nIterations = 0
	for c_locs in locations:
		nIterations += len(c_locs)

	counter = 0
	for c_index in range(len(choices)):
		s.results['subslice'].append({'subslice':Result((0,0),0,0),
								  	'slice':Result((0,0),0,0)})

		s.results['slice'].append({'subslice':Result((0,0),0,0),
								  	'slice':Result((0,0),0,0)})

		s.results['hybrid'].append({'subslice':Result((0,0),0,0),
								  	'slice':Result((0,0),0,0),
									'hybrid_sim':0})

		s_flipped.results['subslice'].append({'subslice':Result((0,0),0,0),
								  	'slice':Result((0,0),0,0)})

		s_flipped.results['slice'].append({'subslice':Result((0,0),0,0),
								  	'slice':Result((0,0),0,0)})

		s_flipped.results['hybrid'].append({'subslice':Result((0,0),0,0),
								  	'slice':Result((0,0),0,0),
									'hybrid_sim':0})

		for sub_loc in locations[c_index]:
			sx, sy = sub_loc

			# creates subspace specified by loc and transforms it to retina-form
			goal_subret = s.subtemplate.createRetina(choices[c_index].pix[sx:sx+s.subsize,sy:sy+s.subsize])

			sub_sim, sub_wedge = rotational_search(s.subslice.ret,goal_subret)    # format: (most_sim, wedge_index)
			slice_sim, slice_wedge, slice_loc = sliceSearch(choices[c_index].pix,s,np.array(sub_loc),sub_wedge)   # format: (most_sim,wedge_index,choice_loc)

			if sub_sim > s.results['subslice'][c_index]['subslice'].sim:
			   s.results['subslice'][c_index] = {'subslice':Result(sub_loc,sub_wedge,sub_sim),
												'slice':Result(slice_loc,slice_wedge,slice_sim)}

			if slice_sim > s.results['slice'][c_index]['slice'].sim:
				s.results['slice'][c_index] = {'subslice':Result(sub_loc,sub_wedge,sub_sim),
												   'slice':Result(slice_loc,slice_wedge,slice_sim)}

			hybrid_sim = np.average([sub_sim,slice_sim],weights=similarity_wts)

			if hybrid_sim > s.results['hybrid'][c_index]['hybrid_sim']:
				s.results['hybrid'][c_index] = {'subslice':Result(sub_loc,sub_wedge,sub_sim),
												   'slice':Result(slice_loc,slice_wedge,slice_sim),
												   'hybrid_sim':hybrid_sim}

			sub_sim, sub_wedge = rotational_search(s_flipped.subslice.ret,goal_subret)    # format: (most_sim, wedge_index)
			slice_sim, slice_wedge, slice_loc = sliceSearch(choices[c_index].pix,s_flipped,np.array(sub_loc),sub_wedge)   # format: (most_sim,wedge_index,choice_loc)

			if sub_sim > s_flipped.results['subslice'][c_index]['subslice'].sim:
			   s_flipped.results['subslice'][c_index] = {'subslice':Result(sub_loc,sub_wedge,sub_sim),
												'slice':Result(slice_loc,slice_wedge,slice_sim)}

			if slice_sim > s_flipped.results['slice'][c_index]['slice'].sim:
				s_flipped.results['slice'][c_index] = {'subslice':Result(sub_loc,sub_wedge,sub_sim),
												   'slice':Result(slice_loc,slice_wedge,slice_sim)}

			hybrid_sim = np.average([sub_sim,slice_sim],weights=similarity_wts)

			if hybrid_sim > s_flipped.results['hybrid'][c_index]['hybrid_sim']:
				s_flipped.results['hybrid'][c_index] = {'subslice':Result(sub_loc,sub_wedge,sub_sim),
												   'slice':Result(slice_loc,slice_wedge,slice_sim),
												   'hybrid_sim':hybrid_sim}

			if counter % 50 == 0:
				print(str(round(100*(float(counter)/nIterations),2)) + r'% done')

			counter += 1

	if tiered:
		refined_search(choices_tuple, slices)

def refined_search(choices_tuple, slices):
	s_small,s_small_flipped,s,s_flipped = slices
	padded_choices_small, padded_choices = choices_tuple

def sliceSearch(choiceSpace,s,sub_loc,sub_wedge_index):
	min_angle = (sub_wedge_index - 0.5) / s.subtemplate.nWedges * tao
	max_angle = (sub_wedge_index + 0.5) / s.subtemplate.nWedges * tao

	radians_per_slice_wedge = tao / s.template.nWedges
	min_slice_wedge = int(math.floor(min_angle / radians_per_slice_wedge))
	max_slice_wedge = int(math.ceil(max_angle / radians_per_slice_wedge))

	# print min_slice_wedge, max_slice_wedge

	most_sim = (0,0,(0,0))
	target_ret = s.ret.copy()
	target_ret.rotate(min_slice_wedge)
	for wedge in range(min_slice_wedge,max_slice_wedge+1):
		angle = float(wedge) / s.template.nWedges * tao
		# gets vector to sublice center in easel slot's coordinates
		vectorToSlice = sub_loc + (s.subslice_center - s.subslice.start)
		# gets vector to slice center in easel slot's coordinates
		vectorToSlice += rotate(s.center_delta,angle)
		# gets vector to slice upper right corner in easel slot's coordinates
		vectorToSlice -= s.slice_center
		sx, sy = closestPixel(vectorToSlice)
		# print sx, sy, wedge, angle*360 / tao

		# creates subspace and transforms it to retina-form
		subspace = choiceSpace[sx:sx+s.size,sy:sy+s.size]
		if subspace.shape == (s.size,s.size):
			goal_ret = s.template.createRetina(subspace)
			# goal_ret.visualize('test_slot_%d.jpg' % wedge)
			mm = sim.maxMin(target_ret.retina,goal_ret.retina)
			if mm > most_sim[0]:
				most_sim = (mm,wedge,(sx,sy))

			# ad = sim.absDiff(target_ret.retina,goal_ret.retina,relevant=True)
			# print ad
			# if ad > most_sim[0]:
				# most_sim = (ad,wedge,(sx,sy))

		target_ret.rotate()

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

				if global_counter % 5000 == 0:
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
	f.write('\nHybrid Results:\n\n')
	writeSubResults(choices,s,s_flipped,'hybrid',f,path)

	f.write('\nSubslice Results:\n\n')
	writeSubResults(choices,s,s_flipped,'subslice',f,path)

	f.write('\nSlice Results:\n\n')
	writeSubResults(choices,s,s_flipped,'slice',f,path)

def writeSubResults(choices,s,s_flipped,result_type,f,path):
	size = s.size
	subsize = s.subsize
	nWedges = s.template.nWedges
	sub_nWedges = s.subtemplate.nWedges

	best_choice = -1
	flipped = False
	if result_type == 'hybrid':
		best_choice_nonflipped = np.argmax([res['hybrid_sim'] for res in s.results[result_type]])
		best_choice_flipped = np.argmax([res['hybrid_sim'] for res in s_flipped.results[result_type]])
		flipped = s.results[result_type][best_choice_nonflipped]['hybrid_sim'] < \
					s_flipped.results[result_type][best_choice_flipped]['hybrid_sim']
		best_choice = best_choice_flipped if flipped else best_choice_nonflipped
	else:
		best_choice_nonflipped = np.argmax([res[result_type].sim for res in s.results[result_type]])
		best_choice_flipped = np.argmax([res[result_type].sim for res in s_flipped.results[result_type]])
		flipped = s.results[result_type][best_choice_nonflipped][result_type].sim < \
					s_flipped.results[result_type][best_choice_flipped][result_type].sim
		best_choice = best_choice_flipped if flipped else best_choice_nonflipped

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
		if result_type == 'hybrid':
			flipped = res_nonflipped['hybrid_sim'] < res_flipped['hybrid_sim']
		else:
			flipped = res_nonflipped[result_type].sim < res_flipped[result_type].sim

		res = res_flipped if flipped else res_nonflipped
		#choice = choices[c_index]
		f.write('\tSlot %d:\n\n' % c_index)

		f.write('\t\tSubslice Region:\n')
		f.write('\t\t\t-Flipped? %r\n' % flipped)
		f.write('\t\t\t-Location: %s\n' % str(res['subslice'].start))
		f.write('\t\t\t-Rotation: wedge %d out of %d, which corresponds to the\n\t\t\tsubslice being rotated %.2f degrees anti-clockwise\n'
			% (res['subslice'].wedge_index,sub_nWedges,round(float(res['subslice'].wedge_index)/sub_nWedges * 360, 2)))
		f.write('\t\t\t-Similarity: %f\n\n' % res['subslice'].sim)
		sub_ret = s_flipped.subslice.ret.copy() if flipped else s.subslice.ret.copy()
		sub_ret.rotate(res['subslice'].wedge_index)
		sub_ret.visualize(os.path.join(directory,'subslice_target.jpg'))
		px, py = res['subslice'].start
		subslice_subspace = choices[c_index].pix[px:px+subsize,py:py+subsize]
		s.subtemplate.createRetina(subslice_subspace).visualize(os.path.join(directory,'subslice_choice.jpg'))

		f.write('\t\tSlice:\n')
		f.write('\t\t\t-Flipped? %r\n' % flipped)
		f.write('\t\t\t-Location: %s\n' % str(res['slice'].start))
		f.write('\t\t\t-Rotation: wedge %d out of %d, which corresponds to the\n\t\t\tslice being rotated %.2f degrees anti-clockwise\n'
			% (res['slice'].wedge_index,nWedges,round(float(res['slice'].wedge_index)/nWedges * 360, 2)))
		f.write('\t\t\t-Similarity: %f\n' % res['slice'].sim)
		s_ret = s_flipped.ret.copy() if flipped else s.ret.copy()
		s_ret.rotate(res['slice'].wedge_index)
		s_ret.visualize(os.path.join(directory,'slice_target.jpg'))
		sx, sy = res['slice'].start
		s_subspace = choices[c_index].pix[sx:sx+size,sy:sy+size]
		s.template.createRetina(s_subspace).visualize(os.path.join(directory,'slice_choice.jpg'))

		if result_type == 'hybrid':
			f.write('\n\t\tHybrid Similarity: %f\n' % res['hybrid_sim'])

		f.write('\n')

		choiceIm = choice.im
		boxes = []
		box_index = 0
		for start, box_size in [(res['subslice'].start,subsize), \
								(res['slice'].start,size)]:
			end = start + np.array([box_size-1,box_size-1])
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
