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

import math
pi = math.pi

import similarity2 as sim
import taskpic as tp
#import pix2retina as p2r
sys.path.append(r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual')
import retina_newest as r

resize_factor = 4

# for initial coarse density delta search
hn_coarse = 1

# for refined rotation search
#hn_refined = 6

coarse_queue_size = 50   # size of pq for each choice in coarse density search
useTransparentPix = True

problems_dir = r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\problems'

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
# 				'07_scaled',
# 				'08_scaled']

problem_list = ['sample_multi_slot_hard']

choices_dir = 'easel_slots'
targets_dir = 'choices'
results_dir = os.path.join('tiered_results', 'run_transparent_3')

#slice_index = 2

from collections import namedtuple
Result = namedtuple("Result","start wedge_index sim")   # simple container used to hold results for primary,secondary,etc. regions for each choice
FlipResult = namedtuple("Result","start wedge_index flipped sim")

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
				padded = np.zeros((max_dim,max_dim), dtype=pix.dtype)
				padded[start_x:start_x + sx, start_y:start_y + sy] = pix
				slicePic.pix = np.copy(padded)
				slicePic.update()
				pix = slicePic.pix

				# resizes image for faster density_delta search
				slicePic_small = slicePic.resize(resize_factor)
				#pix_small = slicePic_small.pix

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

				print('Initializing....creating templates and retina objects...may take some time....')

				# empirically this has been found to be a good tradeoff of speed vs.
				# efficiency as the retina gets bigger
				hn_refined = int(round(max_dim / 25.0))
				s_small = r.Slice(slicePic_small,prob,target,useTransparentPix=useTransparentPix,hn=hn_coarse)
				# s_small.ret.visualize().show()
				s = r.Slice(slicePic,prob,target,useTransparentPix=useTransparentPix,hn=hn_refined)
				# s.ret.visualize().show()
				print('\n....DONE!!!\n')
				# sys.exit(0)

				temp = s.template
				f.write('Using retina with following parameters:\n')
				f.write('\t-size: %d x %d\n' % (temp.size, temp.size))
				f.write('\t-number of rings: %d\n' % temp.nRings)
				f.write('\t-number of wedges: %d\n' % temp.nWedges)
				f.write('\t-blindspot_radius: %f\n\n' % temp.bs_radius)

				print('Starting coarse density search...')
				# returns locations already rescaled
				locations = sim_density_search(padded_choices_small,s_small.ret,coarse_queue_size,slice_path,padded_choices,slicePic)

				#print locations
				print('...DONE!\n')

				print('Starting rotational search...')
				print(sum([len(c_locs) for c_locs in locations]))
				s_flipped = s.flip()
				primarySearch(padded_choices,s,s_flipped,locations)
				print('...DONE!\n')

				print('Writing results...')

				f.write('Using slice %s:...\n' % slice_name)

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
		s.results.append(Result((0,0),0,0))
		s_flipped.results.append(Result((0,0),0,0))

		for p_loc in locations[c_index]:
			sx, sy = p_loc

			# creates subspace specified by loc and transforms it to retina-form
			goal_ret = s.template.createRetina(choices[c_index].pix[sx:sx+s.size,sy:sy+s.size])

			p_sim, p_wedge = rotational_search(s.ret,goal_ret)    # format: (most_sim, wedge_index)
			if p_sim > s.results[c_index].sim:
			   s.results[c_index] = Result(p_loc,p_wedge,p_sim)

			p_sim, p_wedge = rotational_search(s_flipped.ret,goal_ret)    # format: (most_sim, wedge_index)
			if p_sim > s_flipped.results[c_index].sim:
			   s_flipped.results[c_index] = Result(p_loc,p_wedge,p_sim)

			if counter % 50 == 0:
				print(str(round(100*(float(counter)/nIterations),2)) + r'% done')

			counter += 1

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
def sim_density_search(choices,rot_ret,queueSize,cur_results_dir,choices_original,slice_original):
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
		#print cx, cy, size, size
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

				'''
				if counter < queueSize:
					hq.heappush(most_sim[c_index],(res,(sx,sy)))
				else:
					hq.heappushpop(most_sim[c_index],(res,(sx,sy)))
				'''

				if global_counter % 500 == 0:
					print(str(round(100*(float(global_counter)/nIterations),2)) + r'% done')

				#print(str(counter) + ': ' + str(most_sim['sp']))

				counter += 1
				global_counter += 1

		ssx, ssy = choices_original[c_index].size()
		tx, ty = slice_original.size()

		#print ssx, ssy, tx, ty

		# rescales locations and appends other local locations
		all_sim_choice.sort(reverse=True)
		for delta, pos in all_sim_choice[:queueSize]:
			x,y = pos
			x *= resize_factor
			y *= resize_factor

			# odd-sized box around location for refined search
			searchBox_size = resize_factor + (1 - resize_factor % 2)
			shift = int(searchBox_size) / 2

			x0 = max(x-shift,0)
			y0 = max(y-shift,0)
			xn = min(x+shift,ssx-tx)
			yn = min(y+shift,ssy-ty)

			#print x0, y0, xn, yn

			if (ssx-tx-xn < resize_factor): # handles literal edge cases where pixels are left out due to rounding troubles
				xn = ssx-tx
			if (ssy-ty-yn < resize_factor): # handles literal edge cases where pixels are left out due to rounding troubles
				yn = ssy-ty

			#print x0, y0, xn, yn
			#print()

			for px in range(x0,xn+1):
				for py in range(y0,yn+1):
					most_sim[c_index].append((delta,(px,py)))
		#print(len(most_sim[c_index]))

		deltas, locs = tuple(zip(*most_sim[c_index]))

		# need to flip axes to make points
		point_locs = [(y,x) for (x,y) in locs]

		with open(os.path.join(slice_easel_choice_path, 'queue_deltas.csv'), 'wb') as deltas_file:
			wr = csv.writer(deltas_file,delimiter=',')
			wr.writerows(most_sim[c_index])

		# format: (cutoff, color)
		# cutoffs refer to fraction of distance between worst and best deltas in queue
		'''
		heatmap_input_alt = [	(0.0 , 'violet'),
							(0.5 , 'indigo'),
							(0.8 , 'blue'),
							(0.95 , 'green'),
							(0.97 , 'yellow'),
							(0.98 , 'orange'),
							(0.99 , 'red')	]
		'''

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

		points_im = Image.new('RGB',(ssy,ssx),color='white')
		pointer = ImageDraw.Draw(points_im)

		end_index = 0
		for cutoff in reversed(heatmap_cutoffs[1:]):
			start_index = end_index
			while deltas[end_index] > cutoff:
				end_index += 1

			pointer.point(point_locs[start_index:end_index],fill=heatmap[cutoff])

		final_color = heatmap[heatmap_cutoffs[0]]
		pointer.point(point_locs[end_index:],fill=final_color)

		slot_im = choices_original[c_index].im.convert('RGB')
		overlay_im = Image.blend(points_im, slot_im, alpha=0.2)

		overlay_im.save(os.path.join(slice_easel_choice_path, 'sim_deltas.png'))

	return [[tup[1] for tup in choice] for choice in most_sim]


	#return [[tup[1] for tup in choice] for choice in most_sim]

def writeResults(choices,s,s_flipped,f,path):
	size = s.size
	nWedges = s.template.nWedges

	best_choice_nonflipped = np.argmax([res.sim for res in s.results])
	best_choice_flipped = np.argmax([res.sim for res in s_flipped.results])

	flipped = s.results[best_choice_nonflipped].sim < s_flipped.results[best_choice_flipped].sim
	best_choice = best_choice_flipped if flipped else best_choice_nonflipped

	flip_string = 'Flipped' if flipped else 'Not flipped'
	print('(%s) Best slot: %d' % (flip_string, best_choice))
	f.write('\t(%s) Best Slot: %d\n\n' % (flip_string, best_choice))
	for c_index, choice in enumerate(choices):
		directory = os.path.join(path,'slot%d'%c_index)
		if not os.path.exists(directory):
			os.makedirs(directory)

		res_nonflipped = s.results[c_index]
		res_flipped = s_flipped.results[c_index]

		flipped = res_nonflipped.sim < res_flipped.sim
		res = res_flipped if flipped else res_nonflipped
		#choice = choices[c_index]
		f.write('\tSlot %d:\n\n' % c_index)

		f.write('\t\tSlice in Slot/Choice:\n')
		f.write('\t\t\t-Flipped? %r\n' % flipped)
		f.write('\t\t\t-Location: %s\n' % str(res.start))
		f.write('\t\t\t-Rotation: wedge %d out of %d, which corresponds to the\n\t\t\tslice being rotated %.2f degrees anti-clockwise\n'
			% (res.wedge_index,nWedges,round(float(res.wedge_index)/nWedges * 360, 2)))
		f.write('\t\t\t-Similarity: %f\n\n' % res.sim)
		p_ret = s_flipped.ret.copy() if flipped else s.ret.copy()
		p_ret.rotate(res.wedge_index)
		p_ret.visualize(os.path.join(directory,'slice_rotated_in_slot.jpg'))
		px, py = res.start
		p_subspace = choices[c_index].pix[px:px+s.size,py:py+s.size]
		s.template.createRetina(p_subspace).visualize(os.path.join(directory,'matching_slot_subspace.jpg'))

		f.write('\n')

		choiceIm = choice.im
		boxes = []

		end = res.start + np.array([size-1,size-1])
		boxes.append(([tuple(res.start),tuple(end)],'red'))

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

if __name__ == "__main__":
	results_path = os.path.join(os.getcwd(),results_dir)
	if not os.path.exists(results_path):
		os.makedirs(results_path)
	cProfile.run('main()', os.path.join(os.getcwd(),results_dir,'stats'))
	#main()
