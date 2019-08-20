import os
import csv

problem_list = ['03_scaled',
				'04_scaled',
 				'05_scaled',
 				'06_scaled',
 				'07_scaled',
 				'08_scaled',
 				'09',
 				'10_BW',
 				'11_scaled',
 				'12',
 				'13',
 				'14',
                       'sample_multi_slot_hard']
# problem_list = ['11_scaled']

problems_dir = r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\problems'

# slice_dir = r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\slice_based_approach\tiered_results\run0B'
subslice_dir = r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\subslice_based_approach\two_region_results\run_transparent'

def main():
	output = [['Problem', 'Choice', 'Slice', 'Slice-Slot', 'Subslice-Slot', 'Real Slot', '', 'Slice Correct?', 'Subslice Correct?']]
	for prob in problem_list:
		solution = []

		with open(os.path.join(problems_dir, prob, 'solution.txt'), 'rb') as f:
			solution = f.readlines()

		solution = [e.split()[1].strip() for e in solution]

		for i, choice in enumerate(os.listdir(os.path.join(subslice_dir, prob))):
			for c_slice in os.listdir(os.path.join(subslice_dir, prob, choice)):
				# slice_path = os.path.join(slice_dir, prob, choice, c_slice)
				subslice_path = os.path.join(subslice_dir, prob, choice, c_slice)

				slice_slot = ''
				subslice_slot = ''
				# with open(os.path.join(slice_path, 'console_readout.txt'), 'rb') as f:
				# 	slice_results = f.readlines()
				# 	slice_slot = slice_results[10].strip().split()[-1]
				with open(os.path.join(subslice_path, 'console_readout.txt'), 'rb') as f:
					subslice_results = f.readlines()
					subslice_slot = subslice_results[23].strip().split()[-1]

				# slice_isCorr = str(int(slice_slot == solution[i]))
				slice_isCorr = ''
				subslice_isCorr = str(int(subslice_slot == solution[i]))
				#subslice_isCorr = '1'

				output.append([prob, choice, c_slice, slice_slot, subslice_slot, solution[i], '', slice_isCorr, subslice_isCorr])
			output.append(['---', '---', '---', '---', '---', '---', '---', '---', '---'])
		output.append(['---', '---', '---', '---', '---', '---', '---', '---', '---'])

	with open('subslice_2.csv', 'wb') as file:
		wr = csv.writer(file,delimiter=',')
		wr.writerows(output)

if __name__ == '__main__':
	main()
