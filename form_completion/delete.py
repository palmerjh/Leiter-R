import os
import csv
import shutil

results_dirs = [#r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\hybrid_approach\results',
				r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\slice_based_approach\tiered_results']
				#r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\subslice_based_approach\results',
				#r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\subslice_based_approach\three_region_results',
				#r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\subslice_based_approach\two_region_results']
# problem_list = ['11_scaled']


def main():
	#output = [['Problem', 'Choice', 'Slice', 'Slice-Slot', 'Subslice-Slot', 'Real Slot', '', 'Slice Correct?', 'Subslice Correct?']]
	for dir in results_dirs:
		for res in os.listdir(dir):
			print res
			res = os.path.join(dir,res)
			for prob in os.listdir(res):
				if prob == 'stats':
					continue
				print res
				print prob
				prob = os.path.join(res,prob)
				for choice in os.listdir(prob):
					choice = os.path.join(prob,choice)
					for slice in os.listdir(choice):
						slice = os.path.join(choice,slice)
						for slot in os.listdir(slice):
							slot = os.path.join(slice,slot)
							if os.path.isdir(slot):
								file = os.path.join(slot,'queue_deltas.csv')
								if os.path.isfile(file):
									os.remove(os.path.join(file))

if __name__ == '__main__':
	main()
