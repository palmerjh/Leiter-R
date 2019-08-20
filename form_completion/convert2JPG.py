import os
from PIL import Image

problems_dir = r'C:\Users\palmerjh\Documents\AIVAS\leiter_demo\actual\form_completion\problems'
problem_list = ['12',
				'13',
				'14']

def main():
	for prob in problem_list:
		folder = os.path.join(problems_dir, prob)
		for choice in os.listdir(os.path.join(folder, 'choices')):
			for slice in os.listdir(os.path.join(folder, 'choices', choice)):
				toJPG(os.path.join(folder, 'choices', choice, slice))

		for slot in os.listdir(os.path.join(folder, 'easel_slots')):
			toJPG(os.path.join(folder, 'easel_slots', slot))

	folder = os.path.join(problems_dir, '15')
	for slot in os.listdir(os.path.join(folder, 'easel_choices')):
		for slice in os.listdir(os.path.join(folder, 'easel_choices', slot)):
			toJPG(os.path.join(folder, 'easel_choices', slot, slice))

	for slice in os.listdir(os.path.join(folder, 'choices')):
		toJPG(os.path.join(folder, 'choices', slice))

def toJPG(file):
	pic_name, ext = file.split('.')
	im = Image.open(file)
	rgb_im = im.convert('RGB')

	par_dir = os.path.abspath(os.path.join(file, os.pardir))
	os.remove(file)
	rgb_im.save(os.path.join(par_dir,'%s.jpg' % pic_name), 'JPEG')

if __name__ == '__main__':
    main()
