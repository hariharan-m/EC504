from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
from kdTree import *
# UPLOAD_FOLDER = "/home/hari/Workspace/EC504/Server/uploaded_files/"
UPLOAD_FOLDER = "/home/hari/Workspace/EC504/Server/static/output_files"



kdtree = None
tree = None
dataSet = []
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

@app.route('/')
def upload_file_view():
	return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		filename = secure_filename(f.filename)
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
		kdTreeBuild()
		kdTreeSearch()
		return gallery()

def kdTreeBuild():
	global kdtree
	global dataSet
	global tree
	with open("test_data.txt") as f:
		for line in f:
			arr = list(map(float, line.split()))
			dataSet.append(arr)
	print(dataSet)
	x = [3, 4]
	kdtree = KdTree()
	tree = kdtree.create(dataSet, 0)
# kdtree.preOrder(tree)

def kdTreeSearch():
	x=3
	print("The NN of " + str(x) + " is " + str(kdtree.search(tree, x)))
	print("The 3NN of " + str(x) + " is " + str(kdtree.Ksearch(tree, x, 3)))

@app.route('/gallery')
def gallery():
	 hists = os.listdir('/home/hari/Workspace/EC504/Server/static/output_files')
	 hists = ['output_files/' + file for file in hists]
	 return render_template('report.html', hists = hists)


if __name__ == '__main__':
	app.config['uploaded_files']	
	app.run(debug = True)
