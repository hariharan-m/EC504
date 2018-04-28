from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
# UPLOAD_FOLDER = "/home/hari/Workspace/EC504/Server/uploaded_files/"
UPLOAD_FOLDER = "/home/hari/Workspace/EC504/Server/static/output_files"
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
		return gallery()

@app.route('/gallery')
def gallery():
    hists = os.listdir('/home/hari/Workspace/EC504/Server/static/output_files')
    hists = ['output_files/' + file for file in hists]
    return render_template('report.html', hists = hists)


if __name__ == '__main__':
	app.config['uploaded_files']	
	app.run(debug = True)
