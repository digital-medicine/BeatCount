from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    file = request.files['video']

    if file.filename == '':
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('select_roi', filename=filename))

    return redirect(url_for('index'))

@app.route('/select_roi/<filename>')
def select_roi(filename):
    return render_template('select_roi.html', filename=filename)

@app.route('/process', methods=['POST'])
def process():
    filename = request.form['filename']
    roi_coords = request.form.getlist('roi_coords[]')
    # Process the video and perform ROI tracking
    # Implement your video processing code here
    return render_template('result.html')

if __name__ == '__main__':
    app.run()
