from flask import Flask, Response, render_template, request, redirect, url_for, send_file
from Detector import Detector
import io

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('select.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    file.save('static/uploads/' + file)

@app.route('/process', methods=['POST'])
def process():
    path_video = request.json.get('path_video')
    rois = request.json.get('rois')

    detector = Detector(path=path_video, rois=rois)
    detector.process()

    #detector.detect_peaks()

    df = detector.get_results().to_csv(index=False)
    headers = {
        'Content-Disposition': 'attachment; filename=mydata.csv',
        'Content-Type': 'text/csv',
    }

    return Response(
        df.encode('utf-8'),
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=timeseries.csv"})

if __name__ == '__main__':
    app.run(debug=True)
