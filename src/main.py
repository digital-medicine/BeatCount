from flask import Flask, Response, render_template, request, redirect, url_for
from Detector import Detector

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
    path_result = request.json.get('path_result')
    rois = request.json.get('rois')

    detector = Detector(path=path_video, destination=path_result, rois=rois)
    detector.process()

    #detector.detect_peaks()
    #detector.save_plots()
    df = detector.get_results()
    print(df)
    return Response(
        df.to_string(),
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=timeseries.csv"})

if __name__ == '__main__':
    app.run(debug=True)
