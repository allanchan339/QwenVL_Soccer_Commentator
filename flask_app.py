from flask import Flask, request, jsonify, send_from_directory, render_template_string
from werkzeug.utils import secure_filename
import os
import cv2

#WARN: Dont use this in production as no security measures are implemented

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded_videos'
PROCESSED_FOLDER = 'processed_videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        # Extract avatar (first frame)
        return jsonify({'message': 'Video uploaded successfully', 'filename': filename}), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/send_processed_video/<filename>', methods=['GET'])
def send_processed_video(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5001, debug=debug_mode)