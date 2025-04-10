import os
import subprocess
from datetime import datetime, datetime as dt
import glob
import re
import logging

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify, render_template, send_file


app = Flask(__name__)
OUTPUT_DIR = "tweets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stemmer = StemmerFactory().create_stemmer()

def load_inset_lexicon():
    try:
        df = pd.read_csv('inset_lexicon.csv')
        logger.info("InSet Lexicon loaded dari file.")
        return dict(zip(df['kata'], df['skor']))
    except FileNotFoundError:
        logger.warning("InSet Lexicon tidak ditemukan, pakai fallback.")
        return {
            'bagus': 3, 'hebat': 3, 'suka': 2, 'senang': 2,
            'jelek': -3, 'buruk': -3, 'benci': -2, 'kecewa': -2,
            'biasa': 0, 'cukup': 0, 'mantap': 2, 'gagal': -2
        }

inset_dict = load_inset_lexicon()
vader_analyzer = SentimentIntensityAnalyzer()
vader_analyzer.lexicon.update({
    'bagus': 2.0, 'hebat': 2.5, 'suka': 1.5, 'senang': 1.8,
    'jelek': -2.0, 'buruk': -2.5, 'benci': -1.8, 'kecewa': -1.5,
    'biasa': 0.0, 'cukup': 0.0, 'mantap': 2.0, 'gagal': -2.0
})
sentistrength_dict = {
    'bagus': 3, 'hebat': 3, 'suka': 2, 'senang': 2,
    'jelek': -3, 'buruk': -3, 'benci': -2, 'kecewa': -2,
    'biasa': 0, 'cukup': 0, 'mantap': 2, 'gagal': -2, 'tidak': -1
}

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|@\w+|#\w+|[^\w\s]', '', text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def validate_date(date_str):
    try:
        dt.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def label_inset(text):
    if not text:
        return 'Netral'
    tokens = word_tokenize(text)
    score = sum(inset_dict.get(stemmer.stem(token), 0) for token in tokens)
    return 'Positif' if score > 0 else 'Negatif' if score < 0 else 'Netral'

def label_vader(text):
    if not text:
        return 'Netral'
    scores = vader_analyzer.polarity_scores(text)
    return 'Positif' if scores['compound'] > 0.05 else 'Negatif' if scores['compound'] < -0.05 else 'Netral'

def label_sentistrength(text):
    if not text:
        return 'Netral'
    tokens = word_tokenize(text)
    score = 0
    for i, token in enumerate(tokens):
        if token == 'tidak' and i + 1 < len(tokens):
            score += sentistrength_dict.get(stemmer.stem(tokens[i + 1]), 0) * -1
        else:
            score += sentistrength_dict.get(stemmer.stem(token), 0)
    return 'Positif' if score > 0 else 'Negatif' if score < 0 else 'Netral'

LEXICON_METHODS = {
    'inset': label_inset,
    'vader': label_vader,
    'sentistrength': label_sentistrength
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.get_json()
    if not data:
        logger.error("Request JSON kosong.")
        return jsonify({'message': 'Data tidak valid.'}), 400

    required = ['keyword', 'startDate', 'endDate', 'authToken']
    if not all(data.get(k) for k in required):
        logger.warning("Field wajib kosong.")
        return jsonify({'message': 'Semua field harus diisi.'}), 400

    keyword = data['keyword'].strip()
    start_date = data['startDate']
    end_date = data['endDate']
    auth_token = data['authToken'].strip()

    if len(keyword) < 3:
        logger.warning(f"Keyword terlalu pendek: {keyword}")
        return jsonify({'message': 'Keyword minimal 3 karakter.'}), 400
    if not validate_date(start_date) or not validate_date(end_date):
        logger.warning(f"Tanggal tidak valid: {start_date}, {end_date}")
        return jsonify({'message': 'Format tanggal harus YYYY-MM-DD.'}), 400
    if dt.strptime(start_date, '%Y-%m-%d') > dt.strptime(end_date, '%Y-%m-%d'):
        logger.warning(f"Tanggal mulai lebih besar dari tanggal selesai: {start_date} > {end_date}")
        return jsonify({'message': 'Tanggal mulai harus sebelum tanggal selesai.'}), 400
    if len(auth_token) < 10:
        logger.warning("Auth token terlalu pendek.")
        return jsonify({'message': 'Auth token tidak valid.'}), 400

    timestamp = get_timestamp()
    raw_file = os.path.join(OUTPUT_DIR, f"tweets_{timestamp}.csv")
    preprocessed_file = os.path.join(OUTPUT_DIR, f"preprocessed_{timestamp}.csv")

    # Sesuai CLI kamu
    search_keyword = f"{keyword} since:{start_date} until:{end_date} lang:id"
    limit = 50
    command = f'npx --yes tweet-harvest@2.6.1 -o "{raw_file}" -s "{search_keyword}" -l {limit} --tab "LATEST" --token "{auth_token}"'

    try:
        logger.info(f"Menjalankan scraping: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        logger.debug(f"Return code: {result.returncode}")
        logger.debug(f"Stdout: {result.stdout}")
        logger.debug(f"Stderr: {result.stderr}")

        if result.returncode != 0:
            logger.error(f"Scraping gagal: {result.stderr}")
            return jsonify({'message': f'Error saat scraping: {result.stderr}'}), 500
        
        if not os.path.exists(raw_file):
            logger.info(f"Scraping selesai tapi tidak ada data di file: {raw_file}")
            return jsonify({'message': 'Scraping selesai tapi tidak ada data. Mungkin token invalid atau tidak ada tweet.'}), 200

        df = pd.read_csv(raw_file)
        if 'text' not in df.columns:
            logger.error("Kolom 'text' tidak ada di CSV.")
            return jsonify({'message': 'Kolom "text" tidak ditemukan.'}), 400

        df['cleaned_text'] = df['text'].apply(preprocess_text)
        df.to_csv(preprocessed_file, index=False)
        logger.info(f"Scraping dan praproses selesai: {raw_file}, {preprocessed_file}")

        return jsonify({
            'message': 'Data berhasil dikumpulkan dan diproses!',
            'raw_filename': os.path.basename(raw_file),
            'preprocessed_filename': os.path.basename(preprocessed_file)
        })
    except Exception as e:
        logger.exception(f"Error scraping: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        logger.warning(f"File tidak ditemukan: {filename}")
        return jsonify({'message': 'File tidak ditemukan.'}), 404
    logger.info(f"Mengunduh file: {filename}")
    return send_file(file_path, as_attachment=True)

@app.route('/label')
def label_page():
    return render_template('label.html')

@app.route('/label', methods=['POST'])
def label():
    method = request.form.get('method')
    lexicon = request.form.get('lexicon', 'inset')
    csv_file = request.files.get('csvFile')

    if method not in ['automatic', 'manual']:
        logger.warning(f"Metode tidak valid: {method}")
        return jsonify({'message': 'Metode tidak valid.'}), 400
    if lexicon not in LEXICON_METHODS:
        logger.warning(f"Lexicon tidak valid: {lexicon}")
        return jsonify({'message': 'Lexicon tidak valid.'}), 400

    if csv_file:
        if not csv_file.filename.endswith('.csv'):
            logger.warning(f"File bukan CSV: {csv_file.filename}")
            return jsonify({'message': 'File harus berformat CSV.'}), 400
        timestamp = get_timestamp()
        input_path = os.path.join(OUTPUT_DIR, f"input_{timestamp}.csv")
        csv_file.save(input_path)
    else:
        preprocessed_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "preprocessed_*.csv")))
        if not preprocessed_files:
            logger.warning("Tidak ada file praproses ditemukan.")
            return jsonify({'message': 'Tidak ada file praproses ditemukan.'}), 400
        input_path = preprocessed_files[-1]

    try:
        df = pd.read_csv(input_path)
        if 'cleaned_text' not in df.columns:
            logger.error("Kolom 'cleaned_text' tidak ada di CSV.")
            return jsonify({'message': 'Kolom "cleaned_text" tidak ditemukan.'}), 400

        if method == 'automatic':
            df['sentiment'] = df['cleaned_text'].apply(LEXICON_METHODS[lexicon])
            timestamp = get_timestamp()
            output_file = os.path.join(OUTPUT_DIR, f"labeled_{timestamp}.csv")
            df.to_csv(output_file, index=False)
            preview_data = df[['cleaned_text', 'sentiment']].head(100).to_dict(orient='records')
            logger.info(f"Pelabelan otomatis selesai: {output_file}")
            return jsonify({
                'message': f'Pelabelan otomatis selesai dengan {lexicon.capitalize()}!',
                'filename': os.path.basename(output_file),
                'preview_data': preview_data
            })
        else:
            data = df[['cleaned_text']].to_dict(orient='records')
            logger.info("Siap untuk pelabelan manual.")
            return jsonify({
                'message': 'Siap untuk pelabelan manual.',
                'method': 'manual',
                'data': data,
                'original_filename': os.path.basename(input_path)
            })
    except Exception as e:
        logger.exception(f"Error pelabelan: {str(e)}")
        return jsonify({'message': f'Error saat pelabelan: {str(e)}'}), 500
    finally:
        if csv_file and os.path.exists(input_path):
            os.remove(input_path)

@app.route('/save_labels', methods=['POST'])
def save_labels():
    data = request.get_json()
    if not data or 'labels' not in data or 'filename' not in data:
        logger.warning("Data JSON untuk save_labels tidak valid.")
        return jsonify({'message': 'Data tidak valid.'}), 400

    labels = data['labels']
    original_filename = data['filename']

    try:
        input_path = os.path.join(OUTPUT_DIR, original_filename)
        if not os.path.exists(input_path):
            logger.warning(f"File asli tidak ditemukan: {original_filename}")
            return jsonify({'message': 'File asli tidak ditemukan.'}), 404

        df = pd.read_csv(input_path)
        df['sentiment'] = 'Netral'
        for label in labels:
            index = int(label['index'])
            if index < 0 or index >= len(df):
                logger.warning(f"Indeks tidak valid: {index}")
                return jsonify({'message': 'Indeks tidak valid.'}), 400
            df.at[index, 'sentiment'] = label['sentiment']
        
        timestamp = get_timestamp()
        output_file = os.path.join(OUTPUT_DIR, f"labeled_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Label manual disimpan: {output_file}")
        return jsonify({
            'message': 'Label berhasil disimpan!',
            'filename': os.path.basename(output_file)
        })
    except Exception as e:
        logger.exception(f"Error menyimpan label: {str(e)}")
        return jsonify({'message': f'Error saat menyimpan: {str(e)}'}), 500

@app.route('/preview_data/<filename>', methods=['POST'])
def preview_data(filename):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        logger.warning(f"File preview tidak ditemukan: {filename}")
        return jsonify({'error': 'File tidak ditemukan.'}), 404

    try:
        draw = int(request.form.get('draw', 1))
        start = int(request.form.get('start', 0))
        length = int(request.form.get('length', 10))
        search_value = request.form.get('search[value]', '')

        df = pd.read_csv(file_path)
        if 'cleaned_text' not in df.columns or 'sentiment' not in df.columns:
            logger.error("Kolom 'cleaned_text' atau 'sentiment' tidak ada.")
            return jsonify({'error': 'Kolom tidak lengkap.'}), 400

        if search_value:
            df = df[df['cleaned_text'].str.contains(search_value, case=False, na=False)]

        total_records = len(df)
        data = df.iloc[start:start + length][['cleaned_text', 'sentiment']].to_dict(orient='records')
        logger.debug(f"Preview data dikirim: {start} - {start + length}")
        return jsonify({
            'draw': draw,
            'recordsTotal': total_records,
            'recordsFiltered': total_records,
            'data': data
        })
    except Exception as e:
        logger.exception(f"Error preview: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    from os import environ
    if environ.get('FLASK_ENV') != 'production':
        logger.info("Menjalankan dalam mode development...")
        app.run(debug=True, port=5000)
    else:
        logger.info("Mode production, gunakan WSGI server seperti Gunicorn.")