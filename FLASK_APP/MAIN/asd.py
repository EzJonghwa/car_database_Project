from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
import cx_Oracle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# 모델 로드
model = load_model('./cat_model.h5')

# 클래스 라벨 설정
labels = ['abyssinian', 'american-bobtail', 'american-curl', 'american-shorthair',
          'american-wirehair', 'australian-mist', 'balinese', 'bengal', 'bombay',
          'british-longhair', 'british-shorthair', 'burmese', 'burmilla', 'chartreux',
          'cornish-rex', 'cymric', 'devon-rex', 'don-sphynx', 'egyptian-mau', 'exotic-shorthair',
          'german-rex', 'havana', 'japanese-bobtail', 'khao-manee', 'korat',
          'kurilian-bobtail', 'laperm', 'maine-coon', 'manx', 'munchkin', 'neva-masquerade',
          'norwegian-forest-cat', 'ocicat', 'oriental', 'persian', 'peterbald',
          'pixiebob', 'ragamuffin', 'ragdoll', 'russian-blue', 'sacred-birman',
          'savannah', 'scottish-fold', 'scottish-straight', 'selkirk-rex', 'siamese',
          'siberian', 'singapura', 'snowshoe', 'sokoke', 'somali', 'sphynx', 'thai',
          'tonkinese', 'turkish-angora', 'turkish-van']

# 예측 함수
def fn_predict(image_path):
    image = load_img(image_path, target_size=(224, 224))
    test = img_to_array(image).reshape((1, 224, 224, 3))
    pred = model.predict(test)
    top_5_idx = np.argsort(pred[0])[::-1][:5]
    result = {labels[idx]: f"{pred[0][idx] * 100:.2f}%" for idx in top_5_idx}
    return result

# Oracle 데이터베이스에서 데이터 가져오기
def get_data_from_db(query):
    conn = cx_Oracle.connect("personal", "personal", "192.168.0.144:1521/xe")
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# 추천 함수에 필요한 매트릭스 생성 및 유사도 계산
def get_similarity_matrix():
    data = get_data_from_db("""
                SELECT a.cat_id, a.charactor_id, b.charactor_list
                FROM cat_character a, characterlist b
                WHERE a.charactor_id = b.charactor_id
                ORDER BY a.cat_id, a.charactor_id
    """)
    attribute_df = data.pivot_table(index='CAT_ID', columns='CHARACTOR_LIST', aggfunc='size', fill_value=0)
    similarity_matrix = cosine_similarity(attribute_df)
    return pd.DataFrame(similarity_matrix, index=attribute_df.index, columns=attribute_df.index)

# 추천 품종 함수
def recommend_similar_breeds(cat_id, similarity_df, top_n=5):
    if cat_id in similarity_df.index:
        similar_cats = similarity_df[cat_id].sort_values(ascending=False)[1:top_n + 1]
        return {f"{i+1}": breed_id for i, breed_id in enumerate(similar_cats.index)}
    return None

@app.route('/', methods=['GET'])
def main():
    return render_template('main.html')

@app.route('/board')
def board():
    query = """
    SELECT b.cat_image, a.* 
    FROM (SELECT cat_nm, cat_id, cat_from, cat_category, cat_size FROM cat_info) a,
         (SELECT cat_id, cat_image FROM cat_img) b
    WHERE a.cat_id = b.cat_id
    ORDER BY a.cat_id
    """
    data = get_data_from_db(query)
    return render_template('board.html', data=data.to_dict(orient='records'))

@app.route('/boardview/<int:id>')
def boardview(id):


        # print("로그인 상태")
    return render_template('boardview.html', board=board[0], num=id)



@app.route('/search', methods=['GET', 'POST'])
def search():
    similarity_df = get_similarity_matrix()
    if request.method == 'POST':
        cat_id = request.form['cat_id']
        similar_breeds = recommend_similar_breeds(cat_id, similarity_df)
        return render_template('search.html', cat_id=cat_id, similar_breeds=similar_breeds)
    return render_template('search.html', similar_breeds=None)

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file and file.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            file.save(image_path)
            result = fn_predict(image_path)
            return render_template('index.html', result=result, image_path=image_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
