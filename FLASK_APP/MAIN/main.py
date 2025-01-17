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

    result = []
    for idx in top_5_idx:
        label = labels[idx]
        confidence = f"{pred[0][idx] * 100:.2f}%"

        # 데이터베이스에서 해당 품종의 이름과 이미지 가져오기
        query = f"""
        
        select c_image , c_nm
       from(SELECT b.cat_image c_image, a.cat_nm c_nm ,a.cat_id c_id 
       ,a.cat_from c_from,a.cat_category c_cate,a.cat_size c_size
       FROM (SELECT cat_nm, cat_id, cat_from, cat_category, cat_size FROM cat_info) a,
            (SELECT cat_id, cat_image FROM cat_img) b
       WHERE a.cat_id = b.cat_id
       ORDER BY a.cat_id)
       where c_id ='{label}'
        """
        cat_info = get_data_from_db(query)

        if not cat_info.empty:
            cat_image = cat_info.iloc[0]['C_IMAGE']
            cat_name = cat_info.iloc[0]['C_NM']
            result.append({
                'label': label,
                'confidence': confidence,
                'cat_image': cat_image,
                'cat_name': cat_name
            })
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

# 데이터베이스에서 고양이 성격 리스트를 가져오는 함수
def get_cat_traits(id):
    query = f"""
    select c_id ,c_nm, c_list
    from (select a.cat_id c_id, a.cat_nm c_nm, d.idd c_idd, d.list c_list
    from cat_info a , 
    (select b.cat_id as id
    , b.charactor_id as idd, c  .charactor_list as list
    from cat_character b , characterlist c 
    where b.charactor_id =c.charactor_id) d
    where a.cat_id = d.id
    order by a.cat_id)
    where c_id ='{id}'
    """
    traits = get_data_from_db(query)

    return traits['C_LIST'].tolist()  # 성격 리스트를 반환




def cat_filter_from():
    query_from="""
        select cat_from
        from cat_info
        group by cat_from
    """

    q_from = get_data_from_db(query_from)
    return q_from['CAT_FROM'].tolist()


def cat_filter_size():
    query_size = """
              select cat_size
              from cat_info
              group by cat_size
          """

    q_size = get_data_from_db(query_size)

    return q_size['CAT_SIZE'].tolist()


def cat_filter_category():
    query_category = """
              select cat_category
              from cat_info
              group by cat_category
          """

    q_category = get_data_from_db(query_category)

    return q_category['CAT_CATEGORY'].tolist()


def get_cat_similar(id):
    query=f"""
    select r_id, c_img , c_nm
        from
                (select a.cat_id c_id, a.RECOMMENDED_CAT_ID r_id, b.c_img c_img, b.c_nm c_nm
                from (select cat_id ,RECOMMENDED_CAT_ID
                from CAT_RECOMMENDATIONS )a ,(select a.cat_id ,b.cat_image c_img ,a.cat_nm c_nm
                from (select cat_nm,cat_id
                from cat_info) a ,(select cat_id, cat_image from cat_img) b
                where a.cat_id = b.cat_id) b
                where a.RECOMMENDED_CAT_ID = b.cat_id)
        where c_id ='{id}'
    """
    similar = get_data_from_db(query)

    return similar.to_dict(orient="records")


@app.route('/', methods=['GET'])
def main():
    return render_template('home.html')

@app.route('/main', methods=['GET'])
def mainpage():
    return render_template('main.html')

# @app.route('/mainpage', methods=['GET'])
# def mainpage():
#     return render_template('main.html')

@app.route('/board', methods=['GET', 'POST'])
def board():
    query = """
    SELECT b.cat_image c_image, a.cat_nm c_nm ,a.cat_id c_id 
    ,a.cat_from c_from,a.cat_category c_cate,a.cat_size c_size
    FROM (SELECT cat_nm, cat_id, cat_from, cat_category, cat_size FROM cat_info) a,
         (SELECT cat_id, cat_image FROM cat_img) b
    WHERE a.cat_id = b.cat_id
    ORDER BY a.cat_id

    """
    data = get_data_from_db(query)
    return render_template('board.html', data=data.to_dict(orient='records'))







@app.route('/boardview/<string:id>')
def boardview(id):
    # 고양이 품종 이름이나 기타 문자열 값을 사용
    query = f"""
    select *
    from (SELECT b.cat_image c_image, a.cat_nm c_nm ,a.cat_id c_id,a.cat_exp c_exp,a.cat_from c_from,a.cat_category c_cate,a.cat_size c_size ,a.cat_care c_care, a.cat_life c_life
    FROM (SELECT cat_nm, cat_id, cat_from, cat_category, cat_size,cat_exp,cat_care,cat_life FROM cat_info) a,
         (SELECT cat_id, cat_image FROM cat_img) b  
    WHERE a.cat_id = b.cat_id
    ORDER BY a.cat_id)
    where c_id = '{id}'
    """
    data = get_data_from_db(query)
    traits = get_cat_traits(id)
    similar = get_cat_similar(id)
    if not data.empty:
        cat_info = data.iloc[0]
        return render_template('boardview.html', cat=cat_info, traits=traits , similar=similar)
    else:
        return render_template('boardview.html', error="고양이 정보를 찾을 수 없습니다.")


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
    app.run(debug=True, host='0.0.0.0')
