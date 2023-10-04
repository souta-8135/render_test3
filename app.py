import shutil
from flask import Flask, render_template, session, request, redirect, url_for
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

#coding:utf-8
#coding:utf-8
import numpy as np
import cv2
import math
import copy
from matplotlib import pyplot as plt

#インスタンスの作成
app = Flask(__name__)

#暗号鍵の作成
key = os.urandom(21)
app.secret_key = key

#idとパスワードの設定
id_pwd = {'Conan': 'Heiji'}

#データベース設定
URI = 'postgresql://postgres:souta8135@localhost/flasktest'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = URI
db = SQLAlchemy(app)

#テーブル内容の設定
class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title= db.Column(db.String(30), index=True, unique=True)
    file_path = db.Column(db.String(64), index=True, unique=True)
    dt = db.Column(db.DateTime, nullable=False, default=datetime.now)

#テーブルの初期化
@app.cli.command('initdb')
def initdb():
    db.create_all()
    
#メイン
@app.route('/')
def index():
    if not session.get('login'):
        return redirect(url_for('login'))
    else:
        data = Data.query.all()
        return render_template('index.html', data=data)

@app.route('/login')
def login():
        return render_template('login.html')

@app.route('/logincheck', methods=['POST'])
def logincheck():
    user_id = request.form['user_id']
    password = request.form['password']

    if user_id in id_pwd:
        if password == id_pwd[user_id]:
            session['login'] = True
        else:
            session['login'] = False
    else:
        session['login'] = False
    
    if session['login']:
        return redirect(url_for('index'))
    else:
        return redirect(url_for('login'))
    
@app.route('/logout')
def logout():
    session.pop('login',None)
    return redirect(url_for('index'))

#ファイルアップロード
@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/register', methods=['POST'])
def register():
    title = request.form['title']
    f = request.files['file']
    file_path = 'static/' + secure_filename(f.filename)
    f.save(file_path)

    registered_file = Data(title=title, file_path=file_path)
    db.session.add(registered_file)
    db.session.commit()

    return redirect(url_for('index'))

@app.route('/delete/<int:id>', methods=['GET'])
def delete(id):
    data = Data.query.get(id)
    delete_file = data.file_path
    db.session.delete(data)
    db.session.commit()
    os.remove(delete_file)
    return redirect(url_for('index'))

@app.route('/show')
def show():
    # 画像ファイルの読み込み(カラー画像(3チャンネル)として読み込まれる)
    img = cv2.imread("C:\\Users\\koizumi\\static\\result.jpg")

    # 画像の表示
    cv2.imshow("結果画像", img)

    # キー入力待ち(ここで画像が表示される)
    cv2.waitKey()
    return render_template('index.html')      


@app.route('/snake')
def snake():
#　日付を取得する,フォルダを作成する
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M")
    dir_for_output = "./python/date_save_dir/" + current_time

    # カメラを開く
    cap = cv2.VideoCapture(0)

    # 画像をキャプチャする
    ret, frame = cap.read()

    # 画像を保存する
    dir_save = dir_for_output + " image.jpg"
    cv2.imwrite(dir_save, frame)

    shutil.copyfile(dir_save, "./python/date_save_dir/data/input.jpg")

    # カメラを閉じる
    cap.release()
    
    #画像の読み込み
    test = cv2.imread("./python/date_save_dir/data/input.jpg", cv2.IMREAD_COLOR)#BGRなので気をつける

    gray_test = cv2.imread("./python/date_save_dir/data/input.jpg",cv2.IMREAD_GRAYSCALE)
    height = test.shape[0]
    width = test.shape[1] 
    #画像の書き出し
    cv2.imwrite('./Img/test.jpg', test)
    cv2.imwrite('./Img/gray_test.jpg',gray_test)

    N = 2000

    v = np.zeros((N,2))
    start_v = np.zeros((N,2))
    vec_g = np.zeros(2)
    for i in range(0,N):
        if(i<N/4):
            v[i] = [height/(N/4)*i,0]
        elif(i<2*N/4):
            v[i] = [height-1,width/(N/4)*(i-N/4)]
        elif(i<3*N/4):
            v[i] = [height-1-height/(N/4)*(i-2*N/4),width-1]
        else:
            v[i] = [0,width-1 - width/(N/4)*(i-3*N/4)]


    #初期輪郭点が円形の場合はこっち
    # for i in range(0,N):
    #     v[i] = [ height/2*math.sin(2*math.pi*i/N)+height/2, width/2*math.cos(2*math.pi*i/N)+width/2]
    #     start_v[i] = [ height/2*math.sin(2*math.pi*i/N)+height/2, width/2*math.cos(2*math.pi*i/N)+width/2]


    start_v = copy.deepcopy(v)
    display = copy.deepcopy(test)

    #パラメータ
    alpha =1
    beta = 1
    gamma = 10
    kappa =1

    def EpsIn(vec0,vec1,vec2):#test
        value = 0
        value += alpha*np.linalg.norm(vec1-vec0)**2+beta*np.linalg.norm(vec2-2*vec1+vec0)**2
        value /= 2
    #     print("In:"+str(value))
        return value

    def EpsEx(vec0,pix):#gray
        value = 0
        x = int(vec0[0])
        y = int(vec0[1])

        if(x+1 >= height or y+1 >= width):
            return float('inf') 
        else:
            I = [abs(int(pix[x+1,y]) - int(pix[x,y])) ,abs(int(pix[x,y+1])-int(pix[x,y]))]
            value = -gamma*np.linalg.norm(I)**2
    #         print("Ex:"+str(value))
            return value

    def EpsCon(vec0,vec_g):#test
        value = 0
        value += kappa*np.linalg.norm((vec0[0] - vec_g[0],vec0[1]-vec_g[1]))**2
    #     print("Con:"+str(value))
        return value

    def Energy(vec0,vec1,vec2,vec_g,pix):
        value = 0
        value = EpsIn(vec0,vec1,vec2)+EpsEx(vec0,pix)+EpsCon(vec0,vec_g)
    #     print("Result:"+str(value))
        return value
    #探索
    n = 500
    dx = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    dy = [1, 0, -1, 1, 0, -1, 1, 0, -1]
    # dx = [1,1,1,0,0,0,-1,-1,-1]
    # dy = [1,-1,0,1,-1,0,1,-1,0]
    #210
    #543
    #876

    flag = 4
    for loop in range(0,n):
        for i in range(0,N):
            flag = 4
            eps_min = float('inf') 
            vec_g = [0,0]

            #重心中心にするならこれ
    #         for j in range(0,N):
    #             vec_g += [v[j,0],v[j,1]]

            for j in range(0,9):            
                move  = [v[i,0]+dx[j], v[i,1]+dy[j]]
                if(move[0] < 0 or move[1] < 0 or move[0] >= height  or move[1] >= width):
                    continue #はみ出し処理

                #重心中心にするならこれ
                #vec_g += [dx[j],dy[j]]
                #vec_g =[vec_g[0]/N, vec_g[1]/N]
                #画像中心を基準に
                vec_g = [int(height/2),int(width/2)]

                energy = Energy(move,v[(i+1)%N],v[(i+2)%N],vec_g,gray_test)
                if(eps_min>energy):
                    eps_min = energy
                    flag = j
            v[i] += [dx[flag],dy[flag]]

            #逐次書き出し
        if(loop%10==0):
            cv2.imwrite('./Img/result'+str(loop)+'.jpg', display)
            display = copy.deepcopy(test)
            for i in range(0,N):
                cv2.line(display, (int(v[i,1]),int(v[i,0])), (int(v[(i+1)%N,1]),int(v[(i+1)%N,0])), (0, 255, 0), 2)



    for i in range(0,N):
        cv2.line(display, (int(v[i,1]),int(v[i,0])), (int(v[(i+1)%N,1]),int(v[(i+1)%N,0])), (0, 255, 0), 2)

    for i in range(0,N):
        cv2.line(display, (int(start_v[i,1]),int(start_v[i,0])), (int(start_v[(i+1)%N,1]),int(start_v[(i+1)%N,0])), (255, 0, 0), 2)

    #結果画像の保存
    cv2.imwrite('./Img/result.jpg', display)

    return redirect(url_for('index'))

@app.route('/opencv')
def opencv():
    # カラー画像の読み込み 
    img = cv2.imread('./sample3.jpg', 1)
 
    # グレースケール化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#    単純二値化
    ret, img_binary = cv2.threshold(img_gray,60, 255,cv2.THRESH_BINARY)
    
    # 輪郭抽出
    contours, hierarchy = cv2.findContours(img_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    # 輪郭を元画像に描画
    img_contour = cv2.drawContours(img, contours, 0, (0, 255, 0), -1)

    print(len(contours)) #　←　抽出した輪郭の個数を表示するコードを挿入する
 
    # ここから画像描画
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    plt.show()
    plt.close()

    return redirect(url_for('index'))

#サーバの起動
if __name__ == '__main__':
    app.run(debug=True)