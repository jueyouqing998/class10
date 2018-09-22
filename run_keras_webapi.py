# -*- coding: utf-8 -*-
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf
import os

#允许上传的图片类型
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])  

#验证上传文件
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#初始化 Flask application 和the Keras model
app = flask.Flask(__name__)

graph = None
model = None

def load_model():
	global graph
	graph = tf.get_default_graph()
	# 导入 pre-trained Keras 模型
	global model
	model = ResNet50(weights="imagenet")

#图像预处理
def prepare_image(image, target):
	# 将图像转换为RGB模式
	if image.mode != "RGB":
		image = image.convert("RGB")
	# 将图像缩放为固定的大小
	image = image.resize(target)
    # 将图像转换为array
	image = img_to_array(image)
    # 根据imagenet_utils的方式来准备图片书ju
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)
	# 返回预处理结果
	return image

# 主要预测过程
@app.route("/predict", methods=["POST"])
def predict():
    #返回数据初始化 
	data = {"success": False}
	#确保输入数据为图片，调用方式为POST
	if flask.request.method == "POST":
		if flask.request.files.get("image") and allowed_file(flask.request.files.get("image").filename):
			# 读入图片数据
			image = flask.request.files["image"].read()
			print(os.path.abspath(flask.request.files.get("image").filename))
			image = Image.open(io.BytesIO(image))
			# 调用预处理程序
			image = prepare_image(image, target=(224, 224))
			# 调用预测程序
			with graph.as_default():
				preds = model.predict(image)
            # 返回预测结果
			results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []
			# 打印预测结果
			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)
			# 如果运行到这里，显然是成功了
			data["success"] = True
	# 将返回结果jsonify
	#return res.jsonify(data)
	# 解决Apache跨域访问
	res=flask.make_response(flask.jsonify(data))
	res.headers['Access-Control-Allow-Origin'] = '*'
	res.headers['Access-Control-Allow-Methods'] = 'POST,GET,OPTIONS'
	res.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
	return res

# 主函数
if __name__ == "__main__":
	print(("* 导入keras模型中..."
		"请等待模型导入完毕……"))
	load_model()
	app.run(host='114.64.249.179',port=5000)
