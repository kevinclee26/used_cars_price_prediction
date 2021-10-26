from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np
from sqlalchemy import create_engine

with open('assets/ohe.sav', 'rb') as f: 
	ohe=pickle.load(f)
with open('assets/rfr.sav', 'rb') as f: 
	rfr=pickle.load(f)

engine=create_engine('postgresql://postgres:postgres@127.0.0.1:5432/used_cars')

app=Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/sample')
def sample_predict(): 
	# take some input (Python)
	sample_input=['audi', 100000]
	# break into cat and numerical columns
	sample_cat=sample_input[0]
	sample_num=sample_input[1]
	# transform input
	transformed_cat=ohe.transform([[sample_cat]]).toarray()
	transformed_input=np.concatenate([transformed_cat[0], np.array([sample_num])])

	# use transformed input to predict
	# [0, 0, 0, 1, 0, etc. , 100000]
	prediction=rfr.predict([transformed_input])

	return jsonify({'output': prediction[0]})

@app.route('/maker_list')
def maker_list(): 
	result=engine.execute('select distinct maker from used_cars').fetchall()
	return_list=[each_result[0] for each_result in result]
	return jsonify(return_list)

@app.route('/data')
def get_data(): 
	result=engine.execute('select * from used_cars')
	return_list=[dict(each_result) for each_result in result]
	return jsonify(return_list)

@app.route('/predict/')
@app.route('/predict')
def predict(): 
	maker=request.args.get('maker')
	mileage=request.args.get('mileage')
	transformed_maker=ohe.transform([[maker]]).toarray()
	transformed_input=np.concatenate([transformed_maker[0], np.array([mileage])])

	prediction=rfr.predict([transformed_input])
	return jsonify({'output': round(prediction[0], 2)})

@app.route('/predict_price/')
@app.route('/predict_price')
def predict_price(): 
	maker=request.args.get('maker')
	mileage=request.args.get('mileage')
	transformed_maker=ohe.transform([[maker]]).toarray()
	transformed_input=np.concatenate([transformed_maker[0], np.array([mileage])])

	prediction=rfr.predict([transformed_input])
	# return jsonify({'output': round(prediction[0], 2)})
	return render_template('predict_price.html', result=round(prediction[0], 2))
	# return jsonify({'maker': maker, 'mileage': mileage})

if __name__=='__main__': 
	app.run()