from flask import Flask, request, jsonify
import csv
import random
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model (Pickle file)
model = pickle.load(open('my_model.pkl', 'rb'))
scale = pickle.load(open('scaler.pkl', 'rb'))

# Load CSV data into a list of dictionaries
with open('data_test.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

@app.route('/')
def home():
    return "Welcome to API!"

@app.route('/load_data', methods=['GET'])
def load_data():
    # Get a random row from the data
    selected_row = random.choice(rows)
    # Return the random row data as JSON
    return jsonify({'col1':selected_row[0],'col2':selected_row[1],'col3':selected_row[2],
                     'col4':selected_row[3], 'col5':selected_row[4],'col6':selected_row[5],
                     'col7':selected_row[6], 'col8':selected_row[7],'col9':selected_row[8],
                      'col10':selected_row[9], 'col11':selected_row[10],'col12':selected_row[11],
                      'col13':selected_row[12], 'col14':selected_row[13],'col15':selected_row[14],
                      'col16':selected_row[15], 'col17':selected_row[16],'col18':selected_row[17],
                       'col19':selected_row[18], 'col20':selected_row[19],'col21':selected_row[20],
                       'col22':selected_row[21], 'col23':selected_row[22],'col24':selected_row[23],'col25':selected_row[24]})

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request JSON
    data = request.json
    feature1 = data['input1']
    feature2 = data['input2']
    feature3 = data['input3']
    feature4 = data['input4']
    feature5 = data['input5']
    feature6 = data['input6']
    feature7 = data['input7']
    feature8 = data['input8']
    feature9 = data['input9']
    feature10 = data['input10']
    feature11 = data['input11']
    feature12 = data['input12']
    feature13 = data['input13']
    feature14 = data['input14']
    feature15 = data['input15']
    feature16 = data['input16']
    feature17 = data['input17']
    feature18 = data['input18']
    feature19 = data['input19']
    feature20 = data['input20']
    feature21 = data['input21']
    feature22 = data['input22']
    feature23 = data['input23']
    feature24 = data['input24']

    sc = [[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11,
         feature12,feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20, feature21,
         feature22,feature23, feature24]]
    row = np.array(sc).reshape(1,-1)
    features = scale.transform(row.reshape(1,-1))
    prediction = model.predict(features)

    if prediction == 1:
        return jsonify({'prediction_text': 'Cancer'})
    else:
        return jsonify({'prediction_text': 'Normal'})

if __name__ == '__main__':
    app.run(debug=True)
