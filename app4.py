from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import scale
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('rsi.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Read the uploaded file
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Get number of components from form data
        n_components = int(request.form.get('components', 2))
        # Get number of neighbors from form data
        n_neighbors = int(request.form.get('neighbors', 5))

        # Validate number of components 
        max_components = X.shape[1] - 1
        n_components = min(n_components, max_components)

        def kpcaalgorithm(a1, n_components):
            scaled = scale(a1, with_mean=True, with_std=False)
            model = KernelPCA(n_components=n_components, kernel='linear')
            model.fit(a1)
            transformed = model.transform(a1)
            return transformed

        col = len(df.columns)
        halfdf = col // 2

        df1 = df.iloc[:, :halfdf]
        df2 = df.iloc[:, halfdf:]

        df1res = kpcaalgorithm(df1, n_components)
        res1 = np.array(df1res)

        df2res = kpcaalgorithm(df2, n_components)
        res2 = np.array(df2res)

        pcares = np.concatenate((res1, res2), axis=1)

        df = pd.DataFrame(pcares)
        dfres3 = kpcaalgorithm(df, n_components)

        subpcares = np.array(dfres3)

        x_train_w, x_test_w, y_train_w, y_test_w = train_test_split(subpcares, y, test_size=0.3, random_state=0)

        clr_k = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', algorithm='auto')
        clr_k.fit(x_train_w, y_train_w)
        y_pred_k = clr_k.predict(x_test_w)
        acc_k = accuracy_score(y_test_w, y_pred_k)

        os.remove(file_path)

        return render_template('rsi.html', row=(acc_k*100))

    else:
        os.remove(file_path)
        return 'Uploaded file format not supported. Please upload a CSV file.'

if __name__ == '__main__':
    app.run(debug=True)
