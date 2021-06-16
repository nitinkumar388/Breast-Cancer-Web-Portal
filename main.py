from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

try:
    file = open('model.pkl','rb')
    clf = pickle.load(file)
except EOFError:
    data = list()





@app.route('/', methods = ["GET", "POST"])
def cancer():
    if request.method == "POST":
        myDict = request.form
        mean_radius = (myDict['mean_radius'])
        mean_texture = (myDict['mean_texture'])
        mean_perimeter = (myDict['mean_perimeter'])
        mean_area = (myDict['mean_area'])
        mean_smoothness = (myDict['mean_smoothness'])
        mean_compactness = (myDict['mean_compactness'])
        mean_concavity = (myDict['mean_concavity'])
        mean_concave_points = (myDict['mean_concave_points'])
        mean_symmetry = (myDict['mean_symmetry'])
        mean_fractal_dimension = (myDict['mean_fractal_dimension'])
        radius_error = (myDict['radius_error'])
        texture_error = (myDict['texture_error'])
        perimeter_error = (myDict['perimeter_error'])
        area_error = (myDict['area_error'])
        smoothness_error = (myDict['smoothness_error'])
        compactness_error = (myDict['compactness_error'])
        cancavity_error = (myDict['cancavity_error'])
        concave_points_error = (myDict['concave_points_error'])
        symmetry_error = (myDict['symmetry_error'])
        fractal_dimension_error = (myDict['fractal_dimension_error'])
        worst_radius = (myDict['worst_radius'])
        worst_texture = (myDict['worst_texture'])
        worst_perimeter = (myDict['worst_perimeter'])
        worst_area = (myDict['worst_area'])
        worst_smoothness = (myDict['worst_smoothness'])
        worst_compactness = (myDict['worst_compactness'])
        worst_concavity = (myDict['worst_concavity'])
        worst_concave_points = (myDict['worst_concave_points'])
        worst_symmetry = (myDict['worst_symmetry'])
        worst_fractal_dimension = (myDict['worst_fractal_dimension']) 
        
        inputFeatures = [mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,area_error,smoothness_error,compactness_error,cancavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        #infProb1 = clf.predict([inputFeatures])[0]
        #print(infProb)
        #return 'nkb' +str(infProb*100) #+'or' + str(infProb1)
        #return render_template('index.html')
        return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')
    
    

    




if __name__ == '__main__':
    app.run(debug=True)
    