# Bee classifier

With this little **streamlit app** I want to demonstrate the potential of **convolutional neural networks** (CNNs) for image classification. 
It makes use of [Kaggle's BeeImage Dataset](https://www.kaggle.com/jenny18/honey-bee-annotated-images).

Note that several parts of the app were taken from elsewhere:
- SessionState.py adds per-session state to Streamlit and was taken from [here](https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92)
- Bee images (except the ones from the BeeImage Dataset) were taken from Wikipedia (links at the bottom of the app)

You can check out the app in action on [Streamlit Sharing](https://share.streamlit.io/jochen-eds/bee-app/main/bee_app.py).

To run the app on your local, navigate into your folder and run the following code in your command line tool:  
`conda create --name myenv --file requirements.txt python=3.7`  
`conda activate myenv`  
`streamlit run bee_app.py`
