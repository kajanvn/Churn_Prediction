**Churn_Prediction**
    This project contains both dev and app modules

**📊 Churn Prediction App**
        A machine learning project to predict customer churn using a trained model served through a FastAPI backend and a user-friendly HTML interface. The project is containerized using Docker for easy deployment.

**🔧 Features**

FastAPI backend for serving predictions.

Simple HTML UI for user input.

Docker support for deployment.

Model files included for inference.

**Docker image**
      docker run -d -p 8000:8000 churn-predictor

**UI**: http://localhost:8000/

**Swagger**: http://localhost:8000/docs

**📁 Project Structure**

churn_project/

├── churn_app/ # FastAPI app, UI, model files

├── churn_dev/ # training code
