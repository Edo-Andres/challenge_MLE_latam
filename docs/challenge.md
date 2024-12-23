# Latam Software Engineer (ML & LLMs) Challenge

- Author: Eduardo Echeverría C.


## Part I

#### Notebook Exploration and Bug Fixing: `barplot` errors
Original version:
```python
sns.barplot(flights_by_airline.index, flights_by_airline.values, alpha=0.9)
```

Fixted version
```python
sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values, alpha=0.9)
```

### Model Selection and Final Decision

The selection of the final model was based on the results of the six trained models. After evaluating their metrics (precision, recall, F1-score, and accuracy), it was concluded that:

- XGBoost and Logistic Regression achieved the best and very similar results.
- Reducing the features to the top 10 most important did not degrade model performance.
- Balancing the classes significantly improved recall for the minority class ("1").

Considering these points, the **Logistic Regression model with the 10 most important features and class balancing (`reg_model_2`)** was chosen for its simplicity, computational efficiency, and comparable performance to XGBoost. This decision ensures an interpretable and robust solution for practical deployment.

### Saving the Final Model

The selected model, `reg_model_2`, was saved using `joblib`. The choice of `joblib` over `pickle` was due to its efficiency in handling large numerical arrays, which are commonly present in machine learning models. Below is the code used to save the model:

```python
# Save models reg_model_2 using joblib 
import joblib
joblib.dump(reg_model_2, 'reg_model_2.pkl')
```


## Part II

### Deploy the Model in an API (Framework FastAPI).

To deploy the `reg_model_2` Logistic Regression model. This API provides endpoints for health checks and predictions. The implementation is modular, ensuring clarity and scalability.

#### FastAPI Structure

1. **`api.py`**:
   - Initializes the FastAPI application.
   - Defines endpoints for health checks (`/health`) and predictions (`/predict`).
   - Uses the `DelayModel` class to load the trained model and preprocess inputs.

2. **`model.py`**:
   - Contains the `DelayModel` class for:
     - Preprocessing input data.
     - Loading the trained model using `joblib`.
     - Predicting delays for new inputs.

3. **API Endpoints**:
   - **`/health`**: Returns the API's health status (`status: OK`).
   - **`/predict`**: Accepts flight information (`OPERA`, `TIPOVUELO`, `MES`) and returns delay predictions.

    Example json:
   ```
   [
     {
       "OPERA": "LATAM",
       "TIPOVUELO": "Internacional",
       "MES": 5
     },
     {
       "OPERA": "Avianca",
       "TIPOVUELO": "Nacional",
       "MES": 12
     }
   ]
    ```
### Required Libraries

The following libraries are required to run the API:

- `FastAPI`: To create the RESTful API.
- `Pydantic`: For input data validation.
- `Uvicorn`: ASGI server to run the FastAPI application.
- `Pandas`: For handling and preprocessing tabular data.
- `Joblib`: For loading the trained Logistic Regression model.
- `Scikit-learn`: For the Logistic Regression model.

### Running the API

To run the API, use the following command:
```
uvicorn challenge.api:app --reload
```

## Part III

### Deploying the API in GCP with Cloud Run

The API was successfully deployed to Google Cloud Platform (GCP) using **Cloud Run**. This platform was chosen for its simplicity and ability to manage containerized applications without worrying about server infrastructure. The deployed API is accessible at the following URL:

**[API Documentation and Endpoints](https://challenge-api-64468935141.us-central1.run.app/docs)**

#### Steps Taken:

1. **Dockerization:**
   - A `Dockerfile` was created to containerize the FastAPI application. The container exposes the application on port `8080`, as required by Cloud Run.
   - Key commands in the `Dockerfile` include:
     ```dockerfile
     CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
     ```
   - The container was built and tested locally to ensure it functioned as expected.

2. **Uploading the Docker Image:**
   - The Docker image was tagged and pushed to **Google Artifact Registry**. This is a secure and scalable location for storing container images.
   - Commands used:
     ```bash
     docker tag challenge-api us-central1-docker.pkg.dev/andresec/challenge-repo/challenge-api:latest
     docker push us-central1-docker.pkg.dev/andresec/challenge-repo/challenge-api:latest
     ```

3. **Deploying with Cloud Run:**
   - The API was deployed to Cloud Run using the `gcloud` CLI. The service was configured to allow unauthenticated access and allocated 512 MiB of memory.
   - Deployment command:
     ```bash
     gcloud run deploy challenge-api \
         --image us-central1-docker.pkg.dev/andresec/challenge-repo/challenge-api:latest \
         --platform managed \
         --region us-central1 \
         --allow-unauthenticated \
         --memory 512Mi
     ```

4. **Testing and Verification:**
   - After deployment, the `/health` endpoint was accessed to confirm that the API was running correctly.
   - API documentation is automatically generated by FastAPI and can be accessed at the `/docs` endpoint.

#### Challenges and Solutions:

- **Port Configuration:**
  - Cloud Run requires applications to listen on port `8080`. The `Dockerfile` was updated to ensure this configuration.

- **Image Size:**
  - The size of the Docker image was reduced by using a slim base image and removing unnecessary dependencies.

This deployment ensures a scalable, secure, and easily accessible API for predicting flight delays.

