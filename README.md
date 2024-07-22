# Face and ID Card Processing APIs

This Django REST API module provides various endpoints for face and ID card processing. It includes features for face registration, face matching, image retrieval, entry deletion, and ID number extraction from images.

## Features

- **Face Registration**: Register faces and save them to the database.
- **Face Matching**: Match uploaded faces against registered faces.
- **Image Retrieval**: Retrieve registered face images.
- **Entry Deletion**: Delete registered face entries.
- **ID Number Extraction**: Extract ID numbers from uploaded ID card images.

## Prerequisites

- Docker
- Docker Compose

## Getting Started

### Clone the Repository

```sh
git clone https://github.com/your-repo/your-project.git
cd your-project
```

### Setting Up Environment Variables

Create a `.env` file in the project root directory and configure the following environment variables:

```env
SECRET_KEY=your_secret_key
DEBUG=True
ALLOWED_HOSTS=*

DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=db
DB_PORT=5432
```

### Docker Setup

Make sure you have Docker and Docker Compose installed on your machine. Then, follow these steps to build and run the Docker containers:

1. **Build the Docker containers**

```sh
docker-compose build
```

2. **Run the Docker containers**

```sh
docker-compose up
```

This will start the Django application and PostgreSQL database in separate containers.

### Apply Migrations

After starting the containers, you need to apply the database migrations:

```sh
docker-compose exec web python manage.py migrate
```

### Access the Application

Open your web browser and go to `http://localhost:8000` to access the application.

## API Endpoints

### Face Registration

**URL**: `/face_registration/`  
**Method**: `POST`  
**Description**: Register a face image.

**Request**:

```json
{
    "image": "path/to/your/image.jpg"
}
```

**Response**:

```json
{
    "status": true,
    "message": "Face registered successfully.",
    "match": "matched_face_id",
    "score": 0.95,
    "image_url": "http://localhost:8000/media/faces/matched_face_id.jpg"
}
```

### Face Matching

**URL**: `/face_match/`  
**Method**: `POST`  
**Description**: Match an uploaded face image against registered faces.

**Request**:

```json
{
    "image": "path/to/your/image.jpg"
}
```

**Response**:

```json
{
    "status": true,
    "message": "Face matched successfully.",
    "match": "matched_face_id",
    "score": 0.95,
    "image_url": "http://localhost:8000/media/faces/matched_face_id.jpg"
}
```

### Image Retrieval

**URL**: `/view_image/<face_id>/`  
**Method**: `GET`  
**Description**: Retrieve the registered face image associated with the given `face_id`.

**Response**:

```json
{
    "status": true,
    "image_URL": "http://localhost:8000/media/faces/<face_id>.jpg",
    "error": null
}
```

### Entry Deletion

**URL**: `/delete_entry/<face_id>/`  
**Method**: `POST`  
**Description**: Delete the registered face entry associated with the given `face_id`.

**Response**:

```json
{
    "status": true,
    "message": "Image and entry deleted successfully."
}
```

### ID Number Extraction

**URL**: `/id_image/`  
**Method**: `POST`  
**Description**: Process the uploaded ID image to detect a card and extract the ID number.

**Request**:

```json
{
    "image": "path/to/your/id_image.jpg"
}
```

**Response**:

```json
{
    "status": true,
    "ID": "extracted_id_number",
    "card_URL": "http://localhost:8000/media/id_card/timestamp.jpg",
    "error": null
}
```

## Directory Structure

```plaintext
.
├── core
│   ├── model_files
│   │   └── best.tflite
│   ├── modules
│   │   ├── preprocessing.py
│   │   └── __init__.py
│   ├── serializers.py
│   ├── views.py
│   └── ...
├── Dockerfile
├── docker-compose.yml
├── manage.py
└── ...
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

This README file provides a comprehensive overview of your Django REST API, including setup instructions, API endpoints, and other essential information.
# face-server
