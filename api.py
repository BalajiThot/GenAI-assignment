from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
from typing import Optional
import docx
import io
import uuid
import zipfile
from pathlib import Path
import logging

from langgraph_workflow import process_srs_document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SRS to FastAPI Generator",
    description="A service that converts Software Requirements Specification documents into FastAPI projects",
    version="1.0.0"
)

# Store generated projects (in a real app, use a database)
generated_projects = {}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SRS to FastAPI Generator API",
        "endpoints": {
            "/process-srs": "Upload SRS document to generate FastAPI project",
            "/projects/{project_id}": "Get status of a generated project"
        }
    }

@app.post("/process-srs")
async def process_srs(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Process an SRS document and generate a FastAPI project.
    
    - **file**: A .docx SRS document
    """
    # Validate file type
    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")
    
    try:
        # Generate a unique ID for this project
        project_id = str(uuid.uuid4())
        
        # Read the file content
        contents = await file.read()
        doc = docx.Document(io.BytesIO(contents))
        
        # Extract text from the document
        srs_content = ""
        for paragraph in doc.paragraphs:
            srs_content += paragraph.text + "\n"
        
        # Process in the background to avoid timeout
        background_tasks.add_task(
            process_srs_background_task,
            project_id=project_id,
            srs_content=srs_content
        )
        
        # Return the project ID for status checking
        return {
            "message": "SRS document submitted for processing",
            "project_id": project_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error processing SRS: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing SRS: {str(e)}")

@app.get("/projects/{project_id}")
async def get_project(project_id: str):
    """
    Get the status of a generated project.
    
    - **project_id**: The ID of the project
    """
    if project_id not in generated_projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return generated_projects[project_id]

def process_srs_background_task(project_id: str, srs_content: str):
    """Background task to process the SRS document."""
    try:
        # Process the SRS document
        result = process_srs_document(srs_content)
        
        # Create a temporary directory to store the generated files
        project_dir = tempfile.mkdtemp(prefix=f"project_{project_id}_")
        
        # Write the generated files
        write_generated_files(project_dir, result['generated_code'])
        
        # Create a zip file
        zip_path = create_zip_file(project_dir, project_id)
        
        # Update project status
        generated_projects[project_id] = {
            "status": "completed",
            "project_id": project_id,
            "zip_path": zip_path,
            "langsmith_logs": f"https://smith.langchain.com/project/srs_generator_project/runs/{project_id}"
        }
        
    except Exception as e:
        logger.error(f"Error in background task: {str(e)}")
        generated_projects[project_id] = {
            "status": "failed",
            "project_id": project_id,
            "error": str(e)
        }

def write_generated_files(base_dir: str, generated_code: dict):
    """Write the generated code to files."""
    # Create directory structure
    os.makedirs(os.path.join(base_dir, "app", "api", "routes"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "app", "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "app", "services"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "tests"), exist_ok=True)
    
    # Write models
    if 'models' in generated_code:
        with open(os.path.join(base_dir, "app", "models", "models.py"), "w") as f:
            f.write(generated_code['models'])
    
    # Write database.py
    if 'database' in generated_code:
        with open(os.path.join(base_dir, "app", "database.py"), "w") as f:
            f.write(generated_code['database'])
    
    # Write route files
    for key, code in generated_code.items():
        if key.startswith('route_'):
            route_name = key.replace('route_', '')
            with open(os.path.join(base_dir, "app", "api", "routes", f"{route_name}.py"), "w") as f:
                f.write(code)
    
    # Write main.py
    if 'main' in generated_code:
        with open(os.path.join(base_dir, "app", "main.py"), "w") as f:
            f.write(generated_code['main'])
    
    # Create __init__.py files
    with open(os.path.join(base_dir, "app", "api", "routes", "__init__.py"), "w") as f:
        f.write("# Routes initialization")
    
    with open(os.path.join(base_dir, "app", "models", "__init__.py"), "w") as f:
        f.write("# Models initialization")
    
    # Create a requirements.txt file
    with open(os.path.join(base_dir, "requirements.txt"), "w") as f:
        f.write("fastapi>=0.68.0\nuvicorn>=0.15.0\nsqlalchemy>=1.4.23\npsycopg2-binary>=2.9.1\npydantic>=1.8.2\npython-multipart>=0.0.5\nalembic>=1.7.4\npython-dotenv>=0.19.0\n")
    
    # Create a README.md file
    with open(os.path.join(base_dir, "README.md"), "w") as f:
        f.write("# Generated FastAPI Project\n\n")
        f.write("This project was automatically generated from an SRS document.\n\n")
        f.write("## Setup\n\n")
        f.write("1. Create a virtual environment: `python -m venv venv`\n")
        f.write("2. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\\Scripts\\activate` (Windows)\n")
        f.write("3. Install dependencies: `pip install -r requirements.txt`\n")
        f.write("4. Set up the database: Update the `.env` file with your database credentials\n")
        f.write("5. Run migrations: `alembic upgrade head`\n")
        f.write("6. Start the server: `uvicorn app.main:app --reload`\n")
    
    # Create a .env file
    with open(os.path.join(base_dir, ".env"), "w") as f:
        f.write("DATABASE_URL=postgresql://username:password@localhost:5432/fastapi_db\n")
        f.write("DEBUG=True\n")
    
    # Create a Dockerfile
    with open(os.path.join(base_dir, "Dockerfile"), "w") as f:
        f.write("FROM python:3.9\n\n")
        f.write("WORKDIR /app\n\n")
        f.write("COPY requirements.txt .\n")
        f.write("RUN pip install --no-cache-dir -r requirements.txt\n\n")
        f.write("COPY . .\n\n")
        f.write("CMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n")

def create_zip_file(directory: str, project_id: str) -> str:
    """Create a zip file of the generated project."""
    zip_path = f"project_{project_id}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory)
                zipf.write(file_path, arcname)
    
    return zip_path

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)