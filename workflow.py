import os
from typing import Dict, TypedDict, List, Any, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
import json
import re
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Configure language models
openai_model = ChatOpenAI(model="gpt-4", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
llama_model = ChatGroq(model="llama3-70b-8192", temperature=0.1, api_key=os.getenv("GROQ_API_KEY"))

# Define the state schema
class GraphState(TypedDict):
    srs_content: str
    requirements: Dict[str, Any]
    database_schema: Dict[str, Any]
    endpoints: List[Dict[str, Any]]
    test_cases: Dict[str, List[str]]
    generated_code: Dict[str, str]
    execution_results: Dict[str, Any]
    errors: List[str]

# Define output models for structured parsing
class DatabaseSchema(BaseModel):
    tables: List[Dict[str, Any]] = Field(description="List of database tables with their columns, types, and relationships")

class ApiEndpoint(BaseModel):
    path: str = Field(description="API endpoint path")
    method: str = Field(description="HTTP method (GET, POST, PUT, DELETE)")
    parameters: List[Dict[str, str]] = Field(description="List of parameters with name, type, and description")
    response: Dict[str, Any] = Field(description="Expected response structure")
    description: str = Field(description="Description of the endpoint functionality")

class Requirements(BaseModel):
    functional: List[str] = Field(description="List of functional requirements")
    authentication: Dict[str, Any] = Field(description="Authentication requirements")
    database: Dict[str, Any] = Field(description="Database requirements")
    business_logic: List[str] = Field(description="Business logic requirements")

# Node 1: Parse SRS Document
def parse_srs(state: GraphState) -> GraphState:
    """Extract key requirements from the SRS document."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert system analyst who extracts key requirements from SRS documents.
        Extract the following information:
        1. Functional requirements
        2. Authentication & authorization requirements
        3. Database requirements
        4. Business logic requirements
        
        Format the output as a JSON object with these categories."""),
        HumanMessage(content=f"Here is the SRS document:\n\n{state['srs_content']}")
    ])
    
    response = openai_model.invoke(prompt)
    
    # Extract JSON from response
    json_match = re.search(r'```json\n(.*?)```', response.content, re.DOTALL)
    if json_match:
        requirements_json = json_match.group(1)
    else:
        requirements_json = response.content
    
    try:
        requirements = json.loads(requirements_json)
    except json.JSONDecodeError:
        # Fallback parsing if JSON is malformed
        requirements = {
            "functional": [],
            "authentication": {},
            "database": {},
            "business_logic": []
        }
        
    # Update state
    state['requirements'] = requirements
    return state

# Node 2: Extract Database Schema
def extract_database_schema(state: GraphState) -> GraphState:
    """Extract database schema from the requirements."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a database architect. 
        Based on the provided requirements, define a complete database schema with tables, columns, relationships, and constraints.
        Format your output as JSON with a 'tables' key containing an array of table definitions."""),
        HumanMessage(content=f"Requirements: {json.dumps(state['requirements'])}")
    ])
    
    response = llama_model.invoke(prompt)
    
    # Extract JSON from response
    json_match = re.search(r'```json\n(.*?)```', response.content, re.DOTALL)
    if json_match:
        schema_json = json_match.group(1)
    else:
        schema_json = response.content
    
    try:
        schema = json.loads(schema_json)
    except json.JSONDecodeError:
        # Fallback parsing if JSON is malformed
        schema = {"tables": []}
    
    # Update state
    state['database_schema'] = schema
    return state

# Node 3: Define API Endpoints
def define_api_endpoints(state: GraphState) -> GraphState:
    """Define API endpoints based on requirements and database schema."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a FastAPI expert.
        Based on the provided requirements and database schema, define all necessary API endpoints.
        For each endpoint, specify:
        1. Path
        2. HTTP method
        3. Request parameters
        4. Response structure
        5. Description
        
        Format your output as JSON with an 'endpoints' key containing an array of endpoint definitions."""),
        HumanMessage(content=f"""
        Requirements: {json.dumps(state['requirements'])}
        Database Schema: {json.dumps(state['database_schema'])}
        """)
    ])
    
    response = openai_model.invoke(prompt)
    
    # Extract JSON from response
    json_match = re.search(r'```json\n(.*?)```', response.content, re.DOTALL)
    if json_match:
        endpoints_json = json_match.group(1)
    else:
        endpoints_json = response.content
    
    try:
        endpoints_data = json.loads(endpoints_json)
        if 'endpoints' in endpoints_data:
            endpoints = endpoints_data['endpoints']
        else:
            endpoints = endpoints_data
    except json.JSONDecodeError:
        # Fallback parsing if JSON is malformed
        endpoints = []
    
    # Update state
    state['endpoints'] = endpoints
    return state

# Node 4: Generate Test Cases
def generate_test_cases(state: GraphState) -> GraphState:
    """Generate test cases for the API endpoints."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a test engineer specializing in FastAPI applications.
        Write pytest test cases for each API endpoint following Test-Driven Development principles.
        Include tests for:
        1. Success cases
        2. Error handling
        3. Edge cases
        4. Input validation
        
        Format your output as JSON with endpoint paths as keys and arrays of test functions as values."""),
        HumanMessage(content=f"""
        API Endpoints: {json.dumps(state['endpoints'])}
        Database Schema: {json.dumps(state['database_schema'])}
        """)
    ])
    
    response = llama_model.invoke(prompt)
    
    # Extract JSON from response
    json_match = re.search(r'```json\n(.*?)```', response.content, re.DOTALL)
    if json_match:
        tests_json = json_match.group(1)
    else:
        tests_json = response.content
    
    try:
        test_cases = json.loads(tests_json)
    except json.JSONDecodeError:
        # Fallback parsing if JSON is malformed
        test_cases = {}
    
    # Update state
    state['test_cases'] = test_cases
    return state

# Node 5: Generate Code
def generate_code(state: GraphState) -> GraphState:
    """Generate the FastAPI application code."""
    generated_code = {}
    
    # Generate database models
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a FastAPI developer specializing in SQLAlchemy models.
        Generate SQLAlchemy model classes for the given database schema.
        Include proper relationships, constraints, and docstrings.
        Format your output as pure Python code without any markdown or code blocks."""),
        HumanMessage(content=f"Database Schema: {json.dumps(state['database_schema'])}")
    ])
    
    response = openai_model.invoke(prompt)
    generated_code['models'] = extract_code(response.content)
    
    # Generate database connection
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a FastAPI developer.
        Generate database.py file with SQLAlchemy connection configuration.
        Include proper connection pooling, session management, and Base declarative model.
        Format your output as pure Python code without any markdown or code blocks.""")
    ])
    
    response = openai_model.invoke(prompt)
    generated_code['database'] = extract_code(response.content)
    
    # Generate API routes
    for endpoint in state['endpoints']:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a FastAPI developer.
            Generate a FastAPI route file for the specified endpoint.
            Include proper input validation, error handling, documentation, and database interactions.
            Format your output as pure Python code without any markdown or code blocks."""),
            HumanMessage(content=f"""
            Endpoint: {json.dumps(endpoint)}
            Database Models: {generated_code['models']}
            """)
        ])
        
        response = openai_model.invoke(prompt)
        endpoint_path = endpoint['path'].strip('/').replace('/', '_')
        generated_code[f'route_{endpoint_path}'] = extract_code(response.content)
    
    # Generate main.py
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a FastAPI developer.
        Generate the main.py file that initializes the FastAPI application.
        Include proper middleware, CORS configuration, and route imports.
        Format your output as pure Python code without any markdown or code blocks."""),
        HumanMessage(content=f"API Endpoints: {json.dumps(state['endpoints'])}")
    ])
    
    response = openai_model.invoke(prompt)
    generated_code['main'] = extract_code(response.content)
    
    # Update state
    state['generated_code'] = generated_code
    return state

# Node 6: Validate Code
def validate_code(state: GraphState) -> Dict[str, Any]:
    """Validate the generated code for errors and consistency."""
    errors = []
    
    # Simple validation (in a real system, you would compile and run tests)
    for file_name, code in state['generated_code'].items():
        if not code.strip():
            errors.append(f"Empty code for {file_name}")
        
        # Check for import errors (simplified)
        if "import" not in code:
            errors.append(f"Missing imports in {file_name}")
    
    # Update state
    state['errors'] = errors
    
    # Determine next node based on errors
    if errors:
        return {"next": "fix_code"}
    else:
        return {"next": "write_files"}

# Node 7: Fix Code
def fix_code(state: GraphState) -> GraphState:
    """Fix code issues identified during validation."""
    updated_code = {}
    
    for file_name, code in state['generated_code'].items():
        # Check if there are specific errors for this file
        file_errors = [error for error in state['errors'] if file_name in error]
        
        if file_errors:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a FastAPI debugging expert.
                Fix the following code based on the reported errors.
                Format your output as pure Python code without any markdown or code blocks."""),
                HumanMessage(content=f"""
                Code: {code}
                
                Errors: {file_errors}
                """)
            ])
            
            response = openai_model.invoke(prompt)
            updated_code[file_name] = extract_code(response.content)
        else:
            updated_code[file_name] = code
    
    # Update state
    state['generated_code'] = updated_code
    state['errors'] = [] # Clear errors since they've been addressed
    return state

# Node 8: Write Files
def write_files(state: GraphState) -> GraphState:
    """Write the generated code to files."""
    # In a real implementation, this would actually write to files
    # For this example, we'll just simulate it
    execution_results = {"status": "success", "files_written": []}
    
    # Models
    execution_results["files_written"].append("app/models/models.py")
    
    # Database
    execution_results["files_written"].append("app/database.py")
    
    # Routes
    for endpoint in state['endpoints']:
        endpoint_path = endpoint['path'].strip('/').replace('/', '_')
        execution_results["files_written"].append(f"app/api/routes/{endpoint_path}.py")
    
    # Main
    execution_results["files_written"].append("app/main.py")
    
    # Update state
    state['execution_results'] = execution_results
    return state

# Helper function to extract code from response
def extract_code(response_text):
    """Extract code from markdown code blocks or return the text as is."""
    code_match = re.search(r'```(?:python)?\n(.*?)```', response_text, re.DOTALL)
    if code_match:
        return code_match.group(1)
    return response_text

# Build the LangGraph workflow
def build_workflow():
    # Create a new graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("parse_srs", parse_srs)
    workflow.add_node("extract_database_schema", extract_database_schema)
    workflow.add_node("define_api_endpoints", define_api_endpoints)
    workflow.add_node("generate_test_cases", generate_test_cases)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("validate_code", validate_code)
    workflow.add_node("fix_code", fix_code)
    workflow.add_node("write_files", write_files)
    
    # Add edges
    workflow.add_edge("parse_srs", "extract_database_schema")
    workflow.add_edge("extract_database_schema", "define_api_endpoints")
    workflow.add_edge("define_api_endpoints", "generate_test_cases")
    workflow.add_edge("generate_test_cases", "generate_code")
    workflow.add_edge("generate_code", "validate_code")
    workflow.add_conditional_edges(
        "validate_code",
        lambda x: x["next"],
        {
            "fix_code": "fix_code",
            "write_files": "write_files"
        }
    )
    workflow.add_edge("fix_code", "validate_code")
    workflow.add_edge("write_files", END)
    
    # Set the entry point
    workflow.set_entry_point("parse_srs")
    
    # Compile the graph
    return workflow.compile()

# Initialize the workflow
workflow = build_workflow()

# Function to process an SRS document
def process_srs_document(srs_content):
    """Process an SRS document through the LangGraph workflow."""
    # Initialize state
    initial_state = {
        "srs_content": srs_content,
        "requirements": {},
        "database_schema": {},
        "endpoints": [],
        "test_cases": {},
        "generated_code": {},
        "execution_results": {},
        "errors": []
    }
    
    # Create a memory saver for checkpointing
    memory = MemorySaver()
    
    # Run the workflow with checkpointing
    for event in workflow.stream(initial_state, config={"checkpointer": memory}):
        if event["type"] == "end":
            return event["state"]