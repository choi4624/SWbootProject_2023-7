from fastapi import FastAPI
from routes import app

if __name__ == "__main__":
    app.run(debug=True)
