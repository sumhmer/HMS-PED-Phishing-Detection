import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(ROOT / ".env")

from routes.hybrid import hybrid_bp  # noqa: E402
from routes.llm import llm_bp  # noqa: E402

app = Flask(__name__)
app.register_blueprint(llm_bp)
app.register_blueprint(hybrid_bp)
CORS(app)

if __name__ == "__main__":
    app.run(port=5000, debug=False)
