from flask import Blueprint, request, jsonify
from src.agents.orchestrator.orchestrator import Orchestrator

reporter_bp = Blueprint("reporter", __name__)

@reporter_bp.route("/generate-report", methods=["POST"])
def generate_report():
    """
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    request_json = request.get_json()
    if "prompt" not in request_json:
        return jsonify({"error": "Missing prompt"}), 400

    orchestrator = Orchestrator(
        ollama_model="gemma3:4b",
        ollama_base_url="http://localhost:11434"
    )

    prompt = request_json["prompt"]
    orchestrator.run(prompt)

    return jsonify({"message": "Report was successfully created"}), 200