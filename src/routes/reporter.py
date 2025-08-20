import io
from flask import Blueprint, request, jsonify, send_file
from src.agents.orchestrator.orchestrator import Orchestrator

reporter_bp = Blueprint("reporter", __name__)

@reporter_bp.route("/generate-report", methods=["POST"])
def generate_report():
    """
    Route to generate report
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
    report_bytes = orchestrator.run(prompt)

    return send_file(
        io.BytesIO(report_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="report.pdf"
    )