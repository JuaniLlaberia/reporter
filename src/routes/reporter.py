from flask import Blueprint, request, jsonify

reporter_bp = Blueprint("reporter", __name__)

@reporter_bp.route("/generate-report", methods=["POST"])
def generate_report():
    pass