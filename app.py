from flask import render_template,Flask,request,Response
from prometheus_client import Counter,generate_latest
from flipkart.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder

from dotenv import load_dotenv
load_dotenv()

# This metric counts the total number of HTTP requests received by the Flask application.
# It is a Prometheus Counter named "http_requests_total" with a description "Total HTTP Request".
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Request")

def create_app():

    app = Flask(__name__)

    vector_store = DataIngestor().ingest(load_existing=True)
    rag_chain = RAGChainBuilder(vector_store).build_chain()

    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")
    
    @app.route("/get" , methods=["POST"])
    def get_response():

        user_input = request.form["msg"]

        reponse = rag_chain.invoke(
            {"input" : user_input},
            config={"configurable" : {"session_id" : "user-session"}}
        )["answer"]

        return reponse
    
    @app.route("/metrics")
    def metrics():
        # The /metrics endpoint exposes Prometheus metrics for the Flask application.
        # By default, the prometheus_client library provides several in-built metrics:
        # - process_cpu_seconds_total: Total user and system CPU time spent in seconds.
        # - process_resident_memory_bytes: Resident memory size in bytes.
        # - process_virtual_memory_bytes: Virtual memory size in bytes.
        # - process_start_time_seconds: Start time of the process since Unix epoch in seconds.
        # - python_gc_objects_collected_total: Number of objects collected during GC, per generation.
        # - python_gc_objects_uncollectable_total: Number of uncollectable objects found during GC, per generation.
        # - python_info: Python runtime information (version, implementation, etc).
        # - http_requests_total: (Custom) Total HTTP requests received by the Flask app.
        # These metrics help monitor resource usage, application health, and request rates.
        return Response(generate_latest(), mimetype="text/plain")
    
    return app

if __name__=="__main__":
    app = create_app()
    app.run(host="0.0.0.0",port=5001,debug=True)