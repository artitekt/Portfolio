# Real-Time AI Data Pipeline

A professional demonstration of a real-time AI processing pipeline using event-driven architecture with async Python.

## Project Overview

This project showcases a production-ready real-time AI data pipeline that processes streaming events through multiple stages: data ingestion, feature processing, AI inference, and result publishing. The architecture is designed for high-throughput, low-latency applications and demonstrates modern Python async programming patterns.

The pipeline processes events from various sources (sensors, user actions, system metrics), extracts meaningful features, runs AI inference, and publishes results in real-time.

## Architecture Overview

The pipeline follows a clean event-driven architecture:

```
Data Source → Event Stream → Processing Engine → AI Inference → Result Publisher
```

### Architecture Diagram

```
+------------------+       +------------------+       +------------------+
|   Data Source    |──────>|    Event Bus     |──────>| Feature Processor|
|  (Simulated)     |       |  (Async Queue)   |       | (Statistical)    |
+------------------+       +--------+---------+       +--------+---------+
                                    |                       |
                                    v                       v
                            +------------------+       +------------------+
                            |   AI Inference   |<──────|  Feature Store   |
                            | (Mock/ONNX/etc)  |       | (Windowed Data)  |
                            +--------+---------+       +------------------+
                                     |
                                     v
                            +------------------+
                            | Result Publisher |
                            | (Console/File)   |
                            +------------------+
```

### Pipeline Components

- **Data Source**: Simulated real-time event generation (sensors, user actions, system events)
- **Event Bus**: Async message bus with topic-based routing and queue management
- **Feature Processor**: Statistical and temporal feature extraction with sliding windows
- **AI Inference**: Pluggable inference engines (Mock, Scikit-learn, ONNX, TensorFlow, PyTorch)
- **Result Publisher**: Multi-channel output (Console, File, Webhook, Metrics)

## Key Features

- **Async Event-Driven Pipeline**: Built on asyncio for high-throughput processing
- **Modular Architecture**: Pluggable components for easy extension and testing
- **Real-Time Processing**: Sub-millisecond latency with efficient memory management
- **Multiple AI Frameworks**: Support for Mock, Scikit-learn, ONNX, TensorFlow, and PyTorch models
- **Performance Monitoring**: Built-in metrics tracking and health checks
- **Scalable Design**: Batch processing, backpressure handling, and horizontal scaling ready
- **Production Ready**: Configuration management, error handling, and logging

## Running the Demo

### Installation

```bash
# Clone or copy the project
cd portfolio/realtime_ai_pipeline

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Basic demo (10 seconds, 1 event/sec)
PYTHONPATH=src python examples/run_pipeline.py --duration 10

# Custom configuration
PYTHONPATH=src python examples/run_pipeline.py --duration 30 --rate 2.0

# With configuration file
PYTHONPATH=src python examples/run_pipeline.py --config config.yaml
```

### Expected Output

```
🚀 Setting up Real-time AI Pipeline...
✅ Configuration loaded
✅ Logging setup complete
✅ Pipeline created

🎯 Starting Real-time AI Pipeline Demo
============================================================
Pipeline initialized
Components starting
Streaming events
Processing features
Running AI inference
✅ Pipeline started successfully

🎯 PREDICTION RESULT [14:32:15]
   Event ID: abc123-def456
   Prediction: 0.7421
   Confidence: 0.6834
   Total Latency: 0.45ms
   Processing Breakdown:
     feature_processing_ms: 0.12ms
     inference_ms: 0.28ms
     publishing_ms: 0.05ms
--------------------------------------------------

FINAL PIPELINE SUMMARY
----------------------
Events Processed:     10
Features Generated:   10
Predictions Made:     10
Results Published:    10
Errors:               0
Average Latency:      0.33ms
```

## Configuration

The pipeline supports flexible configuration via YAML, JSON, or environment variables:

```yaml
data_source:
  event_rate: 2.0
  event_types: ["sensor", "user_action", "system_event"]

processor:
  features:
    statistical: true
    temporal: true
    categorical: true
  engineering:
    window_sizes: [5, 10, 20]

inference:
  model:
    type: "mock"  # mock, sklearn, onnx, tensorflow, pytorch
    input_size: 10
    confidence_threshold: 0.5

publisher:
  console: true
  file: true
  file_path: "results.jsonl"
  metrics: true
  metrics_port: 8080
```

## Data Flow

### 1. Event Generation
- Simulated data sources generate events at configurable rates
- Event types: sensor readings, user actions, system metrics
- Each event contains structured data with unique IDs and timestamps

### 2. Feature Processing
- **Statistical Features**: mean, std, min, max, percentiles, skewness, kurtosis
- **Temporal Features**: time-based patterns, trends, inter-event times
- **Categorical Features**: one-hot encoding, categorical counts
- **Advanced Features**: rolling windows, lag features, frequency analysis

### 3. AI Inference
- **Mock Model**: Linear model with configurable noise for testing
- **Scikit-learn**: Random forest, logistic regression with confidence scores
- **ONNX**: High-performance inference with pre-trained models
- **TensorFlow/PyTorch**: Deep learning model support with GPU acceleration

### 4. Result Publishing
- **Console**: Real-time formatted output for monitoring
- **File**: JSONL format for downstream processing and analysis
- **Webhook**: HTTP callbacks for system integration
- **Metrics**: HTTP endpoint for Prometheus-style monitoring

## Performance

The pipeline is optimized for production workloads:

- **Throughput**: 1000+ events/second (depending on hardware and model complexity)
- **Latency**: Sub-10ms average processing time for mock models
- **Memory**: Efficient sliding windows and circular buffers
- **Scalability**: Horizontal scaling via distributed message bus

## Use Cases

This architecture is suitable for various real-time AI applications:

- **IoT Sensor Processing**: Real-time sensor data analysis and anomaly detection
- **User Behavior Analytics**: Clickstream analysis and personalization
- **System Monitoring**: Log analysis and predictive maintenance
- **Financial Processing**: Real-time risk assessment and fraud detection
- **Industrial AI**: Manufacturing quality control and process optimization

## Development

### Running Tests

```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src
```

### Development Mode

```bash
# Enable debug logging
export PIPELINE_LOG_LEVEL=DEBUG

# Run with higher event rate
PYTHONPATH=src python examples/run_pipeline.py --rate 5.0 --duration 30
```

### Adding Components

The modular design makes it easy to extend the pipeline:

```python
# Custom data source
class CustomDataSource(DataSource):
    async def _generate_events(self):
        # Custom event generation logic
        pass

# Custom model adapter
class CustomModelAdapter(ModelAdapter):
    def predict(self, input_data):
        # Custom model inference
        pass
```

## Production Deployment

### Environment Variables

```bash
# Configuration
export PIPELINE_EVENT_RATE=10.0
export MODEL_TYPE=onnx
export MODEL_PATH=/models/model.onnx

# Outputs
export ENABLE_CONSOLE=false
export ENABLE_FILE=true
export WEBHOOK_URL=https://api.example.com/webhook
export METRICS_PORT=9090
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY src/ ./src/
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "examples/run_pipeline.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-pipeline
  template:
    metadata:
      labels:
        app: ai-pipeline
    spec:
      containers:
      - name: pipeline
        image: ai-pipeline:latest
        env:
        - name: PIPELINE_EVENT_RATE
          value: "5.0"
```

## Monitoring and Observability

The pipeline includes comprehensive monitoring:

- **Performance Metrics**: Latency, throughput, error rates by stage
- **Component Health**: Service health checks and dependency monitoring
- **Resource Usage**: Memory, CPU utilization, and queue depths
- **Business Metrics**: Prediction accuracy, confidence distributions

Access metrics at `http://localhost:8080/metrics` (when enabled)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is provided as a demonstration of real-time AI pipeline architecture for educational and portfolio purposes.

## Acknowledgments

This architecture is inspired by production event-driven systems and real-time AI processing patterns used in modern data platforms.
