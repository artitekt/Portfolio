"""
Data ingestion components.
"""

from .data_source import DataSource, FileDataSource, APIDataSource
from .stream_producer import StreamProducer, BackpressureProducer, MetricsProducer

__all__ = [
    "DataSource",
    "FileDataSource", 
    "APIDataSource",
    "StreamProducer",
    "BackpressureProducer",
    "MetricsProducer"
]
