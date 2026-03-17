#!/usr/bin/env python3
"""
Live Data Module for Autonomous AI System

This module handles real-time data ingestion from external APIs.
Provides LiveDataClient class for fetching cryptocurrency price data.
"""

import time
import json
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Optional


class LiveDataClient:
    """Client for fetching real-time cryptocurrency data from CoinGecko API."""
    
    def __init__(self):
        """Initialize the live data client."""
        self.api_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        self.last_known_price = None
        self.last_fetch_time = None
    
    def _fetch_from_api(self) -> Optional[Dict]:
        """Fetch data from CoinGecko API."""
        try:
            request = urllib.request.Request(
                self.api_url,
                headers={'User-Agent': 'Autonomous-AI-System/1.0'}
            )
            
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    return data
                else:
                    print(f"[Error] API returned status code: {response.status}")
                    return None
                    
        except urllib.error.URLError as e:
            print(f"[Error] Network error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"[Error] JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"[Error] Unexpected error: {e}")
            return None
    
    def get_event(self) -> Dict:
        """
        Fetch real-time price data and return as event format.
        
        Returns:
            dict: Event with price, asset, and timestamp
        """
        # Try to fetch from API
        data = self._fetch_from_api()
        
        # If API fails, retry once
        if data is None:
            print("[Retry] First attempt failed, retrying...")
            time.sleep(0.5)
            data = self._fetch_from_api()
        
        # Process successful response
        if data and 'bitcoin' in data and 'usd' in data['bitcoin']:
            price = float(data['bitcoin']['usd'])
            self.last_known_price = price
            self.last_fetch_time = datetime.now()
            
            print(f"[Data] Fetched BTC price: {price}")
            
            return {
                "price": price,
                "asset": "bitcoin",
                "timestamp": self.last_fetch_time.isoformat()
            }
        
        # Fallback to last known price
        elif self.last_known_price is not None:
            print(f"[Fallback] Using last known price: {self.last_known_price}")
            return {
                "price": self.last_known_price,
                "asset": "bitcoin",
                "timestamp": datetime.now().isoformat()
            }
        
        # No data available
        else:
            print("[Error] No price data available")
            return {
                "price": 0.0,
                "asset": "bitcoin",
                "timestamp": datetime.now().isoformat()
            }


# Test the client when run directly
if __name__ == "__main__":
    client = LiveDataClient()
    
    print("Testing LiveDataClient...")
    print("=" * 40)
    
    # Test multiple fetches
    for i in range(3):
        print(f"\nTest {i+1}:")
        event = client.get_event()
        print(f"Event: {event}")
        
        if i < 2:  # Add delay between tests
            time.sleep(1)
