#!/usr/bin/env python3
"""
Live Data Module for Autonomous AI System

This module handles real-time data ingestion from external APIs.
Provides LiveDataClient class for fetching cryptocurrency price data.
"""

import time
import json
import requests
from datetime import datetime
from typing import Dict, Optional


class LiveDataClient:
    """Client for fetching real-time cryptocurrency data from Binance API."""
    
    def __init__(self):
        """Initialize the live data client."""
        self.api_url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    
    def _fetch_from_api(self) -> Optional[float]:
        """Fetch data from Binance API."""
        try:
            response = requests.get(self.api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    return float(data['price'])
                else:
                    print("[Data] Invalid response format from Binance API")
                    return None
            else:
                print(f"[Data] API request failed with status code: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"[Data] API request failed or rate limited")
            print(f"[Data] Suggestion: reduce request frequency or upgrade API access")
            return None
        except (ValueError, KeyError) as e:
            print(f"[Data] Error parsing API response: {e}")
            return None
        except Exception as e:
            print(f"[Data] Unexpected error: {e}")
            return None
    
    def get_event(self) -> Optional[Dict]:
        """
        Fetch real-time price data and return as event format.
        
        Returns:
            dict: Event with price, asset, and timestamp, or None if data unavailable
        """
        # Try to fetch from API
        price = self._fetch_from_api()
        
        # If API fails, return None (no fallback or retry as per requirements)
        if price is None:
            return None
        
        # Successful response
        print(f"[Data] BTC price: {price}")
        
        return {
            "price": price,
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
        if event:
            print(f"Event: {event}")
        else:
            print("Event: None (API failed)")
        
        if i < 2:  # Add delay between tests
            time.sleep(1)
