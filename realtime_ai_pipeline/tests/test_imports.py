def test_project_imports():
    """Test that realtime_ai_pipeline package imports correctly."""
    try:
        from . import src
        print("✅ realtime_ai_pipeline package imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_project_imports()
