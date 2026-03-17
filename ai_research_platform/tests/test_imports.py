def test_project_imports():
    """Test that ai_research_platform package imports correctly."""
    try:
        from . import src
        print("✅ ai_research_platform package imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_project_imports()
