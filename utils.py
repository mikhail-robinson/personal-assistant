import time


def generate_session_id():
    """Generate a unique session ID for tracking interactions"""
    return f"session_{int(time.time())}"
