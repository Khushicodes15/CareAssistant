import socket

def is_connected() -> bool:
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except socket.error:
        return False

def get_internet_status() -> dict:
    connected = is_connected()
    return {
        "connected": connected,
        "message": "Internet connection available" if connected else "No internet connection"
    }